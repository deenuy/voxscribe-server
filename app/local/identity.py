"""Who is calling? Verifies the account service's access token and names the tenant.

The account service signs short-lived access tokens with an Ed25519 key and publishes the public
keys. The engine holds no secret: it checks the signature against those keys (cached, with a copy on disk so
verification keeps working offline), and requires the right issuer, audience, expiry and token type.

The verified `sub` claim names the tenant, never an e-mail address or a header the caller can choose. The person
stays the unit of isolation (D14). Four more claims, verified with the same signature, are used for access decisions
and never to choose a schema:

  org      the organization the person acts for. Missing: none (a person on their own)
  product  the product the token was issued for (a profile mode: personal, ideascribe, senior_care, classroom).
           Missing: the token is not tied to a product. Present and not the engine's loaded profile: refused
           (product_mismatch)
  device   the device the token was issued for. Missing: any device. Present and not this engine's device:
           the token cannot bind or unbind the engine's audio device (device_mismatch)
  role     what the person may do here (ROLES below). Missing or unknown: the least, no device control

A claim that is present must be a non-empty string; anything else makes the token not valid.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jwt
from jwt import PyJWK

from app.local.errors import EngineError

log = logging.getLogger(__name__)

JWKS_TTL_S = 300
REFETCH_MIN_INTERVAL_S = 30


# Roles. Kept to what the engine decides today; a role grants nothing else.
#   owner, admin, member   use the engine, and bind (arm, trigger) and unbind (disarm) the engine's audio device
#   user                   the account service's role for a person on their own: the same as member
#   viewer                 use the engine's request routes, but not bind or unbind the audio device
# A missing role, or one not named here, is treated as viewer.
DEVICE_CONTROL_ROLES = frozenset({"owner", "admin", "member", "user"})
ROLES = DEVICE_CONTROL_ROLES | {"viewer"}

_OPTIONAL_CLAIMS = ("org", "product", "device", "role")


class AccessDenied(EngineError):
    """A verified person asked for something their claims do not allow. Named by code, answered 403."""

    def __init__(self, code: str, detail: str):
        super().__init__(code, detail, 403)


class IdentityError(Exception):
    """The identity token is missing, malformed, expired, or not ours to trust."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class Identity:
    subject: str
    role: str | None = None
    organization: str | None = None
    product: str | None = None
    device: str | None = None

    @property
    def may_control_device(self) -> bool:
        return self.role in DEVICE_CONTROL_ROLES


class IdentityVerifier:
    def __init__(self, *, issuer: str, audience: str, jwks_url: str, jwks_file: str | None = None, fetch=None):
        self.issuer = issuer
        self.audience = audience
        self.jwks_url = jwks_url
        self.jwks_file = Path(jwks_file) if jwks_file else None
        self._fetch = fetch or self._http_fetch
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at = 0.0
        self._attempted_at = 0.0
        if self.jwks_file and self.jwks_file.exists():
            try:
                self._load(json.loads(self.jwks_file.read_text()))
            except Exception:  # noqa: BLE001 - a bad cache file only costs a fetch
                log.warning("Ignoring unreadable identity key cache %s", self.jwks_file)

    async def _http_fetch(self) -> dict:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(self.jwks_url)
            res.raise_for_status()
            return res.json()

    def _load(self, document: dict) -> None:
        keys = {}
        for jwk in document.get("keys", []):
            if jwk.get("kid") and jwk.get("kty") == "OKP" and jwk.get("crv") == "Ed25519":
                keys[jwk["kid"]] = PyJWK.from_dict(jwk)
        if not keys:
            raise ValueError("no usable signing keys")
        self._keys = keys

    async def _refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self._keys and now - self._fetched_at < JWKS_TTL_S:
            return
        if now - self._attempted_at < REFETCH_MIN_INTERVAL_S and self._keys:
            return  # do not hammer the account service for a token with an unknown key
        self._attempted_at = now
        try:
            document = await self._fetch()
            self._load(document)
            self._fetched_at = time.monotonic()
            if self.jwks_file:
                try:
                    self.jwks_file.parent.mkdir(parents=True, exist_ok=True)
                    self.jwks_file.write_text(json.dumps(document))
                except OSError:
                    log.warning("Could not write the identity key cache %s", self.jwks_file)
        except Exception as exc:  # noqa: BLE001
            if not self._keys:
                raise IdentityError("the identity service's keys are not available") from exc
            log.warning("Could not refresh identity keys, using the cached ones: %s", exc)

    async def verify(self, token: str) -> Identity:
        if not token:
            raise IdentityError("identity token required")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise IdentityError("identity token is not valid") from None
        if header.get("alg") != "EdDSA" or not header.get("kid"):
            raise IdentityError("identity token is not valid")
        await self._refresh()
        key = self._keys.get(header["kid"])
        if key is None:
            await self._refresh(force=True)  # a rotated key: look once more
            key = self._keys.get(header["kid"])
        if key is None:
            raise IdentityError("identity token is signed with an unknown key")
        try:
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["EdDSA"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "sub", "typ", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError:
            raise IdentityError("identity token has expired") from None
        except jwt.PyJWTError:
            raise IdentityError("identity token is not valid") from None
        if claims.get("typ") != "access" or not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise IdentityError("identity token is not an access token")
        for name in _OPTIONAL_CLAIMS:
            value = claims.get(name)
            if value is not None and (not isinstance(value, str) or not value):
                raise IdentityError("identity token is not valid")
        return Identity(subject=claims["sub"], role=claims.get("role"), organization=claims.get("org"),
                        product=claims.get("product"), device=claims.get("device"))


def check_product(app, identity: Identity) -> None:
    """A token issued for one product does not open an engine that serves another."""
    if identity.product is None:
        return
    loaded = getattr(getattr(app.state, "profile", None), "mode", None)
    if identity.product != loaded:
        raise AccessDenied("product_mismatch", "this credential was issued for another product than the one this engine serves")


def require_device_control(identity: Identity | None, device: str) -> None:
    """Binding or unbinding the engine's audio device needs a role that may do it, and a token that is not tied
    to another device. None (isolation off: no person on the request) is not checked, as before."""
    if identity is None:
        return
    if not identity.may_control_device:
        raise AccessDenied("role_not_permitted", "this role may not arm or disarm the audio device")
    if identity.device is not None and identity.device != device:
        raise AccessDenied("device_mismatch", "this credential was issued for another device")


async def caller_for(app, token: str | None) -> Identity | None:
    """The verified person behind an identity token, or None when isolation is off (one shared scope).

    Raises IdentityError when isolation is on and the token is missing or not valid, and AccessDenied when it is
    valid but issued for another product."""
    verifier = getattr(app.state, "identity", None)
    if verifier is None:
        return None
    identity = await verifier.verify(token or "")
    check_product(app, identity)
    return identity


async def tenant_for(app, token: str | None) -> str | None:
    """The tenant key for a request's identity token, or None when isolation is off. Raises as caller_for does."""
    from . import tenancy

    identity = await caller_for(app, token)
    return None if identity is None else tenancy.tenant_key(identity.subject)
