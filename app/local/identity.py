"""Who is calling? Verifies the account service's access token and names the tenant.

The account service (ideascribe-api) signs short-lived access tokens with an Ed25519 key and publishes the public
keys. The engine holds no secret: it checks the signature against those keys (cached, with a copy on disk so
verification keeps working offline), and requires the right issuer, audience, expiry and token type.

Only the verified `sub` claim is used, never an e-mail address or a header the caller can choose.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jwt
from jwt import PyJWK

log = logging.getLogger(__name__)

JWKS_TTL_S = 300
REFETCH_MIN_INTERVAL_S = 30


class IdentityError(Exception):
    """The identity token is missing, malformed, expired, or not ours to trust."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class Identity:
    subject: str
    role: str | None


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
        return Identity(subject=claims["sub"], role=claims.get("role"))


async def tenant_for(app, token: str | None) -> str | None:
    """The tenant key for a request's identity token, or None when isolation is off (one shared scope).

    Raises IdentityError when isolation is on and the token is missing or not valid."""
    from . import tenancy

    verifier = getattr(app.state, "identity", None)
    if verifier is None:
        return None
    identity = await verifier.verify(token or "")
    return tenancy.tenant_key(identity.subject)
