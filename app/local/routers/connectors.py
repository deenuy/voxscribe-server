"""Connector endpoints: Google account and token management."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.local import connector_tokens, store_misc
from app.local.models import GoogleAccount, GoogleConnector, GoogleConnectorPutRequest

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


@router.get("/google")
async def get_google_connector(request: Request) -> dict:
    """Get Google connector status."""
    store = request.app.state.store

    async with store.pool.connection() as conn:
        account_row = await store_misc.get_connector_account(conn, "google")

    # Check if token is available and not expired
    token_tuple = await connector_tokens.get_google_access_token()
    token_expires_at = None
    if token_tuple:
        _, expires_at = token_tuple
        token_expires_at = iso_z(expires_at)

    connected = account_row is not None and token_tuple is not None

    account = None
    scopes = []
    if account_row:
        account = GoogleAccount(
            sub=account_row["sub"],
            email=account_row["email"],
            name=account_row["name"],
            picture=account_row.get("picture"),
        )
        scopes = account_row.get("scopes") or []

    connector = GoogleConnector(
        connected=connected,
        account=account,
        scopes=scopes,
        tokenExpiresAt=token_expires_at,
        lastSyncAt=iso_z(account_row.get("last_sync_at")) if account_row and account_row.get("last_sync_at") else None,
    )
    return connector.model_dump(exclude_none=False)


@router.put("/google")
async def put_google_connector(request: Request, body: GoogleConnectorPutRequest) -> dict:
    """Store Google connector account and token.

    Body:
        account: {sub, email, name, picture?}
        accessToken: short-lived JWT
        expiresAt: ISO 8601 expiry time
        scopes: list of granted scopes
    """
    store = request.app.state.store

    # Store token in memory
    await connector_tokens.set_google_token(body.accessToken, body.expiresAt)

    # Store account metadata in database
    async with store.pool.connection() as conn:
        # Check if this is a new sign-in (no account) or scope addition
        existing = await store_misc.get_connector_account(conn, "google")
        is_new = existing is None
        scopes_grew = False
        if existing:
            existing_scopes = set(existing.get("scopes") or [])
            new_scopes = set(body.scopes)
            scopes_grew = len(new_scopes - existing_scopes) > 0

        # Store account
        await store_misc.put_connector_account(
            conn,
            "google",
            body.account.sub,
            body.account.email,
            body.account.name,
            body.account.picture,
            body.scopes,
            body.expiresAt,
        )

        # Record auth event
        if is_new:
            await store_misc.create_auth_event(
                conn,
                "google_sign_in",
                {"sub": body.account.sub, "email": body.account.email},
            )
        elif scopes_grew:
            await store_misc.create_auth_event(
                conn,
                "google_scope_added",
                {"sub": body.account.sub, "scopes": body.scopes},
            )

    # Publish token_needed event to notify others (for testing)
    # (The token just arrived, so no need to request, but signal the state change)

    # Return current state
    connector = GoogleConnector(
        connected=True,
        account=body.account,
        scopes=body.scopes,
        tokenExpiresAt=body.expiresAt,
    )
    return connector.model_dump(exclude_none=True)


@router.delete("/google")
async def delete_google_connector(request: Request, removeContent: bool = False) -> JSONResponse:
    """Disconnect Google account.

    Query params:
        removeContent: if true, delete imported Google Drive sources and calendar events

    Returns 204.
    """
    store = request.app.state.store

    async with store.pool.connection() as conn:
        # Get account to record sign-out event
        account_row = await store_misc.get_connector_account(conn, "google")

        if removeContent and account_row:
            # Delete sources of kind 'drive' (cascade deletes chunks)
            await conn.execute("DELETE FROM sources WHERE kind = %s", ("drive",))

            # Delete calendar events and associated meetings
            cursor = await conn.execute("SELECT id FROM calendar_events")
            event_ids = [row[0] for row in await cursor.fetchall()]
            for event_id in event_ids:
                await conn.execute("DELETE FROM meetings WHERE calendar_event_id = %s", (event_id,))
            await conn.execute("DELETE FROM calendar_events")

        # Delete account metadata
        await store_misc.delete_connector_account(conn, "google")

        # Record sign-out event
        if account_row:
            await store_misc.create_auth_event(
                conn,
                "google_sign_out",
                {"sub": account_row["sub"], "email": account_row["email"]},
            )

        await conn.commit()

    # Clear token
    await connector_tokens.clear_google_token()

    return JSONResponse(status_code=204, content={})


@router.get("/google/events")
async def list_google_auth_events(request: Request, limit: int = 50) -> dict:
    limit = max(1, min(int(limit), 200))  # a caller cannot ask the database for a negative or enormous page
    """List recent Google auth events."""
    store = request.app.state.store

    async with store.pool.connection() as conn:
        events_list = await store_misc.list_auth_events(conn, limit)

    return {
        "items": [e.model_dump() for e in events_list],
    }
