"""Google Drive v3 fetching with token management."""

import httpx

from app.local import events
from app.local.errors import ConnectorTokenExpiredError

# Module variable for test override
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


async def get_access_token() -> str:
    """Get Google access token from connector tokens."""
    from app.local.connector_tokens import get_google_access_token
    token = get_google_access_token()
    if not token:
        raise ConnectorTokenExpiredError("No Google access token available")
    return token


async def list_folder_children(folder_id: str, page_token: str | None = None) -> dict:
    """List children of a Drive folder with pagination.

    Args:
        folder_id: Google Drive folder ID
        page_token: Pagination token

    Returns:
        Dict with 'files' list and 'nextPageToken'
    """
    token = await get_access_token()

    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "pageSize": 100,
        "fields": "files(id,name,mimeType),nextPageToken",
    }
    if page_token:
        params["pageToken"] = page_token

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DRIVE_API_BASE}/files",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=10.0,
            )

            if response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired") from e
            raise


async def export_google_doc(file_id: str, mime_type: str = "text/plain") -> str:
    """Export a Google Doc as text.

    Args:
        file_id: Google Drive file ID
        mime_type: Export MIME type (text/plain or text/csv for Sheets)

    Returns:
        File content as string
    """
    token = await get_access_token()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DRIVE_API_BASE}/files/{file_id}/export",
                headers={"Authorization": f"Bearer {token}"},
                params={"mimeType": mime_type},
                timeout=10.0,
            )

            if response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired")

            response.raise_for_status()
            return response.text

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired") from e
            raise


async def download_file(file_id: str, max_size_mb: int = 10) -> bytes:
    """Download a file from Drive.

    Args:
        file_id: Google Drive file ID
        max_size_mb: Maximum file size in MB

    Returns:
        File content as bytes
    """
    token = await get_access_token()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DRIVE_API_BASE}/files/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"alt": "media"},
                timeout=10.0,
            )

            if response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired")

            response.raise_for_status()

            # Check size
            if len(response.content) > max_size_mb * 1024 * 1024:
                raise ValueError(f"File too large: {len(response.content)} bytes")

            return response.content

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                events.publish("connector.token_needed", {"connector": "google"})
                raise ConnectorTokenExpiredError("Google token expired") from e
            raise
