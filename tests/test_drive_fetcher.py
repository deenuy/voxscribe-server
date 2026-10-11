"""Tests for Drive fetching with fake Drive API."""

import pytest
from unittest.mock import patch
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json

from app.local import drive_fetcher
from app.local.errors import ConnectorTokenExpiredError


class FakeDriveHandler(BaseHTTPRequestHandler):
    """Fake Drive API server for testing."""

    def do_GET(self):
        """Handle GET requests."""
        auth = self.headers.get("Authorization", "")

        # Check token
        if not auth.startswith("Bearer test-token"):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid_token"}).encode())
            return

        if "/files" in self.path and "export" in self.path:
            # Export endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Exported Google Doc content")
        elif "/files" in self.path and "alt=media" in self.path:
            # Download endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Downloaded file content")
        elif "/files" in self.path:
            # List files endpoint
            response = {
                "files": [
                    {"id": "file-1", "name": "Document 1", "mimeType": "application/vnd.google-apps.document"},
                    {"id": "file-2", "name": "Sheet 1", "mimeType": "application/vnd.google-apps.spreadsheet"},
                ],
                "nextPageToken": None,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def start_fake_drive_server(port: int):
    """Start a fake Drive API server in a background thread."""
    server = HTTPServer(("127.0.0.1", port), FakeDriveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    return server


@pytest.mark.asyncio
async def test_list_folder_children():
    """Test listing Drive folder children."""
    server = start_fake_drive_server(9990)
    try:
        # Override API base and token getter
        original_api_base = drive_fetcher.DRIVE_API_BASE
        drive_fetcher.DRIVE_API_BASE = "http://127.0.0.1:9990"

        with patch("app.local.drive_fetcher.get_access_token") as mock_token:
            mock_token.return_value = "test-token"

            result = await drive_fetcher.list_folder_children("folder-1")
            assert "files" in result
            assert len(result["files"]) == 2
            assert result["files"][0]["name"] == "Document 1"

        drive_fetcher.DRIVE_API_BASE = original_api_base
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_export_google_doc():
    """Test exporting a Google Doc."""
    server = start_fake_drive_server(9991)
    try:
        original_api_base = drive_fetcher.DRIVE_API_BASE
        drive_fetcher.DRIVE_API_BASE = "http://127.0.0.1:9991"

        with patch("app.local.drive_fetcher.get_access_token") as mock_token:
            mock_token.return_value = "test-token"

            content = await drive_fetcher.export_google_doc("file-1")
            assert "Exported Google Doc content" in content

        drive_fetcher.DRIVE_API_BASE = original_api_base
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_download_file():
    """Test downloading a file from Drive."""
    server = start_fake_drive_server(9992)
    try:
        original_api_base = drive_fetcher.DRIVE_API_BASE
        drive_fetcher.DRIVE_API_BASE = "http://127.0.0.1:9992"

        with patch("app.local.drive_fetcher.get_access_token") as mock_token:
            mock_token.return_value = "test-token"

            content = await drive_fetcher.download_file("file-1")
            assert b"Downloaded file content" in content

        drive_fetcher.DRIVE_API_BASE = original_api_base
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_token_expired():
    """Test 401 response triggers token_needed event."""
    server = start_fake_drive_server(9993)
    try:
        original_api_base = drive_fetcher.DRIVE_API_BASE
        drive_fetcher.DRIVE_API_BASE = "http://127.0.0.1:9993"

        with patch("app.local.drive_fetcher.get_access_token") as mock_token:
            mock_token.return_value = "invalid-token"

            with pytest.raises(ConnectorTokenExpiredError):
                await drive_fetcher.list_folder_children("folder-1")

        drive_fetcher.DRIVE_API_BASE = original_api_base
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_no_token_available():
    """Test error when no token is available."""
    with patch("app.local.drive_fetcher.get_access_token") as mock_token:
        mock_token.side_effect = ConnectorTokenExpiredError("No token")

        with pytest.raises(ConnectorTokenExpiredError):
            await drive_fetcher.list_folder_children("folder-1")
