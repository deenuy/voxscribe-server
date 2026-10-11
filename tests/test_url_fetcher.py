"""Tests for URL fetching with SSRF protection."""

import pytest
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

from app.local import url_fetcher
from app.local.url_fetcher import fetch_url, SSRFGuard


class FakeHTTPHandler(BaseHTTPRequestHandler):
    """Fake HTTP server for testing."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Test content</body></html>")
        elif self.path == "/oversized":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "999999999")
            self.end_headers()
            # Send partial content
            self.wfile.write(b"x" * (11 * 1024 * 1024))
        elif self.path == "/wrong-type":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(b"binary data")
        elif self.path == "/redirect-blocked":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:8888/blocked")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def start_fake_server(port: int):
    """Start a fake HTTP server in a background thread."""
    server = HTTPServer(("127.0.0.1", port), FakeHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Wait for server to start
    return server


@pytest.mark.asyncio
async def test_fetch_valid_url():
    """Test fetching a valid URL."""
    server = start_fake_server(9999)
    try:
        # Override API base for test
        original_override = url_fetcher.API_BASE_URL_OVERRIDE
        url_fetcher.API_BASE_URL_OVERRIDE = "http://127.0.0.1:9999"

        # Allow test override in SSRF guard
        original_blocked = SSRFGuard.is_blocked

        def mock_is_blocked(hostname):
            if hostname == "127.0.0.1":
                return False  # Allow for test
            return original_blocked(hostname)

        SSRFGuard.is_blocked = mock_is_blocked

        try:
            content = await fetch_url("http://127.0.0.1:9999/ok")
            assert "Test content" in content
        finally:
            SSRFGuard.is_blocked = original_blocked
            url_fetcher.API_BASE_URL_OVERRIDE = original_override
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_fetch_oversized_url():
    """Test that oversized responses are rejected."""
    server = start_fake_server(10000)
    try:
        original_override = url_fetcher.API_BASE_URL_OVERRIDE
        url_fetcher.API_BASE_URL_OVERRIDE = "http://127.0.0.1:10000"

        original_blocked = SSRFGuard.is_blocked

        def mock_is_blocked(hostname):
            if hostname == "127.0.0.1":
                return False
            return original_blocked(hostname)

        SSRFGuard.is_blocked = mock_is_blocked

        try:
            with pytest.raises((ValueError, Exception)):  # Could be httpx error or size check
                await fetch_url(
                    "http://127.0.0.1:10000/oversized", max_size_mb=5
                )
        finally:
            SSRFGuard.is_blocked = original_blocked
            url_fetcher.API_BASE_URL_OVERRIDE = original_override
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_fetch_wrong_content_type():
    """Test that wrong content type is rejected."""
    server = start_fake_server(10001)
    try:
        original_override = url_fetcher.API_BASE_URL_OVERRIDE
        url_fetcher.API_BASE_URL_OVERRIDE = "http://127.0.0.1:10001"

        original_blocked = SSRFGuard.is_blocked

        def mock_is_blocked(hostname):
            if hostname == "127.0.0.1":
                return False
            return original_blocked(hostname)

        SSRFGuard.is_blocked = mock_is_blocked

        try:
            with pytest.raises(ValueError, match="Unsupported content type"):
                await fetch_url("http://127.0.0.1:10001/wrong-type")
        finally:
            SSRFGuard.is_blocked = original_blocked
            url_fetcher.API_BASE_URL_OVERRIDE = original_override
    finally:
        server.shutdown()


def test_ssrf_guard_blocks_loopback():
    """Test that SSRF guard blocks loopback addresses."""
    assert SSRFGuard.is_blocked("127.0.0.1") is True
    assert SSRFGuard.is_blocked("localhost") is True


def test_ssrf_guard_blocks_private():
    """Test that SSRF guard blocks private addresses."""
    assert SSRFGuard.is_blocked("10.0.0.1") is True
    assert SSRFGuard.is_blocked("192.168.1.1") is True
    assert SSRFGuard.is_blocked("172.16.0.1") is True


@pytest.mark.asyncio
async def test_invalid_url_scheme():
    """Test that non-HTTP URLs are rejected."""
    with pytest.raises(ValueError, match="Only http and https"):
        await fetch_url("file:///etc/passwd")

    with pytest.raises(ValueError, match="Only http and https"):
        await fetch_url("ftp://example.com")
