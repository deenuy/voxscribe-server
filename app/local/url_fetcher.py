"""URL content fetching with SSRF protection."""

import ipaddress
import socket
import tempfile
from pathlib import Path

import httpx

# Module variable for testing - allows test override
API_BASE_URL_OVERRIDE: str | None = None


class SSRFGuard:
    """SSRF protection: blocks private/reserved IP ranges."""

    # Blocked ranges: loopback, link-local, private, metadata
    BLOCKED_RANGES = [
        ipaddress.ip_network("127.0.0.0/8"),      # Loopback
        ipaddress.ip_network("169.254.0.0/16"),   # Link-local
        ipaddress.ip_network("10.0.0.0/8"),       # Private
        ipaddress.ip_network("172.16.0.0/12"),    # Private
        ipaddress.ip_network("192.168.0.0/16"),   # Private
        ipaddress.ip_network("224.0.0.0/4"),      # Multicast
        ipaddress.ip_network("240.0.0.0/4"),      # Reserved
        ipaddress.ip_network("::1/128"),          # IPv6 loopback
        ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
        ipaddress.ip_network("fc00::/7"),         # IPv6 private
    ]

    @staticmethod
    def is_blocked(hostname: str) -> bool:
        """Check if hostname resolves to a blocked address."""
        try:
            # Resolve hostname to all possible addresses
            addresses = socket.getaddrinfo(hostname, None)
            for addr_info in addresses:
                ip_str = addr_info[4][0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    for blocked_range in SSRFGuard.BLOCKED_RANGES:
                        if ip in blocked_range:
                            return True
                except ValueError:
                    pass
            return False
        except socket.gaierror:
            # DNS resolution failed - allow (will fail later during fetch)
            return False


async def fetch_url(
    url: str,
    timeout: float = 10.0,
    max_size_mb: int = 10,
    content_types: list[str] | None = None,
    max_redirects: int = 5,
) -> str:
    """Fetch URL content with SSRF and security checks.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_size_mb: Maximum content size in MB
        content_types: Allowed content types (e.g., ["text/html", "application/pdf"])
        max_redirects: Maximum number of redirects to follow

    Returns:
        Extracted text content

    Raises:
        ValueError: If URL fails security checks or content type invalid
        httpx.HTTPError: If fetch fails
    """
    # Default content types
    if content_types is None:
        content_types = ["text/html", "text/plain", "text/markdown", "application/pdf"]

    # Validate URL scheme
    if not url.startswith(("http://", "https://")):
        raise ValueError("Only http and https URLs are allowed")

    # Extract hostname and check SSRF
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid URL: no hostname")

        if SSRFGuard.is_blocked(hostname):
            raise ValueError(f"Blocked IP address: {hostname}")
    except Exception as e:
        if "Blocked IP" in str(e):
            raise
        raise ValueError(f"Invalid URL: {str(e)}") from e

    # Fetch with timeouts and size limits
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        try:
            response = await client.get(url, follow_redirects=max_redirects)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if not content_type or content_type not in content_types:
                raise ValueError(f"Unsupported content type: {content_type}")

            # Check content size
            content_length = int(response.headers.get("content-length", 0))
            if content_length > max_size_mb * 1024 * 1024:
                raise ValueError(f"Content too large: {content_length} bytes")

            # Check actual response size
            if len(response.content) > max_size_mb * 1024 * 1024:
                raise ValueError(f"Response too large: {len(response.content)} bytes")

            # Validate final URL is not blocked (after redirects)
            final_hostname = urlparse(str(response.url)).hostname
            if final_hostname and SSRFGuard.is_blocked(final_hostname):
                raise ValueError(f"Redirect to blocked address: {final_hostname}")

            # Parse content based on type using voxscribe-core
            from voxscribe.knowledge.parsers import parse

            if content_type == "text/html":
                # Save to temp file and parse
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name

                try:
                    parsed = parse(temp_path)
                    text = parsed.text
                finally:
                    Path(temp_path).unlink()
            elif content_type in ["text/plain", "text/markdown"]:
                text = response.text
            else:
                text = response.text

            return text

        except httpx.TimeoutException as e:
            raise ValueError(f"Request timeout: {str(e)}") from e
