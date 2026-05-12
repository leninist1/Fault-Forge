"""HTTP helpers for local Train-Ticket calls.

The host environment may define HTTP_PROXY/HTTPS_PROXY for external traffic. Python requests
honors those variables for localhost unless NO_PROXY is set, which can turn healthy local
gateway calls into proxy-generated 502 responses. Local benchmark calls should bypass proxies.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests


def should_bypass_proxy(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local")


def session_for_url(url: str) -> requests.Session:
    session = requests.Session()
    if should_bypass_proxy(url) and os.environ.get("FAULTFORGE_BYPASS_LOCAL_PROXY", "1").lower() not in {"0", "false", "no"}:
        session.trust_env = False
    return session


def request(method: str, url: str, **kwargs):
    with session_for_url(url) as session:
        return session.request(method, url, **kwargs)


def get(url: str, **kwargs):
    return request("GET", url, **kwargs)


def post(url: str, **kwargs):
    return request("POST", url, **kwargs)
