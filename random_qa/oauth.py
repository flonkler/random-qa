import asyncio
import base64
import httpx
import os
import threading
from datetime import datetime, timedelta

from typing import Generator, AsyncGenerator


class CustomAccessTokenAuth(httpx.Auth):
    def __init__(self, issuer_url: str, client_id: str, client_secret: str):
        self.issuer_url = issuer_url
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = None
        self._expires_at = None
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

    def _build_access_token_request(self) -> httpx.Request:
        basic_auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("utf-8")
        return httpx.Request(
            "POST", self.issuer_url,
            headers={"Authorization": f"Basic {basic_auth}"},
            data={"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret}
        )
    
    def _update_access_token(self, issuer_response: httpx.Response) -> None:
        data: dict = issuer_response.json()
        # Update access token and timestamp of expiration
        self._access_token = data.get("access_token")
        self._expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 0))

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        with self._sync_lock:
            if self._expires_at is None or self._expires_at < datetime.now():
                issuer_response = yield self._build_access_token_request()
                issuer_response.read()  # must be called to avoid ResponseNotRead error
                self._update_access_token(issuer_response)
        
        request.headers["Authorization"] = f"Bearer {self._access_token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        async with self._async_lock:
            if self._expires_at is None or self._expires_at < datetime.now():
                issuer_response = yield self._build_access_token_request()
                await issuer_response.aread()  # must be called to avoid ResponseNotRead error
                self._update_access_token(issuer_response)
        
        request.headers["Authorization"] = f"Bearer {self._access_token}"
        yield request

_oauth_sync_client = None
_oauth_async_client = None

def get_oauth_clients(**client_kwargs) -> tuple[httpx.Client, httpx.AsyncClient]:
    global _oauth_sync_client, _oauth_async_client
    if _oauth_sync_client is None and _oauth_async_client is None:
        issuer_url = os.getenv("OAUTH_ISSUER_URL")
        client_id = os.getenv("OAUTH_CLIENT_ID")
        client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        auth = CustomAccessTokenAuth(issuer_url, client_id, client_secret)
        
        _oauth_sync_client = httpx.Client(auth=auth, **client_kwargs)
        _oauth_async_client = httpx.AsyncClient(auth=auth, **client_kwargs)

    return _oauth_sync_client, _oauth_async_client
