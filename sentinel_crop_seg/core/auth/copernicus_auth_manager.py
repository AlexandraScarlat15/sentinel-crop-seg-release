"""
Copernicus Auth Manager

Obtains OAuth2 access tokens for Copernicus Data Space Ecosystem (CDSE).

Credentials are read from environment variables:
- COPERNICUS_USERNAME
- COPERNICUS_PASSWORD
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class CopernicusToken:
    access_token: str
    expires_at_epoch: float


class CopernicusAuthManager:
    """
    OAuth2 token provider for CDSE.

    This class caches the token in memory and refreshes it when needed.
    """

    TOKEN_URL = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    CLIENT_ID = "cdse-public"

    def __init__(self) -> None:
        self._token: Optional[CopernicusToken] = None

    def get_access_token(self) -> str:
        """
        Return a valid access token. Refreshes the token if expired.

        Returns
        -------
        str
            OAuth2 access token.
        """
        if self._token and not self._is_expired(self._token):
            return self._token.access_token

        self._token = self._request_new_token()
        return self._token.access_token

    def _is_expired(self, token: CopernicusToken) -> bool:
        """
        Check whether a cached token is expired (with a small safety margin).
        """
        safety_margin_seconds = 30.0
        return time.time() >= (token.expires_at_epoch - safety_margin_seconds)

    def _request_new_token(self) -> CopernicusToken:
        """
        Request a new token from the CDSE identity service.
        """
        username = os.getenv("COPERNICUS_USERNAME")
        password = os.getenv("COPERNICUS_PASSWORD")

        if not username or not password:
            raise RuntimeError(
                "Copernicus credentials not found in environment variables. "
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD."
            )

        data = {
            "grant_type": "password",
            "client_id": self.CLIENT_ID,
            "username": username,
            "password": password,
        }

        resp = requests.post(self.TOKEN_URL, data=data, timeout=60)
        resp.raise_for_status()

        payload = resp.json()
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")

        if not access_token or not expires_in:
            raise RuntimeError("Invalid token response from Copernicus identity service.")

        return CopernicusToken(
            access_token=access_token,
            expires_at_epoch=time.time() + float(expires_in),
        )
