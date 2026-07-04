"""
Minimal JWT issuing/verification. There's no user database here — this is
meant to sit behind whatever auth service actually owns accounts in a real
deployment, so /token just issues a signed token for a client_id. The
websocket endpoint requires a valid token before it'll accept a connection.
"""
import time

from fastapi import HTTPException, Query, status
from jose import JWTError, jwt

from app.config import settings


def create_access_token(client_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": client_id,
        "iat": now,
        "exp": now + settings.jwt_expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token")
    return payload["sub"]


async def ws_auth(token: str = Query(...)) -> str:
    """FastAPI dependency for the websocket route — token comes in as a
    query param since websocket handshakes don't carry custom headers
    reliably through every proxy/browser combination."""
    return decode_token(token)
