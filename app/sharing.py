import base64
import binascii
import secrets
from uuid import UUID

from starlette.exceptions import HTTPException
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from app.config import AppSettings


class ShareProtectionMiddleware:
    def __init__(self, app, settings: AppSettings, max_body_bytes: int = 5 * 1024 * 1024):
        self.app = app
        self.settings = settings
        self.max_body_bytes = max_body_bytes

    def authenticated(self, header: str) -> bool:
        try:
            scheme, encoded = header.split(" ", 1)
            if scheme.lower() != "basic":
                return False
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeError, binascii.Error):
            return False
        username_ok = secrets.compare_digest(username.encode(), self.settings.share_username.encode())
        password_ok = secrets.compare_digest(password.encode(), self.settings.share_password.encode())
        return username_ok and password_ok

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if self.settings.public_share:
            original_send = send

            async def protected_send(message):
                if message["type"] == "http.response.start":
                    response_headers = MutableHeaders(scope=message)
                    response_headers["X-Content-Type-Options"] = "nosniff"
                    response_headers["X-Frame-Options"] = "DENY"
                    response_headers["Referrer-Policy"] = "same-origin"
                    response_headers["Cache-Control"] = "no-store"
                await original_send(message)

            send = protected_send
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        if self.settings.public_share:
            if not self.authenticated(headers.get("authorization", "")):
                return await JSONResponse(
                    status_code=401, content={"detail": "Sign in to access TutorFlow."},
                    headers={"WWW-Authenticate": 'Basic realm="TutorFlow", charset="UTF-8"', "Cache-Control": "no-store"},
                )(scope, receive, send)
            if scope["method"] not in ("GET", "HEAD", "OPTIONS") and headers.get("origin"):
                expected = f"{scope['scheme']}://{headers.get('host', '')}"
                if headers["origin"] != expected:
                    return await JSONResponse(status_code=403, content={"detail": "Cross-origin changes are disabled in sharing mode."})(scope, receive, send)

        if scope["method"] in ("POST", "PUT", "PATCH"):
            chunks = []
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > self.max_body_bytes:
                    return await JSONResponse(status_code=413, content={"detail": "Request body is too large (maximum 5 MB)."})(scope, receive, send)
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            return await self.app(scope, buffered_receive, send)
        return await self.app(scope, receive, send)


class FrontendFiles(StaticFiles):
    @staticmethod
    def is_lesson_page(path: str) -> bool:
        parts = path.replace("\\", "/").strip("/").split("/")
        if len(parts) != 2 or parts[0] != "lessons":
            return False
        try:
            return str(UUID(parts[1])) == parts[1].lower()
        except ValueError:
            return False

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code == 404 and (path in ("studio", "library") or self.is_lesson_page(path)):
                return await super().get_response("index.html", scope)
            raise
