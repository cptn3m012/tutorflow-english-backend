import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.comfy_client import PictureError
from app.comfy_runtime import comfy_runtime
from app.config import ImageSettings


class ComfyRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        runtime = self.root / "data" / "image-runtime"
        (runtime / "venv" / "Scripts").mkdir(parents=True)
        (runtime / "venv" / "Scripts" / "python.exe").touch()
        (runtime / "venv" / "bin").mkdir()
        (runtime / "venv" / "bin" / "python").touch()
        (runtime / "ComfyUI").mkdir()
        (runtime / "ComfyUI" / "main.py").touch()
        self.settings = ImageSettings(managed_runtime=True)

    def tearDown(self):
        self.directory.cleanup()

    async def test_preexisting_service_is_never_started_or_terminated(self):
        factory = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"system": {"comfyui_version": "test"}}))
        with patch("app.comfy_runtime.PROJECT_ROOT", self.root), \
             patch("app.comfy_runtime.httpx.AsyncClient", side_effect=lambda **kw: factory(transport=transport, **kw)), \
             patch("app.comfy_runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(PictureError, "already in use"):
                async with comfy_runtime(self.settings):
                    pass
        spawn.assert_not_called()

    async def test_owned_runtime_cleanup_runs_after_render_failure_without_login_secrets(self):
        requests = 0
        def handle(request):
            nonlocal requests
            requests += 1
            if requests == 1:
                raise httpx.ConnectError("not started", request=request)
            return httpx.Response(200, json={"system": {"comfyui_version": "test"}})

        process = SimpleNamespace(pid=424242, poll=Mock(return_value=None), wait=Mock(return_value=0), terminate=Mock())
        factory = httpx.AsyncClient
        with patch("app.comfy_runtime.PROJECT_ROOT", self.root), \
             patch.dict("os.environ", {"SHARE_PASSWORD": "secret", "DATABASE_URL": "private", "SAMPLE_API_KEY": "private"}), \
             patch("app.comfy_runtime.httpx.AsyncClient", side_effect=lambda **kw: factory(transport=httpx.MockTransport(handle), **kw)), \
             patch("app.comfy_runtime.subprocess.Popen", return_value=process) as spawn, \
             patch("app.comfy_runtime.subprocess.run") as stop, \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=NotImplementedError) as unsupported:
            with self.assertRaisesRegex(PictureError, "render failed"):
                async with comfy_runtime(self.settings):
                    raise PictureError("render failed")
        environment = spawn.call_args.kwargs["env"]
        for key in ("SHARE_PASSWORD", "DATABASE_URL", "SAMPLE_API_KEY"):
            self.assertNotIn(key, environment)
        if os.name == "nt":
            self.assertEqual(stop.call_args.args[0], ["taskkill", "/PID", "424242", "/T", "/F"])
        process.wait.assert_called_once_with(timeout=10)
        unsupported.assert_not_awaited()
