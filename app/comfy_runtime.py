"""Start and stop a dedicated runtime so Windows releases its GPU and host buffers."""
import asyncio
from contextlib import asynccontextmanager
import os
import subprocess
import time
from urllib.parse import urlsplit

import httpx

from app.comfy_client import PictureError
from app.config import ImageSettings, PROJECT_ROOT


@asynccontextmanager
async def comfy_runtime(settings: ImageSettings):
    if not settings.managed_runtime:
        yield
        return
    root = PROJECT_ROOT / "data" / "image-runtime"
    python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    main = root / "ComfyUI" / "main.py"
    if not python.is_file() or not main.is_file():
        raise PictureError("The dedicated ComfyUI runtime is not installed.")
    async with httpx.AsyncClient(timeout=2) as client:
        try:
            await client.get(f"{settings.comfy_url}/system_stats")
        except httpx.RequestError:
            pass
        else:
            raise PictureError("The dedicated picture port is already in use. Stop the standalone ComfyUI service or disable managed runtime mode.")
    parsed = urlsplit(settings.comfy_url)
    host = "::1" if parsed.hostname == "::1" else "127.0.0.1"
    # The model service does not need the application's database/login secrets.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("DATABASE_URL", "SHARE_USERNAME", "SHARE_PASSWORD")
                   and not key.endswith(("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))}
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    log = (root / "comfy-managed.log").open("ab")
    process = None
    try:
        # Uvicorn --reload uses a SelectorEventLoop on Windows, which does not
        # implement asyncio subprocesses. Popen works under both loop types.
        process = subprocess.Popen([
            str(python), str(main), "--listen", host, "--port", str(parsed.port or 80),
            "--disable-all-custom-nodes", "--offline", "--disable-auto-launch", "--disable-metadata",
            "--cache-none", "--disable-dynamic-vram"], cwd=main.parent, env=environment,
            stdout=log, stderr=log, **options,
        )
        async with httpx.AsyncClient(timeout=2) as client:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise PictureError("The dedicated picture runtime stopped during startup. Check its local logs.")
                try:
                    response = await client.get(f"{settings.comfy_url}/system_stats")
                    response.raise_for_status()
                    if response.json().get("system", {}).get("comfyui_version"):
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.5)
            else:
                raise PictureError("The dedicated picture runtime did not start in time.")
        yield
    finally:
        if process and process.poll() is None:
            if os.name == "nt":
                # The Windows venv redirector starts a child interpreter. Kill only
                # the process tree created above; never a pre-existing service.
                await asyncio.to_thread(subprocess.run,
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=10)
        log.close()
