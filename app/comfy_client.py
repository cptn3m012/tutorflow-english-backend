"""Two fixed, core-node workflows. No workflow supplied by a browser is executed."""
import asyncio
import time
from uuid import uuid4

import httpx

from app.config import ImageSettings
from app.picture_storage import MAX_IMAGE_BYTES


class PictureError(RuntimeError):
    pass


def build_workflow(prompt: str, seed: int, model: str) -> dict:
    flux = model == "flux2-klein-4b"
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors" if flux else "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2" if flux else "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors" if flux else "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage" if flux else "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    }
    if flux:
        nodes.update({
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "11": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1.0}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "13": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": 1024, "height": 1024}},
            "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["6", 0]}},
        })
    else:
        nodes.update({
            "10": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        })
    return nodes


async def free_comfy(settings: ImageSettings):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(f"{settings.comfy_url}/free", json={"unload_models": True, "free_memory": True})
            response.raise_for_status()
            # /free schedules cleanup in ComfyUI's execution thread.
            await asyncio.sleep(1)
    except httpx.HTTPError as error:
        raise PictureError("Could not release the picture model. Check local ComfyUI before generating again.") from error


async def render_picture(prompt: str, seed: int, settings: ImageSettings) -> bytes:
    prompt_id = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), follow_redirects=False) as client:
            response = await client.post(f"{settings.comfy_url}/prompt", json={
                "prompt": build_workflow(prompt, seed, settings.model), "client_id": str(uuid4()),
            })
            response.raise_for_status()
            prompt_id = response.json().get("prompt_id")
            if not isinstance(prompt_id, str):
                raise PictureError("ComfyUI could not start the picture. Check the installed model files.")
            deadline = time.monotonic() + settings.timeout_seconds
            while time.monotonic() < deadline:
                response = await client.get(f"{settings.comfy_url}/history/{prompt_id}")
                response.raise_for_status()
                history = response.json().get(prompt_id)
                if history:
                    if history.get("status", {}).get("status_str") == "error":
                        raise PictureError("The picture model could not finish. Check local ComfyUI and available GPU memory.")
                    images = history.get("outputs", {}).get("9", {}).get("images", [])
                    if not images:
                        raise PictureError("The picture model returned no image.")
                    item = images[0]
                    if item.get("type") not in ("temp", "output"):
                        raise PictureError("ComfyUI returned an unsupported image.")
                    # Download only the output of this server-generated workflow.
                    async with client.stream("GET", f"{settings.comfy_url}/view", params={
                        "filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item["type"],
                    }) as image:
                        image.raise_for_status()
                        content = bytearray()
                        async for chunk in image.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > MAX_IMAGE_BYTES:
                                raise PictureError("The generated picture is too large.")
                        return bytes(content)
                await asyncio.sleep(0.75)
            raise PictureError("Picture generation timed out. Your existing lesson has been kept.")
    except (httpx.HTTPError, KeyError, ValueError) as error:
        raise PictureError("Could not generate a picture. Check local ComfyUI and the installed model files.") from error
    finally:
        if prompt_id:
            # Target only our own prompt, including a queued job after cancellation.
            async with httpx.AsyncClient(timeout=10) as cleanup:
                try:
                    await cleanup.post(f"{settings.comfy_url}/interrupt", json={"prompt_id": prompt_id})
                    await cleanup.post(f"{settings.comfy_url}/queue", json={"delete": [prompt_id]})
                    await cleanup.post(f"{settings.comfy_url}/history", json={"delete": [prompt_id]})
                except httpx.HTTPError:
                    pass
