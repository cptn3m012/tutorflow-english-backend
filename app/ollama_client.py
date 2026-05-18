import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"


async def generate_from_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        return data["response"]