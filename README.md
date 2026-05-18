# TutorFlow English Backend

A FastAPI backend for generating English lesson materials with CEFR-aligned content, visual activities, worksheet assets, image suggestions, and PDF export.

The application is designed for teachers, tutors, and language-learning tools that need structured lesson content generated from a topic, date, level, and teaching preferences.

## Features

- Generate English lesson variants for CEFR levels A1, A2, B1, B2, C1, and C2.
- Create lessons from a custom topic.
- Create date-based seasonal lessons, such as holidays or travel themes.
- Generate target vocabulary, reading texts, comprehension questions, speaking questions, role-play tasks, and sentence frames.
- Align lesson goals with CEFR communication, vocabulary, and grammar targets.
- Configure lesson length, vocabulary count, question count, skill focus, lesson style, and visual activity settings.
- Generate worksheet-style activities.
- Generate `find_in_picture` visual tasks for classroom or printable materials.
- Optionally search for lesson images through Openverse and Pexels.
- Export generated lessons as PDF files.
- Preview generated PDFs directly in the browser.

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic
- HTTPX
- Uvicorn
- Ollama for local LLM generation

By default, the backend connects to a local Ollama API:

```text
http://localhost:11434/api/generate
```

The default model is:

```text
qwen3:14b
```

You can change the Ollama URL and model name in `app/ollama_client.py`.

## Requirements

Before running the project, install:

- Python 3.12 or newer
- pip
- Ollama
- the `qwen3:14b` Ollama model, or another compatible model configured in `app/ollama_client.py`

Pull the default Ollama model:

```bash
ollama pull qwen3:14b
```

Start Ollama:

```bash
ollama serve
```

## Clone the Repository

```bash
git clone https://github.com/cptn3m012/tutorflow-english-backend.git
cd english-lesson-app-backend
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the virtual environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The backend does not require any secret keys for basic lesson generation.

Optional environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `OPENVERSE_API_URL` | Openverse image search API URL | `https://api.openverse.org/v1/images/` |
| `PEXELS_API_URL` | Pexels search API URL | `https://api.pexels.com/v1/search` |
| `PEXELS_API_KEY` | Optional Pexels API key | unset |
| `IMAGE_PROVIDER_MODE` | Image search mode. Use `off`, `offline`, `local`, or `disabled` to disable remote image search. | `remote` |

Example on Windows PowerShell:

```powershell
$env:PEXELS_API_KEY="your_pexels_api_key"
```

Disable remote image search:

```powershell
$env:IMAGE_PROVIDER_MODE="off"
```

Local `.env` files are ignored by Git.

## Run the Application

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Basic backend status response |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/lessons/cefr-levels` | List all supported CEFR levels |
| `GET` | `/lessons/cefr-levels/{level}` | Get details for a single CEFR level |
| `GET` | `/lessons/advanced-options` | Get available lesson generation options |
| `GET` | `/lessons/pdf-options` | Get available PDF export options |
| `POST` | `/lessons/generate` | Generate lesson variants |
| `POST` | `/lessons/export/pdf` | Export lessons as a downloadable PDF |
| `POST` | `/lessons/export/pdf/preview` | Preview lessons as an inline PDF |

## Generate Lessons

Request:

```http
POST http://127.0.0.1:8000/lessons/generate
Content-Type: application/json
Accept: application/json
```

Example body:

```json
{
  "level": "A2",
  "duration": 30,
  "topic": "food and drinks in a cafe",
  "lesson_date": null,
  "variant_count": 2,
  "advanced_options": {
    "reading_min_words": 90,
    "reading_max_words": 130,
    "target_vocabulary_count": 10,
    "reading_question_count": 4,
    "speaking_question_count": 6,
    "sentence_frame_count": 4,
    "visual_word_count": 6,
    "include_visual_activity": true,
    "include_worksheet_assets": true,
    "include_images": true,
    "skill_focus": "speaking",
    "lesson_style": "conversation",
    "extra_instructions": "Make the tasks useful for adult learners."
  }
}
```

Minimal body:

```json
{
  "level": "A2",
  "duration": 30,
  "topic": "food and drinks in a cafe",
  "variant_count": 2
}
```

Either `topic` or `lesson_date` must be provided.

## PDF Export

Download a PDF:

```http
POST http://127.0.0.1:8000/lessons/export/pdf
Content-Type: application/json
Accept: application/pdf
```

Preview a PDF in the browser:

```http
POST http://127.0.0.1:8000/lessons/export/pdf/preview
Content-Type: application/json
Accept: application/pdf
```

The PDF endpoints accept either a single lesson, a list of lessons, or the full response returned by `/lessons/generate`.

PDF export options include:

- template: `clean`, `compact`, or `worksheet`
- font size
- font family
- spacing
- answer key visibility
- visual activity visibility
- worksheet asset visibility
- custom JPEG images as data URLs

## Project Structure

```text
english-lesson-app-backend/
|-- app/
|   |-- api/
|   |   `-- routes_lessons.py
|   |-- cefr_data.py
|   |-- cefr_service.py
|   |-- image_service.py
|   |-- lesson_service.py
|   |-- main.py
|   |-- ollama_client.py
|   |-- pdf_service.py
|   |-- prompts.py
|   |-- schemas.py
|   `-- theme_service.py
|-- requirements.txt
|-- test_main.http
|-- .gitignore
`-- README.md
```

## Main Modules

| File | Purpose |
| --- | --- |
| `app/main.py` | Creates the FastAPI application, configures CORS, and registers routers |
| `app/api/routes_lessons.py` | Defines lesson and PDF API routes |
| `app/lesson_service.py` | Contains the main lesson generation, validation, and enrichment flow |
| `app/ollama_client.py` | Sends generation requests to the local Ollama API |
| `app/prompts.py` | Stores prompt builders for lesson generation and image metadata |
| `app/schemas.py` | Defines Pydantic request and response models |
| `app/cefr_service.py` | Handles CEFR profile lookup, normalization, and target matching |
| `app/cefr_data.py` | Stores CEFR reference data used by the application |
| `app/image_service.py` | Handles image search through Openverse and Pexels |
| `app/pdf_service.py` | Builds and renders PDF lesson exports |
| `app/theme_service.py` | Resolves lesson themes from topics or calendar dates |

## CORS

The backend currently allows local frontend origins:

```text
http://127.0.0.1:5173
http://localhost:5173
http://127.0.0.1:4173
http://localhost:4173
```

For production, update the `allow_origins` list in `app/main.py` with the deployed frontend URL.

## Security Notes

- API keys are not hardcoded in the source code.
- `PEXELS_API_KEY` is read from the environment.
- `.env`, virtual environments, IDE files, Python cache files, and generated sample PDFs are ignored by Git.
- Review environment variables before deploying or publishing the project.

## Useful Commands

Run the backend:

```bash
uvicorn app.main:app --reload
```

Check health status:

```bash
curl http://127.0.0.1:8000/health
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

## Current Status

The backend is ready for local lesson generation and PDF export. Before production deployment, configure environment-specific values such as allowed CORS origins, Ollama model settings, and any optional image provider credentials.

## License

This project is licensed under the MIT License.
