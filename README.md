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
- Save lesson variants and favourites in a shared PostgreSQL library.

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
| `GET` | `/lessons/saved` | List saved lessons (`limit` up to 100, `offset`) |
| `GET` | `/lessons/saved/{uuid}` | Read a saved lesson |
| `PUT` | `/lessons/saved/{uuid}` | Save/import/restore a lesson with an independent UUID |
| `PATCH` | `/lessons/saved/{uuid}` | Set favourite state with `{ "favorite": true }` |
| `DELETE` | `/lessons/saved/{uuid}` | Delete a saved lesson |

## Shared lesson database

Use PostgreSQL for the shared library. Copy `.env.example` to `.env` and set:

```dotenv
DATABASE_URL=postgresql+psycopg://tutorflow:CHANGE_ME@127.0.0.1:5432/tutorflow
```

Create the database and a dedicated login in pgAdmin first; the application login
should own this database, without superuser, role-creation or database-creation rights.
URL-encode special characters in the password. `.env` is loaded automatically;
process environment variables take precedence. Never commit `.env` or database dumps.

The first application startup runs versioned Alembic migrations. It creates the
`saved_lessons` table, but does not create the PostgreSQL database or login.
You can also migrate explicitly with `python -m alembic upgrade head`.
Without `DATABASE_URL`, local development uses the persistent file
`data/tutorflow.db` (SQLite). An unavailable configured PostgreSQL database produces
an error; it never silently switches to another database. Changing the URL does
not move existing data between databases.

Each saved variant stores the full lesson JSON (JSONB in PostgreSQL), level, theme,
duration, source, timestamp and favourite state. Generate and PDF routes retain
their existing contracts; generation alone does not add a lesson to the library.
The frontend explicitly saves selected variants. All visitors share one library;
this version does not contain separate user accounts or private per-user libraries.

Save payloads match the frontend's versioned browser library: `id`, `saved_at`,
`favorite`, `level`, `duration`, `theme`, `mode`, `lesson`. Retrying a PUT with the
same UUID returns the existing record, preserving later favourite changes. DELETE
is idempotent and PUT can restore a deleted record. Browser imports use the same
UUIDs, so repeating an import creates no duplicates. Database failures return 503
without including connection credentials in the response.

Back up PostgreSQL with `pg_dump` or pgAdmin's Backup action. Store backups outside
Git and test restoration before relying on them. For SQLite, stop the backend
before copying its database file.

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

### Reading quality and answer keys

Generation prepares the final reading before creating comprehension questions and worksheets.
Each lesson keeps the existing `reading_questions` string array and adds a corresponding
`reading_answers` array in the same order:

```json
{
  "reading_questions": ["What does Anna order?"],
  "reading_answers": [
    {
      "question": "What does Anna order?",
      "answer": "Tea and a sandwich.",
      "evidence": "She reads the menu and asks the waiter for tea and a sandwich."
    }
  ]
}
```

The backend checks question counts, duplicate questions, answer order, and exact evidence
from the final reading. Cloze exercises use 1-4 distinct sentences from that reading,
replace whole words or phrases, and must reconstruct their source sentences with the answer key.
Learners use the reading to resolve the blanks. A different word from the bank must not
restore another sentence also present in that reading, and blank contexts cannot repeat.
Incoming generated worksheets are rebuilt after the reading is finalized.

A separate model review checks passage coherence, question/answer correctness, supporting
evidence, and cloze ambiguity. Issues trigger repairs of the affected section; a reading
change regenerates its questions, answers, and worksheets. There are at most three content
review rounds with two repair rounds. Structured-output parsing and section generation
have their own limits of three attempts. Exhausted content repairs return an error rather
than padding text or restarting the entire lesson generation.

Question generation and quality review use Ollama's JSON Schema `format` parameter.
Expect an additional question-generation call and at least one quality-review call per
lesson variant. Semantic review is model-assisted and still requires evaluation against
teacher-reviewed examples; it does not certify CEFR proficiency or guarantee every answer.

With `pdf_options.include_answers=true`, PDFs include a reading answer key with answers and
quoted evidence. Student PDFs hide this section by default. Legacy PDF payloads without
`reading_answers` are still accepted. Supplied answer keys with mismatched question order
or evidence absent from the reading are rejected.

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
| `app/reading_service.py` | Generates comprehension questions, answers, and supporting evidence |
| `app/quality_service.py` | Reviews semantic consistency of readings, answers, and cloze exercises |
| `app/worksheet_service.py` | Builds and repairs reading-based cloze exercises |
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

GET, POST, PUT, PATCH and DELETE are allowed through CORS, along with the JSON Content-Type header. Cross-origin cookies are disabled because the current frontend does not use cookie authentication. Content-Disposition is exposed for PDF downloads. For production, configure exact frontend origins; CORS is not authentication.

## Connecting the React Frontend Locally

Run Ollama with `qwen3:14b` and start this backend in its virtual environment:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Use the [TutorFlow frontend](https://github.com/cptn3m012/tutorflow-english-frontend).
In that repository, create `.env.local` with public configuration only:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_APP_MODE=live
```

Restart the frontend with `npm run dev` and open `http://127.0.0.1:5173/studio`.
TutorFlow preserves its demo mode; select **Live API** to call this backend.
Without local configuration, a fresh checkout opens the authored demo lessons.
Start with one lesson variant;
disable images and visual activity for the first connection check. Generation can
take several minutes.

The two applications communicate over HTTP. They remain separate Git repositories.
Both development servers and Ollama should listen on loopback addresses for local use.
The health endpoint checks FastAPI availability, not model readiness.

Never put secrets into `VITE_*` variables: Vite embeds them in browser code.
Optional image provider credentials stay in the backend environment. The ignored `.env`
file is loaded automatically, with process environment variables taking precedence.
Commit only placeholder configuration,
and review `git status` and `git diff --cached` before committing.

Public deployment requires separate HTTPS, authentication and request-limit setup
before exposing generation. Keep Ollama's port 11434 private.

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

Run the regression and API integration tests without contacting Ollama or image providers:

```bash
python -m unittest discover -s tests -v
```

The tests use controlled model responses to verify dependency rebuilding, evidence checks,
bounded repairs, word boundaries, API compatibility, and PDF answer visibility. They do
not benchmark the live model's generation quality or latency.

The backend is ready for local lesson generation and PDF export. Before production deployment, configure environment-specific values such as allowed CORS origins, Ollama model settings, and any optional image provider credentials.

## Acknowledgements

This project was created with assistance from OpenAI Codex.

## License

This project is licensed under the MIT License.
