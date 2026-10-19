# TutorFlow English Backend

A FastAPI backend for generating English lesson materials with CEFR-aligned content, worksheets, local picture-description activities, and PDF export.

The application is designed for teachers, tutors, and language-learning tools that need structured lesson content generated from a topic, date, level, and teaching preferences.

## Features

- Generate English lesson variants for CEFR levels A1, A2, B1, B2, C1, and C2.
- Create lessons from a custom topic.
- Create date-based seasonal lessons, such as holidays or travel themes.
- Generate target vocabulary, reading texts, comprehension questions, speaking questions, role-play tasks, and sentence frames.
- Align lesson goals with CEFR communication, vocabulary, and grammar targets.
- Configure lesson length, vocabulary count, question count, skill focus, lesson style, and visual activity settings.
- Generate worksheet-style activities.
- Generate lesson pictures locally with ComfyUI and FLUX.2 klein 4B.
- Prepare picture-description questions from actual image observations using Qwen3-VL.
- Keep picture generation separate from text generation, with persistent jobs and immutable image versions.
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
http://127.0.0.1:11434/api/generate
```

The default model is:

```text
gpt-oss:20b
```

Configure the model and Ollama base URL through environment variables or a local `.env` file.

## Requirements

Before running the project, install:

- Python 3.12 or newer
- pip
- Ollama
- the `gpt-oss:20b` Ollama model, or another compatible model configured with `OLLAMA_MODEL`

Pull the default Ollama model:

```bash
ollama pull gpt-oss:20b
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
| `OLLAMA_MODEL` | Local model used for lesson generation and review | `gpt-oss:20b` |
| `OLLAMA_BASE_URL` | Ollama server base URL; the client appends `/api/generate` | `http://127.0.0.1:11434` |
| `OLLAMA_NUM_CTX` | Context window in tokens, including the prompt and generated output; allowed range 8192–131072 | `12288` |
| `OLLAMA_TIMEOUT_SECONDS` | Timeout for each model request; allowed range 1–600 seconds | `180` |
| `OLLAMA_KEEP_ALIVE` | How long Ollama keeps the model loaded between requests | `10m` |
| `OLLAMA_REASONING_EFFORT` | GPT-OSS reasoning effort for generation and section repairs: `low`, `medium`, or `high` | `low` |
| `OLLAMA_REVIEW_REASONING_EFFORT` | GPT-OSS reasoning effort for semantic quality review | `medium` |
| `LOCAL_IMAGES_ENABLED` | Enable the local picture worker | `false` |
| `COMFYUI_BASE_URL` | Local ComfyUI origin; loopback addresses only | `http://127.0.0.1:8188` |
| `COMFYUI_MANAGED_RUNTIME` | Start the dedicated runtime only during rendering and stop it before Ollama uses the GPU | `false` |
| `IMAGE_MODEL` | Fixed preset: `flux2-klein-4b` or `z-image-turbo` | `flux2-klein-4b` |
| `IMAGE_VISION_MODEL` | Local Ollama vision model | `qwen3-vl:4b-instruct` |
| `IMAGE_TIMEOUT_SECONDS` | Per-stage picture timeout, 30–600 seconds | `300` |
| `IMAGE_STORAGE_DIR` | Private directory for generated JPEG files | `data/lesson-images` |

Local `.env` files are ignored by Git.

### Local pictures

Install [ComfyUI](https://docs.comfy.org/installation/manual_install) in
`data/image-runtime/ComfyUI`, using a separate Python environment at
`data/image-runtime/venv`. Its dependencies are separate from the backend.
Use core nodes only; no custom extensions or paid APIs are needed.

The [FLUX.2 klein distilled workflow](https://docs.comfy.org/tutorials/flux/flux-2-klein)
uses these weights in ComfyUI's model directories:

| File | Directory | Official download |
| --- | --- | --- |
| `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models` | [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/tree/main) |
| `qwen_3_4b.safetensors` | `models/text_encoders` | [Comfy-Org](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders) |
| `flux2-vae.safetensors` | `models/vae` | [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-dev/tree/main/split_files/vae) |

Pull the planner with `ollama pull qwen3:14b` and the vision model with
`ollama pull qwen3-vl:4b-instruct`. Set
`LOCAL_IMAGES_ENABLED=true` in the private `.env`, and start ComfyUI on loopback.
For the dedicated installation, enable `COMFYUI_MANAGED_RUNTIME=true` to release
the runtime's GPU and host-memory allocations between stages. The backend launches
only its own runtime and never terminates a pre-existing service. Do not also start
standalone ComfyUI in this mode. Alternatively, `scripts/start-images.ps1` starts
an installed standalone service in the background on Windows.
Keep one backend worker: text and picture generation share one GPU semaphore.

Create a job for a saved lesson with
`POST /lessons/saved/{lesson_id}/picture-jobs` and `{"style":"photo"}` or
`{"style":"illustration"}`. Read the latest job with `GET` at the same path.
`GET /lessons/saved/{lesson_id}/picture-jobs/{job_id}/events` streams progress as SSE.
Generated files are available through authenticated `/lesson-images/{asset_id}`
endpoints; browser requests cannot supply a workflow, model filename or local path.

Pictures can also be generated before saving to the library. Create an idempotent
working copy with `PUT /lessons/picture-drafts/{lesson_id}` using the lesson payload,
then use the same picture job endpoints with its UUID. Retrieve its latest content
with `GET /lessons/picture-drafts/{lesson_id}`. Working copies are excluded from
library listings and saved-lesson views. An explicit `PUT /lessons/saved/{lesson_id}`
promotes the working copy, retaining its canonical picture and any active job.

The local `IMAGE_PLANNER_MODEL` (Qwen3 14B by default) plans one scene grounded in a literal source excerpt and
reviews the plan before releasing its GPU memory. Essential objects come from that scene
instead of a global vocabulary whitelist. A separate plan review checks its setting,
activity and cultural fit before rendering. Qwen3-VL observes pixels without the
plan or lesson context, then the larger planner checks those blind observations
against the selected scene. The models release GPU memory between stages.
The planner also specifies incompatible object types for that particular scene.
Clearly observed excluded props reject an image even when a general vision verdict
would approve it; ordinary objects appropriate to another lesson remain available.
The image check assesses required-object presence visually, including natural synonyms.
Scene plans link selected original vocabulary to visible cues or meaningful discussion.
Abstract concepts may use a representative illustrative scenario, anchored to the reading
and explicitly distinguished from events narrated in it. Prominent garbled lettering
requires regeneration; words are rendered separately in the interface and PDF.
Pictures avoid text-heavy screens and generic lectures about the topic.
Questions are prepared after acceptance by the text planner, using blind observations.
An activity has one factual observation, one lesson application and one open discussion
prompt, all anchored in visible objects. Application and discussion use original lesson vocabulary;
abstract words are not forced into factual observations. Application and
discussion answers are examples, not uniquely correct facts. Vocabulary coverage,
sentence starters and a separate activity review are validated before publication.
Used vocabulary is identified from the actual question and answer text rather than model-supplied tags.
Question repairs keep the accepted picture. A picture missing essential objects or contradicting the topic
is regenerated once with targeted feedback; failure keeps the text and previous picture intact.
Pictures and corresponding activities are published in one database transaction.
Both PDF editions embed the stored image; only the teacher edition contains answers
and evidence. Model checks can still make mistakes, so review materials before teaching.

Image files, model weights and runtime logs live under ignored `data/`; PostgreSQL
stores their metadata and jobs. Back up both the database and `data/lesson-images`.
Deleted lesson images are retained for Undo and hidden while the lesson is absent.

### Local model generation

The backend uses Ollama's native `/api/generate` endpoint with a TutorFlow system
prompt and task-specific sampling and output-token limits. Draft lessons,
comprehension questions, quality reviews, cloze repairs and image metadata use
JSON Schema constraints. Reading rewrites return plain passage text. The draft
schema restricts vocabulary and activity counts to the request and CEFR focus
choices to the supplied syllabus; later stages add answer evidence and worksheets.

For `gpt-oss`, generation uses low reasoning effort by default and semantic review
uses medium effort. These settings balance responsiveness with the review's need
to solve questions and check source evidence. Evaluate these configurable defaults
with representative lessons to assess generation quality and latency. Ollama returns reasoning in a
separate field, and TutorFlow consumes only the final `response` as lesson content.
See Ollama's [thinking controls](https://docs.ollama.com/capabilities/thinking) and
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

An unavailable model, connection failure, timeout or incomplete output returns an
error without automatically restarting the whole lesson pipeline. Content and
format repairs have bounded attempts; exhausted section repairs also stop the
request. Increase context or timeout settings only after checking the requested
material length and available resources.

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
| `POST` | `/lessons/topics` | Suggest six topics for a course type, age group and CEFR level |
| `POST` | `/lessons/generate` | Generate lesson variants |
| `POST` | `/lessons/export/pdf` | Export lessons as a downloadable PDF |
| `POST` | `/lessons/export/pdf/preview` | Preview lessons as an inline PDF |
| `GET` | `/lessons/saved` | List saved lessons (`limit` up to 100, `offset`) |
| `GET` | `/lessons/saved/{uuid}` | Read a saved lesson |
| `PUT` | `/lessons/saved/{uuid}` | Save/import/restore a lesson with an independent UUID |
| `PATCH` | `/lessons/saved/{uuid}` | Set favourite state with `{ "favorite": true }` |
| `PATCH` | `/lessons/saved/{uuid}/content` | Edit selected lesson fields with `expected_token` and `changes` |
| `POST` | `/lessons/saved/{uuid}/regenerate` | Regenerate one section with `expected_token`, `section` and optional `instructions` |
| `POST` | `/lessons/saved/{uuid}/quiz` | Generate and save a checked exit quiz with `expected_token` and `question_count` (3-5) |
| `DELETE` | `/lessons/saved/{uuid}` | Delete a saved lesson |

## Choosing a lesson topic

Post `{"english_type":"business","age_group":"adults","level":"B1","exclude_topics":[]}`
to `/lessons/topics`. The local model returns six `{title, description}` topic choices.
Send previous titles in `exclude_topics` (up to 12) to request different suggestions.
This uses the same model queue as lesson and picture generation; a busy model returns 429.
Unavailable models return 503; invalid suggestions return 502 without generating a lesson.

Pass `english_type` (`general` or `business`) and `age_group` (`children`, `teens` or `adults`)
with the chosen `topic` to `/lessons/generate`. Defaults are General English and adults
for existing clients. Audience settings are stored on each lesson and used in generation,
reading rewrites, section repairs and quality review. Business English for children uses
school projects and pretend shops rather than adult workplace assumptions.

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

New saves and picture drafts validate required text, known optional sections,
reading answer order and quoted evidence, picture version links, and any embedded
level, duration or theme against the saved metadata. Invalid payloads return 422
before changing the database. Older lessons without answer keys or picture sections,
demo CEFR descriptions, and unknown extension fields remain supported. Existing
records remain readable without a data migration.

Saved lesson responses include `content_token`. Supply the latest token when editing
or regenerating to protect against concurrent changes (409); favourite changes do
not invalidate the token. Supported sections are `overview`, `vocabulary`, `reading`,
`comprehension`, `speaking`, `activities` and `sentence_frames`. Regeneration shares
the local GPU capacity limit (429) and publishes its result atomically after validation.
Failures leave the saved content intact. Text edits are available without Ollama.
Editing a reading clears its old questions, answers and reading exercises; regenerating
it rebuilds questions, answers and existing cloze exercises. Changed reading, vocabulary
or goals mark related picture activities stale while retaining the image. Generate a
new picture to prepare a matching activity. Active picture jobs block text changes (409).

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

Drafts, question generation, quality review, cloze repairs and image metadata use
Ollama's JSON Schema `format` parameter. Passage rewrites remain plain text.
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
|   |-- generation_schemas.py
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
| `app/generation_schemas.py` | Defines strict model-output contracts for lesson drafts and image metadata |
| `app/prompts.py` | Stores prompt builders for lesson generation and image metadata |
| `app/schemas.py` | Defines Pydantic request and response models |
| `app/cefr_service.py` | Handles CEFR profile lookup, normalization, and target matching |
| `app/cefr_data.py` | Stores CEFR reference data used by the application |
| `app/picture_generation.py` | Plans scenes, renders local pictures and grounds questions in observations |
| `app/picture_worker.py` | Processes persistent picture jobs on the shared GPU queue |
| `app/picture_storage.py` | Stores private JPEG image versions |
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

Run Ollama with `gpt-oss:20b` and start this backend in its virtual environment:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Use the [TutorFlow frontend](https://github.com/cptn3m012/tutorflow-english-frontend).
In that repository, create `.env.local` with public configuration only:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the frontend with `npm run dev` and open `http://127.0.0.1:5173/studio`.
TutorFlow 1.0 uses live generation and the backend lesson library. Saved variants
open on their own `/lessons/{id}` frontend pages.
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

## Exit quizzes

Use `POST /lessons/saved/{id}/quiz` with the current `expected_token` and an optional
`question_count` from 3 to 5 (default 3). A quiz is generated separately for a saved
lesson and contains vocabulary, grammar and communication checks, with multiple-choice
items and a short communication task. It uses the selected CEFR profile, lesson goal,
actual vocabulary and practised structures. Short-answer keys are examples with acceptance
guidance; the application does not automatically grade free-text responses.

Structural validation checks counts, coverage, distinct choices, the selected answer and
vocabulary/grammar targets. A separate model review solves the questions and checks the
key and ambiguity. Generation and review retries are bounded. The existing quiz stays
saved on failure, model contention or a concurrent edit. Successful publication changes
only the quiz on the same lesson record; no database migration is required.

A fingerprint of the level and source lesson text marks the quiz stale after relevant
edits or section regeneration. Picture/favourite changes do not invalidate it. Stale
quizzes are hidden in Student view and omitted from full lesson PDFs. Standalone export
rejects a missing or stale quiz even if the client supplies an incorrect stale flag.

The normal PDF endpoints accept `pdf_options.content: "quiz"` for a separate handout;
`include_answers` controls its key, which starts on a separate page. Full lesson export
includes a current quiz by default; set `include_exit_quiz: false` to omit it. Legacy
lessons without quizzes remain valid. Tests use controlled model responses, not a live
benchmark of question quality.

## Validation

New lesson quality reviews use the complete selected CEFR teaching profile and check
the goal, vocabulary, reading, comprehension, speaking, pair work/role-play, sentence
frames and cloze exercises. Concrete issues trigger bounded repairs of the affected
section, followed by another review. The selected communication and grammar targets
must be practised in the learner tasks, rather than only listed as metadata.

Section regeneration, reading repair and question generation receive the same full
profile. Regeneration reviews only the requested section and its linked dependencies;
it preserves unrelated sections and publishes nothing when the review fails. Manual
edits remain teacher-controlled and do not call the model.

Grammar fallback cues require a progressive verb after `am/is/are` or `was/were`;
copular sentences such as “She is a teacher” are not treated as Continuous. Missing
grammar evidence requires an explicit target rather than silently choosing the first
entry. A1 modal targets now distinguish `can/can't` from the auxiliary `do`; older
focus labels remain readable and normalize to the corrected label when generated.

Profiles are teaching references, not certification. CEFR describes communicative
competence through [can-do descriptors](https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions).
Earlier-level language remains valid at higher levels; a word count or single tense
does not establish a lesson's CEFR level. Automated checks depend on the configured
model and still require teacher review. Existing saved lessons are not re-reviewed
or changed automatically.

Run the regression and API integration tests without contacting Ollama or image providers:

```bash
python -m unittest discover -s tests -v
```

The tests use controlled model responses to verify dependency rebuilding, evidence checks,
bounded repairs, word boundaries, API compatibility, and PDF answer visibility. They do
not benchmark the live model's generation quality or latency.

Database tests use isolated SQLite files to check migrations, restart persistence,
UUID retries, pagination, favourites, deletion/undo and safe database errors.
Additional API tests cover authentication, same-origin writes, upload limits,
generation capacity and frontend routing. On 9 October 2026, all 59 backend tests
passed. A browser integration check also verified saved lesson persistence,
favourites and deletion/undo against PostgreSQL in two isolated browser sessions.

The backend is ready for local lesson generation and PDF export. Before production deployment, configure environment-specific values such as allowed CORS origins, Ollama model settings, and any optional image provider credentials.

## Acknowledgements

This project was created with assistance from OpenAI Codex.

## License

This project is licensed under the MIT License.
