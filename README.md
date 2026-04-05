# AI Slop Detector

> A Flask-based AI writing detector with explainable scoring, file upload, compare mode, and a local JSON API.

AI Slop Detector estimates whether a sample reads more like human writing or AI-generated text. It blends a RoBERTa detector with phrase heuristics, structural style signals, repeated-template checks, and short-text calibration, then surfaces the result in a browser UI with sentence-level signals, phrase hit breakdowns, compare mode, upload support, and recent-analysis history.

## What It Does

- Analyze pasted text directly
- Scrape text from a URL and analyze it
- Upload `.txt`, `.md`, `.markdown`, and `.pdf` files
- Compare two writing samples side by side
- Show sentence-level AI contributions
- Show phrase hit breakdowns with weights
- Flag low-reliability short samples instead of overstating confidence
- Save recent analyses locally in the browser
- Expose JSON endpoints for direct integration

## How It Works

The detector combines multiple signals:

- `Hello-SimpleAI/chatgpt-detector-roberta` via Hugging Face Transformers for the main AI-vs-human classification
- A weighted phrase heuristic layer that looks for common AI-style wording patterns
- Structural style heuristics for overly uniform sentence lengths, repeated sentence openings, uniform paragraph rhythm, and transition-heavy essay structure
- Extra template checks for polished “humanized AI” phrasing and generic essay boilerplate

The backend also:

- chunks longer text for more stable scoring
- scores individual sentences to highlight likely AI-heavy lines
- softens short-text verdicts because tiny samples are noisy
- restores full saved results from recent analysis history

## UI Features

- Light and dark mode
- Paste, URL, upload, compare, and API tabs
- Verdict card with confidence and reliability messaging
- Score breakdown charts
- Sentence signal panel
- Phrase hit breakdown panel
- Recent analysis history stored in `localStorage`

## API

### `POST /api/analyze`

Analyze a single sample.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Paste at least 20 words here for analysis."}'
```

### `POST /api/compare`

Compare two text samples.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"left_text":"First sample text here.","right_text":"Second sample text here."}'
```

### `GET /api/health`

Simple health check endpoint.

## Tech Stack

- Python
- Flask
- Transformers
- PyTorch
- Requests
- Beautiful Soup
- pypdf
- HTML, CSS, JavaScript
- Chart.js

## Project Structure

```text
AI-SLOP-Detector/
├── index.html
├── info/
│   └── index.html
├── app.py
├── railway.toml
├── requirements.txt
├── README.md
├── LICENSE
├── samples/
│   ├── ai/
│   │   ├── ai-generated.txt
│   │   └── humanized-ai.txt
│   └── human/
│       └── nyu-essay-examples.pdf
├── static/
│   ├── logo.svg
│   ├── script.js
│   └── style.css
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mijnap1/AI-SLOP-Detector.git
cd AI-SLOP-Detector
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Deploying to Railway

This repo is configured for Railway with `gunicorn`.

Files added for deployment:

- [railway.toml](./railway.toml)
- `gunicorn` in [requirements.txt](./requirements.txt)

Railway start command:

```bash
gunicorn --bind [::]:$PORT app:app
```

Basic Railway flow:

1. Push this repo to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Railway will install dependencies from `requirements.txt`.
4. Railway will use `railway.toml` for the start command and healthcheck.
5. After deploy, confirm `/api/health` returns `{"status":"ok"}`.

If you later move the frontend to a separate static host, point the frontend requests at the Railway backend URL.

## GitHub Pages + Railway

The recommended production split for this project is:

- frontend on GitHub Pages at `jamieryu.com/ai-slop-detector`
- backend on Railway for Flask + model inference

This repo is prepared for that setup:

- root [index.html](./index.html) and [info/index.html](./info/index.html) can be served statically by GitHub Pages
- Flask also serves those same files locally
- the backend includes CORS for local dev, GitHub Pages, and `jamieryu.com`

### One required step

Set the Railway backend URL in the `<meta name="api-base-url">` tag inside [index.html](./index.html).

Example:

```html
<meta name="api-base-url" content="https://your-railway-app.up.railway.app">
```

Current behavior:

- on `127.0.0.1:5000`, the frontend uses same-origin Flask
- on `127.0.0.1:5500`, the frontend automatically uses `http://127.0.0.1:5000`
- on GitHub Pages, it will use the configured `api-base-url`

### GitHub Pages path

If the GitHub repository name is `ai-slop-detector`, the project site path will be `/ai-slop-detector`, which is the path you want under your custom domain.

## Notes

- The first launch may take longer because the Hugging Face model downloads on first run.
- PDF upload requires `pypdf`, which is included in `requirements.txt`.
- URL scraping works best on article-style pages with readable main content.
- Very short samples are inherently noisy. The app now marks low-reliability cases instead of treating them like strong evidence.
- The `samples/` folder contains local comparison material used to tune the heuristic layer against raw AI, humanized AI, and real human essays.
- Results should be treated as a signal, not proof.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
