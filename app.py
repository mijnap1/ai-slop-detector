from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import os
import re
from functools import lru_cache

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

app = Flask(__name__)

ALLOWED_ORIGINS = {
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://mijnap1.github.io",
    "https://jamieryu.com",
    "https://www.jamieryu.com",
}

print("Loading AI detection model...")
detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
print("Model loaded.")

AI_PHRASES = [
    (r"\bdelve[sd]?\b", "delve", 3.0),
    (r"\bcomprehensive(ly)?\b", "comprehensive", 1.5),
    (r"\bnuanced?\b", "nuanced", 1.5),
    (r"\bit[\'s]+ important to note\b", "it's important to note", 2.5),
    (r"\bin conclusion\b", "in conclusion", 1.5),
    (r"\bto summarize\b", "to summarize", 1.5),
    (r"\bin summary\b", "in summary", 1.5),
    (r"\bfurthermore\b", "furthermore", 1.5),
    (r"\bmoreover\b", "moreover", 1.5),
    (r"\bas an ai\b", "as an AI", 5.0),
    (r"\bas a language model\b", "as a language model", 5.0),
    (r"\bi\'d be happy to\b", "I'd be happy to", 3.5),
    (r"\bi would be happy to\b", "I would be happy to", 3.5),
    (r"\bi hope this helps\b", "I hope this helps", 3.5),
    (r"\bplease note that\b", "please note that", 2.0),
    (r"\bshed(s)? light on\b", "shed light on", 2.5),
    (r"\bparamount\b", "paramount", 2.0),
    (r"\bfoster(s|ed|ing)?\b", "foster", 1.5),
    (r"\bembark(s|ed|ing)?\b", "embark", 2.0),
    (r"\btapestry\b", "tapestry", 3.0),
    (r"\btestament to\b", "testament to", 2.0),
    (r"\bunderscor(e|es|ed|ing)\b", "underscore", 2.0),
    (r"\bpivotal\b", "pivotal", 1.5),
    (r"\bseamless(ly)?\b", "seamless", 1.5),
    (r"\bsynerg(y|ies|istic)\b", "synergy", 2.0),
    (r"\bin the realm of\b", "in the realm of", 2.5),
    (r"\bit is worth noting\b", "it is worth noting", 2.5),
    (r"\bone must consider\b", "one must consider", 2.0),
    (r"\bthriving\b", "thriving", 1.5),
    (r"\binvaluable\b", "invaluable", 1.5),
]

SHORT_TEXT_WORD_THRESHOLD = 120
LOW_RELIABILITY_WORD_THRESHOLD = 50
MIN_ANALYSIS_WORDS = 20
MAX_HISTORY_ITEMS = 8
ALLOWED_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def normalize_model_output(output):
    label = output["label"].lower()
    score = output["score"]
    if "chatgpt" in label or label == "fake" or label == "label_1":
        return score
    return 1.0 - score


@lru_cache(maxsize=2048)
def model_ai_score(text):
    out = detector(text, truncation=True, max_length=512)[0]
    return normalize_model_output(out)


def batch_model_ai_scores(texts):
    clean_texts = [text for text in texts if text]
    if not clean_texts:
        return {}

    unique_texts = list(dict.fromkeys(clean_texts))
    outputs = detector(unique_texts, truncation=True, max_length=512, batch_size=min(len(unique_texts), 8))
    scores = {}
    for text, output in zip(unique_texts, outputs):
        scores[text] = normalize_model_output(output)
    return scores


def compute_phrase_hits(text):
    text_lower = text.lower()
    hits = []
    total_weight = 0.0

    for pattern, label, weight in AI_PHRASES:
        matches = re.findall(pattern, text_lower)
        if not matches:
            continue
        count = len(matches)
        contribution = count * weight
        hits.append({
            "label": label,
            "count": count,
            "weight": weight,
            "contribution": round(contribution, 2),
        })
        total_weight += contribution

    hits.sort(key=lambda item: item["contribution"], reverse=True)
    return hits, total_weight


def heuristic_ai_score(text):
    word_count = max(len(text.split()), 1)
    hits, total_weight = compute_phrase_hits(text)
    density = total_weight / (word_count / 100)
    return min(density / 12.0, 1.0), hits


def score_sentences(sentences, sentence_score_map):
    scored = []

    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count < 4:
            continue
        score = sentence_score_map.get(sentence)
        if score is None:
            continue
        scored.append({
            "text": sentence,
            "ai_score": round(score * 100, 1),
        })

    scored.sort(key=lambda item: item["ai_score"], reverse=True)
    return scored


def reliability_payload(word_count):
    if word_count < LOW_RELIABILITY_WORD_THRESHOLD:
        return {
            "state": "low",
            "label": "Low reliability",
            "message": f"Only {word_count} words were provided. Short samples are noisy, so treat this as a weak signal.",
        }
    if word_count < 90:
        return {
            "state": "medium",
            "label": "Reduced reliability",
            "message": "This sample is still fairly short, so the score is directionally useful but not highly stable.",
        }
    return {
        "state": "normal",
        "label": "Normal reliability",
        "message": "",
    }


def short_text_baseline(word_count, heuristic, sentence_count):
    if heuristic < 0.08:
        if word_count <= 35:
            return 0.44 if sentence_count <= 2 else 0.46
        if word_count <= 50:
            return 0.47
    return 0.5


def verdict_payload(ai_score, human_score, reliability):
    margin = abs(ai_score - human_score)

    if reliability["state"] == "low":
        return "Low-Reliability Sample", "Weak evidence"
    if margin < 8:
        return "Unclear", "Mixed signals"
    if ai_score >= human_score:
        return "AI-Generated", "High confidence" if ai_score >= 85 else "Moderate confidence" if ai_score >= 65 else "Low confidence"
    return "Human-Written", "High confidence" if human_score >= 85 else "Moderate confidence" if human_score >= 65 else "Low confidence"


def analyze_text(text):
    word_count = len(text.split())
    sentences = split_sentences(text)
    sentence_count = len(sentences)
    chunks = chunk_text(text)
    scorable_sentences = [sentence for sentence in sentences if len(sentence.split()) >= 4]
    score_map = batch_model_ai_scores(chunks + scorable_sentences)
    model_ai_scores = [score_map[chunk] for chunk in chunks]
    model_avg = sum(model_ai_scores) / len(model_ai_scores)

    heuristic, phrase_hits = heuristic_ai_score(text)
    sentence_details = score_sentences(sentences, score_map)

    if word_count <= SHORT_TEXT_WORD_THRESHOLD:
        sentence_scores = [item["ai_score"] / 100 for item in sentence_details if len(item["text"].split()) >= 5]
        sentence_avg = sum(sentence_scores) / len(sentence_scores) if sentence_scores else model_avg

        if word_count <= 60:
            blended = 0.55 * model_avg + 0.25 * sentence_avg + 0.20 * heuristic
        else:
            blended = 0.62 * model_avg + 0.23 * sentence_avg + 0.15 * heuristic
    else:
        blended = 0.82 * model_avg + 0.18 * heuristic

    if word_count < LOW_RELIABILITY_WORD_THRESHOLD:
        baseline = short_text_baseline(word_count, heuristic, sentence_count)
        trust = max(0.08, min(0.32, ((word_count - LOW_RELIABILITY_WORD_THRESHOLD) / 30) * 0.24 + 0.08))
        blended = baseline + (blended - baseline) * trust
    elif word_count <= SHORT_TEXT_WORD_THRESHOLD:
        trust = 0.60 + ((word_count - LOW_RELIABILITY_WORD_THRESHOLD) / (SHORT_TEXT_WORD_THRESHOLD - LOW_RELIABILITY_WORD_THRESHOLD)) * 0.20
        blended = 0.5 + (blended - 0.5) * trust

    blended = max(0.0, min(1.0, blended))

    ai_score = round(blended * 100, 1)
    human_score = round((1.0 - blended) * 100, 1)
    reliability = reliability_payload(word_count)
    verdict, confidence_label = verdict_payload(ai_score, human_score, reliability)
    confidence = round(max(ai_score, human_score), 1)

    return {
        "verdict": verdict,
        "ai_score": ai_score,
        "human_score": human_score,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "chunks_analyzed": len(chunks),
        "word_count": word_count,
        "heuristic_score": round(heuristic * 100, 1),
        "reliability": reliability,
        "sentence_details": sentence_details[:8],
        "phrase_hits": phrase_hits[:10],
    }


def scrape_text(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.body
    if not article:
        raise ValueError("Could not extract text from page.")
    text = article.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def parse_uploaded_file(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("No file selected.")

    filename = file_storage.filename
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError("Unsupported file type. Use .txt, .md, or .pdf.")

    if ext == ".pdf":
        if PdfReader is None:
            raise ValueError("PDF support requires pypdf to be installed.")
        reader = PdfReader(file_storage.stream)
        text = " ".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = file_storage.read().decode("utf-8", errors="ignore")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Could not extract readable text from the file.")

    return text, filename


def request_text_payload():
    if request.is_json:
        data = request.get_json() or {}
        return {
            "text": (data.get("text") or "").strip(),
            "url": (data.get("url") or "").strip(),
            "file": None,
            "compare_text": (data.get("compare_text") or "").strip(),
        }

    return {
        "text": (request.form.get("text") or "").strip(),
        "url": (request.form.get("url") or "").strip(),
        "file": request.files.get("file"),
        "compare_text": (request.form.get("compare_text") or "").strip(),
    }


def build_analysis_response(text="", url="", file_storage=None):
    scraped_url = None
    uploaded_filename = None

    if url:
        text = scrape_text(url)
        scraped_url = url
    elif file_storage:
        text, uploaded_filename = parse_uploaded_file(file_storage)

    if len(text.split()) < MIN_ANALYSIS_WORDS:
        reliability = reliability_payload(len(text.split()))
        return {
            "error": None,
            "verdict": "Low-Reliability Sample",
            "ai_score": 50.0,
            "human_score": 50.0,
            "confidence": 50.0,
            "confidence_label": "Too short to call",
            "chunks_analyzed": 1,
            "word_count": len(text.split()),
            "heuristic_score": 0.0,
            "reliability": {
                "state": "low",
                "label": reliability["label"],
                "message": f"Need at least {MIN_ANALYSIS_WORDS} words for a meaningful read. This sample is too short for a trustworthy verdict.",
            },
            "sentence_details": [{"text": sentence, "ai_score": 50.0} for sentence in split_sentences(text)[:6]],
            "phrase_hits": [],
            "scraped_url": scraped_url,
            "scraped_preview": text[:400] + "..." if len(text) > 400 else text,
            "uploaded_filename": uploaded_filename,
        }

    result = analyze_text(text)
    if scraped_url:
        result["scraped_url"] = scraped_url
        result["scraped_preview"] = text[:400] + "..." if len(text) > 400 else text
    if uploaded_filename:
        result["uploaded_filename"] = uploaded_filename
    return result


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(app.root_path, "index.html")


@app.route("/info.html")
@app.route("/how-it-works")
def how_it_works():
    return send_from_directory(app.root_path, "info.html")


@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request_text_payload()
    text = payload["text"]
    url = payload["url"]
    file_storage = payload["file"]

    if not text and not url and not file_storage:
        return jsonify({"error": "Provide text, a URL, or a file."}), 400

    try:
        result = build_analysis_response(text=text, url=url, file_storage=file_storage)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 400


@app.route("/api/compare", methods=["POST", "OPTIONS"])
def compare():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json() or {}
    left_text = (data.get("left_text") or "").strip()
    right_text = (data.get("right_text") or "").strip()

    if not left_text or not right_text:
        return jsonify({"error": "Provide both text samples to compare."}), 400

    try:
        return jsonify({
            "left": build_analysis_response(text=left_text),
            "right": build_analysis_response(text=right_text),
        })
    except Exception as exc:
        return jsonify({"error": f"Comparison failed: {str(exc)}"}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
