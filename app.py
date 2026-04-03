from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)

print("Loading AI detection model...")
# Trained specifically on ChatGPT output — much better on modern AI text
detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
print("Model loaded.")

# AI writing telltale phrases with weights
AI_PHRASES = [
    (r'\bdelve[sd]?\b', 3.0),
    (r'\bcomprehensive(ly)?\b', 1.5),
    (r'\bnuanced?\b', 1.5),
    (r'\bit[\'s]+ important to note\b', 2.5),
    (r'\bin conclusion\b', 1.5),
    (r'\bto summarize\b', 1.5),
    (r'\bin summary\b', 1.5),
    (r'\bfurthermore\b', 1.5),
    (r'\bmoreover\b', 1.5),
    (r'\bas an ai\b', 5.0),
    (r'\bas a language model\b', 5.0),
    (r'\bi\'d be happy to\b', 3.5),
    (r'\bi would be happy to\b', 3.5),
    (r'\bi hope this helps\b', 3.5),
    (r'\bplease note that\b', 2.0),
    (r'\bshed(s)? light on\b', 2.5),
    (r'\bparamount\b', 2.0),
    (r'\bfoster(s|ed|ing)?\b', 1.5),
    (r'\bembark(s|ed|ing)?\b', 2.0),
    (r'\btapestry\b', 3.0),
    (r'\btestament to\b', 2.0),
    (r'\bunderscor(e|es|ed|ing)\b', 2.0),
    (r'\bpivotal\b', 1.5),
    (r'\bseamless(ly)?\b', 1.5),
    (r'\bsynerg(y|ies|istic)\b', 2.0),
    (r'\bin the realm of\b', 2.5),
    (r'\bit is worth noting\b', 2.5),
    (r'\bone must consider\b', 2.0),
    (r'\bthriving\b', 1.5),
    (r'\binvaluable\b', 1.5),
]

def heuristic_ai_score(text):
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    total_weight = 0.0
    for pattern, weight in AI_PHRASES:
        matches = len(re.findall(pattern, text_lower))
        total_weight += matches * weight
    # Normalize by density: 1 heavy hit per 100 words ~ 0.3 signal
    density = total_weight / (word_count / 100)
    return min(density / 12.0, 1.0)

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
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def analyze_text(text):
    chunks = chunk_text(text)
    model_ai_scores = []

    for chunk in chunks:
        out = detector(chunk, truncation=True, max_length=512)[0]
        label = out["label"].lower()
        score = out["score"]
        # Hello-SimpleAI model: "ChatGPT" = AI, "Human" = human
        if "chatgpt" in label or label == "fake" or label == "label_1":
            model_ai_scores.append(score)
        else:
            model_ai_scores.append(1.0 - score)

    model_avg = sum(model_ai_scores) / len(model_ai_scores)

    # Heuristic layer — mild nudge, not the primary signal
    heuristic = heuristic_ai_score(text)

    # Blend: model is authoritative, heuristics fine-tune
    blended = 0.82 * model_avg + 0.18 * heuristic
    blended = max(0.0, min(1.0, blended))

    ai_score = blended
    human_score = 1.0 - blended

    verdict = "AI-Generated" if ai_score >= 0.5 else "Human-Written"
    confidence = max(ai_score, human_score)

    # Confidence tier label
    if confidence >= 0.85:
        confidence_label = "High confidence"
    elif confidence >= 0.65:
        confidence_label = "Moderate confidence"
    else:
        confidence_label = "Low confidence"

    return {
        "verdict": verdict,
        "ai_score": round(ai_score * 100, 1),
        "human_score": round(human_score * 100, 1),
        "confidence": round(confidence * 100, 1),
        "confidence_label": confidence_label,
        "chunks_analyzed": len(chunks),
        "word_count": len(text.split()),
        "heuristic_score": round(heuristic * 100, 1),
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    url = data.get("url", "").strip()

    if not text and not url:
        return jsonify({"error": "Provide text or a URL."}), 400

    scraped_url = None
    if url:
        try:
            text = scrape_text(url)
            scraped_url = url
        except Exception as e:
            return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 400

    if len(text.split()) < 20:
        return jsonify({"error": "Text is too short — need at least 20 words."}), 400

    try:
        result = analyze_text(text)
        if scraped_url:
            result["scraped_url"] = scraped_url
            result["scraped_preview"] = text[:400] + "..." if len(text) > 400 else text
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
