# Task 2 — Chatbot for FAQs

A retrieval-based chatbot that answers a user's question with the most
similar FAQ from a knowledge base. Built for the CodeAlpha Artificial
Intelligence internship.

## How it works

1. **Collect FAQs** — questions, their variations, and answers live in `faqs.json`.
2. **Preprocess the text** — NLTK tokenizes, lowercases, removes stopwords and
   lemmatizes the words (falls back to a lightweight cleaner if NLTK data
   is unavailable).
3. **Match** — every FAQ phrasing is turned into a TF-IDF vector; the user's
   question is compared to all of them using **cosine similarity**.
4. **Respond** — the answer of the best-matching FAQ is returned, with a
   confidence score. Below a threshold, a friendly fallback is shown.
5. **Chat UI** — available both as a Tkinter desktop window and a terminal chat.

## Requirements

- Python 3.8+
- See `requirements.txt`

## Setup (VS Code terminal)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

NLTK corpora are downloaded automatically on first run. To fetch them
manually ahead of time:

```bash
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## Run

Desktop GUI:

```bash
python main.py
```

Terminal chat:

```bash
python main.py --cli
```

## Customising the knowledge base

Edit `faqs.json`. Each entry looks like:

```json
{
  "question": "Will I get a certificate?",
  "variations": ["do I receive a certificate", "completion certificate"],
  "answer": "Yes. On completing the required tasks you receive a ..."
}
```

Add as many entries as you like — the index rebuilds on startup.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — GUI or terminal chat |
| `chatbot.py` | NLP preprocessing + TF-IDF cosine-similarity matching |
| `faqs.json` | The FAQ knowledge base |
| `requirements.txt` | Dependencies |

## Notes

- The confidence threshold (default `0.30`) lives in `chatbot.py`; raise it
  to be stricter, lower it to answer more loosely.
