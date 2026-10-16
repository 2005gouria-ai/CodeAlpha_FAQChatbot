"""
FAQ Chatbot engine.

Pipeline:
  1. Collect FAQs (question + variations + answer) from faqs.json.
  2. Preprocess the text with NLTK (tokenize, lowercase, remove stopwords,
     lemmatize).
  3. Vectorize every FAQ question with TF-IDF.
  4. Match a user question to the most similar FAQ using cosine similarity.
  5. Return the best matching answer.
"""

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# NLTK setup (with a safe fallback if the corpora are not available)
# --------------------------------------------------------------------------
_NLTK_READY = False


def _ensure_nltk():
    """
    Make sure the NLTK corpora we need are available.

    Tries to download anything missing. Returns True only if the resources
    are actually usable afterwards, so the caller can fall back cleanly.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return True
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        resources = {
            "punkt": "tokenizers/punkt",
            "punkt_tab": "tokenizers/punkt_tab",
            "stopwords": "corpora/stopwords",
            "wordnet": "corpora/wordnet",
            "omw-1.4": "corpora/omw-1.4",
        }
        for pkg, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

        # Verify the resources really work before committing to NLTK.
        word_tokenize("is this thing working")
        stopwords.words("english")
        WordNetLemmatizer().lemmatize("running", "v")

        _NLTK_READY = True
        return True
    except Exception:
        return False


class Preprocessor:
    """Turns raw text into a clean bag of lemmatized tokens."""

    def __init__(self):
        self.use_nltk = _ensure_nltk()
        if self.use_nltk:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize

            self._word_tokenize = word_tokenize
            self._lemmatizer = WordNetLemmatizer()
            try:
                self._stopwords = set(stopwords.words("english"))
            except Exception:
                self._stopwords = _BASIC_STOPWORDS
        else:
            self._stopwords = _BASIC_STOPWORDS

    def clean(self, text):
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.strip()

    def tokens(self, text):
        text = self.clean(text)
        if self.use_nltk:
            try:
                words = self._word_tokenize(text)
                return [self._lemmatizer.lemmatize(w) for w in words
                        if w not in self._stopwords and len(w) > 1]
            except Exception:
                pass  # fall through to the simple tokenizer
        return [w for w in text.split()
                if w not in self._stopwords and len(w) > 1]

    def normalize(self, text):
        """Return the preprocessed text as a single string (for TF-IDF)."""
        return " ".join(self.tokens(text))


_BASIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "as",
    "and", "or", "but", "if", "then", "so", "than", "that", "this",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "do", "does", "did", "can", "could", "will",
    "would", "should", "may", "might", "must", "have", "has", "had",
    "not", "no", "yes", "from", "up", "down", "out", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "how", "what", "when",
    "where", "which", "who", "whom", "why",
}


class FAQChatbot:
    """A retrieval-based chatbot that answers from a fixed FAQ set."""

    def __init__(self, faqs_path=None, threshold=0.30):
        if faqs_path is None:
            faqs_path = os.path.join(os.path.dirname(__file__), "faqs.json")
        self.threshold = threshold
        self.pre = Preprocessor()
        self.records = self._load(faqs_path)
        self._build_index()

    # -- data -----------------------------------------------------------
    def _load(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        records = []
        for item in data["faqs"]:
            # Every phrasing of a question points back to the same answer.
            phrasings = [item["question"]] + item.get("variations", [])
            records.append({
                "question": item["question"],
                "phrasings": phrasings,
                "answer": item["answer"],
            })
        self.topic = data.get("topic", "FAQs")
        return records

    # -- indexing -------------------------------------------------------
    def _build_index(self):
        # One row per phrasing; remember which record each row belongs to.
        corpus, owners = [], []
        for idx, rec in enumerate(self.records):
            for phrase in rec["phrasings"]:
                norm = self.pre.normalize(phrase)
                if norm:
                    corpus.append(norm)
                    owners.append(idx)
        self._owners = owners
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(corpus)

    # -- querying -------------------------------------------------------
    def get_response(self, user_query):
        """
        Return (answer, score, matched_question).

        If the best cosine similarity is below `threshold`, `answer` is None
        and the caller can show a fallback message.
        """
        norm = self.pre.normalize(user_query)
        if not norm:
            return None, 0.0, None

        vec = self._vectorizer.transform([norm])
        sims = cosine_similarity(vec, self._matrix)[0]
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score < self.threshold:
            return None, score, None

        rec = self.records[self._owners[best]]
        return rec["answer"], score, rec["question"]
