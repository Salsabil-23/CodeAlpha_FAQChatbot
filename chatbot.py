"""FAQ chatbot engine: NLTK preprocessing + TF-IDF vectors + cosine similarity."""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).parent / "data" / "faqs.json"

# Similarity thresholds (tuned on test questions, see test_chatbot.py)
CONFIDENT_THRESHOLD = 0.35  # answer directly
UNSURE_THRESHOLD = 0.20  # answer, but say we're not sure and suggest alternatives
RELATED_THRESHOLD = 0.15  # minimum score for a FAQ to be offered as a related question

# Words that change the meaning of a support question, so we keep them even though NLTK lists them as stopwords
KEEP_WORDS = {"not", "no", "cannot", "without"}

# Domain synonyms (applied after lemmatization) so different wordings map to the same term
SYNONYMS = {
    "parcel": "package",
    "purchase": "order",
    "buy": "order",
    "refuse": "decline",
    "reject": "decline",
    "genuine": "authentic",
    "original": "authentic",
    "real": "authentic",
    "make": "create",
    "open": "create",
    "reach": "contact",
    "call": "contact",
    "service": "support",
    "deliver": "ship",
    "delivery": "ship",
    "money": "refund",
    "reimburse": "refund",
    "coupon": "code",
    "voucher": "code",
}

SMALL_TALK = {
    "greeting": {
        "keywords": {"hi", "hello", "hey", "morning", "evening", "salut", "bonjour"},
        "answers": [
            "Hello! 👋 I'm the ShopEase assistant. Ask me anything about orders, shipping, returns, payment or your account.",
            "Hi there! How can I help you today?",
        ],
    },
    "thanks": {
        "keywords": {"thanks", "thank", "thx", "merci"},
        "answers": ["You're welcome! 😊 Anything else I can help with?", "Happy to help!"],
    },
    "goodbye": {
        "keywords": {"bye", "goodbye", "see", "later"},
        "answers": ["Goodbye! Have a great day. 👋", "See you soon at ShopEase!"],
    },
}


def ensure_nltk_data():
    """Download the NLTK resources we need the first time the app runs."""
    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english")) - KEEP_WORDS


def preprocess(text: str) -> str:
    """Lowercase, expand negations, strip punctuation, tokenize, remove stopwords, lemmatize, normalize synonyms."""
    text = text.lower()
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    lemmas = [
        _lemmatizer.lemmatize(_lemmatizer.lemmatize(tok, pos="v"), pos="n")
        for tok in tokens
        if tok not in _stopwords and len(tok) > 1
    ]
    return " ".join(SYNONYMS.get(lemma, lemma) for lemma in lemmas)


@dataclass
class BotResponse:
    answer: str
    kind: str  # "faq", "unsure", "small_talk", "fallback"
    score: float = 0.0
    matched_question: str | None = None
    category: str | None = None
    cleaned_input: str = ""
    suggestions: list[str] = field(default_factory=list)


class FAQChatbot:
    def __init__(self, data_file: Path = DATA_FILE):
        data = json.loads(data_file.read_text(encoding="utf-8"))
        self.topic = data["topic"]
        self.faqs = data["faqs"]

        # One row per question variant, remembering which FAQ it belongs to
        self.questions, self.faq_index = [], []
        for i, faq in enumerate(self.faqs):
            for question in faq["questions"]:
                self.questions.append(question)
                self.faq_index.append(i)
        cleaned = [preprocess(q) for q in self.questions]

        # Word-level TF-IDF captures meaning; character n-grams make matching robust to typos
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_matrix = self.word_vectorizer.fit_transform(cleaned)
        self.char_matrix = self.char_vectorizer.fit_transform(cleaned)

    @property
    def categories(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for faq in self.faqs:
            result.setdefault(faq["category"], []).append(faq["questions"][0])
        return result

    def rank(self, user_input: str) -> tuple[str, list[tuple[int, float, int]]]:
        """Return the cleaned input and (faq_id, score, best_row) sorted by similarity."""
        cleaned = preprocess(user_input)
        if not cleaned:
            return cleaned, []
        word_scores = cosine_similarity(self.word_vectorizer.transform([cleaned]), self.word_matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([cleaned]), self.char_matrix)[0]
        scores = 0.6 * word_scores + 0.4 * char_scores

        best: dict[int, tuple[float, int]] = {}
        for row, score in enumerate(scores):
            faq = self.faq_index[row]
            if faq not in best or score > best[faq][0]:
                best[faq] = (float(score), row)
        ranked = sorted(((faq, s, row) for faq, (s, row) in best.items()), key=lambda x: x[1], reverse=True)
        return cleaned, ranked

    def _small_talk(self, user_input: str) -> str | None:
        words = set(re.findall(r"[a-z]+", user_input.lower()))
        if len(words) > 5:
            return None
        for intent in SMALL_TALK.values():
            if words & intent["keywords"]:
                return random.choice(intent["answers"])
        return None

    def get_response(self, user_input: str) -> BotResponse:
        cleaned, ranked = self.rank(user_input)
        top_score = ranked[0][1] if ranked else 0.0

        # Short greetings/thanks are handled by intent matching, unless they clearly ask an FAQ
        small_talk = self._small_talk(user_input)
        if small_talk and top_score < CONFIDENT_THRESHOLD:
            return BotResponse(small_talk, "small_talk", cleaned_input=cleaned)

        if not ranked or top_score < UNSURE_THRESHOLD:
            return BotResponse(
                "Sorry, I couldn't find an answer to that. 🤔 Try rephrasing your question, "
                "or pick one of the suggestions below. You can also contact our team at support@shopease.example.",
                "fallback",
                score=top_score,
                cleaned_input=cleaned,
                suggestions=[self.faqs[i]["questions"][0] for i, _, _ in ranked[:3]],
            )

        faq_id, score, row = ranked[0]
        faq = self.faqs[faq_id]
        response = BotResponse(
            answer=faq["answer"],
            kind="faq" if score >= CONFIDENT_THRESHOLD else "unsure",
            score=score,
            matched_question=self.questions[row],
            category=faq["category"],
            cleaned_input=cleaned,
            suggestions=[self.faqs[i]["questions"][0] for i, s, _ in ranked[1:4] if s >= RELATED_THRESHOLD],
        )
        if response.kind == "unsure":
            response.answer = f"I think you're asking: *\"{faq['questions'][0]}\"*\n\n{faq['answer']}"
        return response


if __name__ == "__main__":
    # Simple command-line chat for quick testing
    bot = FAQChatbot()
    print(f"{bot.topic} — type 'quit' to exit\n")
    while (question := input("You: ").strip()).lower() not in {"quit", "exit"}:
        reply = bot.get_response(question)
        print(f"Bot ({reply.kind}, score={reply.score:.2f}): {reply.answer}\n")
