# 💬 FAQ Chatbot

**CodeAlpha Artificial Intelligence Internship — Task 2**

A customer-support chatbot for a fictional online store (**ShopEase**). It understands questions written in your own words and answers with the most similar FAQ, using **NLTK** for text preprocessing and **TF-IDF + cosine similarity** for matching. It comes with a chat interface built in **Streamlit**.

## ✨ Features

- **42 FAQs** in 8 categories (orders, shipping, returns, payment, account, products, loyalty, contact), each with several phrasings
- **NLP preprocessing with NLTK**: lowercasing, punctuation removal, tokenization, stopword removal, lemmatization
- **Similarity matching**: word-level TF-IDF (1–2 grams) + character n-gram TF-IDF (handles typos), compared with **cosine similarity**
- **Confidence levels**: confident answer, "I think you're asking…" for close matches, and a polite fallback with suggestions when nothing matches
- **Intent matching for small talk** (greetings, thanks, goodbye)
- **Chat UI**: chat history, clickable related questions, browse FAQs by category, clear chat
- **"Show NLP details" toggle**: displays the similarity score, matched FAQ and preprocessed input — useful to explain how it works

## 🛠️ Tech Stack

| Part | Technology |
|------|------------|
| UI | Streamlit |
| Preprocessing | NLTK (tokenizer, stopwords, WordNet lemmatizer) |
| Vectorization | scikit-learn `TfidfVectorizer` |
| Matching | Cosine similarity |

## 📁 Project Structure

```
CodeAlpha_FAQChatbot/
├── app.py             # Streamlit chat interface
├── chatbot.py         # NLP engine (preprocessing + matching)
├── test_chatbot.py    # Accuracy test on unseen, paraphrased questions
├── data/faqs.json     # FAQ dataset
├── requirements.txt
└── README.md
```

## 🚀 Setup

```bash
git clone https://github.com/Salsabil-23/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The required NLTK data is downloaded automatically on first run.

You can also chat in the terminal with `python chatbot.py`, or measure accuracy with `python test_chatbot.py`.

## ⚙️ How It Works

1. **Preprocessing** — every FAQ question and user message goes through the same pipeline:
   `"I can't track my parcels!"` → lowercase → expand negations → remove punctuation → tokenize → remove stopwords → lemmatize → map synonyms → `"not track package"`
2. **Vectorization** — the cleaned questions are turned into TF-IDF vectors (word 1–2 grams and character 3–5 grams).
3. **Matching** — the user's message is vectorized the same way and compared to every FAQ question with **cosine similarity**. The final score is `0.6 × word similarity + 0.4 × character similarity`.
4. **Answering**
   - score ≥ 0.35 → the matching answer
   - 0.20 – 0.35 → "I think you're asking…" + the answer + related questions
   - below 0.20 → fallback message + suggestions
   - short greetings/thanks are detected with keyword-based intent matching

## 🧪 Testing

`test_chatbot.py` asks 27 questions that are **not** in the dataset (paraphrases, typos, off-topic questions) and checks that the bot picks the right category.

**Result: 26/27 (96%).** Adding domain-synonym normalization (e.g. *purchase → order*, *parcel → package*, *refused → declined*) raised accuracy from 70% to 96%.

Known limitation: TF-IDF matches words, not meaning, so a question like *"do you deliver to Canada?"* doesn't know that Canada is a country. Using sentence embeddings (e.g. `sentence-transformers`) would be the next improvement.

## 🔧 Customize

To use another topic, edit `data/faqs.json`: change the `topic` and add objects with a `category`, a list of `questions` and an `answer`.

## 👤 Author

*Salsabil Hafsi* — CodeAlpha AI Intern · [LinkedIn](#) · [GitHub](https://github.com/Salsabil-23)
