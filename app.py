"""CodeAlpha Task 2 - FAQ Chatbot (Streamlit chat UI)."""

import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

WELCOME = (
    "Hi! 👋 I'm the **ShopEase** support assistant. Ask me about orders, shipping, "
    "returns, payment, your account and more — or pick a question in the sidebar."
)
KIND_LABELS = {"faq": "✅ Confident match", "unsure": "🤔 Possible match", "small_talk": "💬 Small talk", "fallback": "❓ No match"}


@st.cache_resource(show_spinner="Loading NLP model…")
def load_bot() -> FAQChatbot:
    return FAQChatbot()


bot = load_bot()
st.session_state.setdefault("messages", [{"role": "assistant", "content": WELCOME}])
st.session_state.setdefault("pending", None)


def ask(question: str):
    st.session_state.pending = question


# ---------- sidebar ----------
with st.sidebar:
    st.header("💬 FAQ Chatbot")
    st.caption(bot.topic)
    show_details = st.toggle("Show NLP details", value=True, help="Display the similarity score and matched FAQ")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

    st.subheader("Browse questions")
    for category, questions in bot.categories.items():
        with st.expander(category):
            for question in questions:
                st.button(question, key=f"side-{question}", on_click=ask, args=(question,), use_container_width=True)

    st.divider()
    st.caption(f"{len(bot.faqs)} FAQs · {len(bot.questions)} question variants · NLTK + TF-IDF + cosine similarity")

# ---------- chat ----------
st.title("💬 ShopEase Support")
st.caption("CodeAlpha AI Internship — Task 2: FAQ Chatbot")


def render_details(meta: dict):
    lines = [f"**{KIND_LABELS[meta['kind']]}** · similarity score: `{meta['score']:.2f}`"]
    if meta.get("matched_question"):
        lines.append(f"Matched FAQ: *{meta['matched_question']}* ({meta['category']})")
    lines.append(f"Preprocessed input: `{meta['cleaned_input'] or '∅'}`")
    st.caption("  \n".join(lines))


for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if meta and show_details:
            render_details(meta)
        # Suggestion buttons only under the latest answer
        if meta and meta["suggestions"] and i == len(st.session_state.messages) - 1:
            st.caption("Related questions:")
            for j, suggestion in enumerate(meta["suggestions"]):
                st.button(suggestion, key=f"sugg-{i}-{j}", on_click=ask, args=(suggestion,))

user_input = st.chat_input("Type your question…") or st.session_state.pending
st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply = bot.get_response(user_input)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply.answer,
            "meta": {
                "kind": reply.kind,
                "score": reply.score,
                "matched_question": reply.matched_question,
                "category": reply.category,
                "cleaned_input": reply.cleaned_input,
                "suggestions": reply.suggestions,
            },
        }
    )
    st.rerun()
