"""Checks the chatbot on paraphrased questions it has never seen (run: python test_chatbot.py)."""

from chatbot import FAQChatbot

# (user question, expected FAQ category or "small_talk"/"fallback", keyword expected in the answer)
TEST_CASES = [
    ("where is my parcel", "Orders", "Track"),
    ("how to follow my order status", "Orders", "Track"),
    ("i want to cancel my purchase", "Orders", "Cancel"),
    ("how long will delivery take", "Shipping", "business days"),
    ("is delivery free?", "Shipping", "$50"),
    ("do you deliver to canada", "Shipping", "countries"),
    ("my order still hasn't arrived", "Shipping", "delay"),
    ("can i send back a product i don't like", "Returns", "Return"),
    ("when do i get my money back", "Returns", "5 business days"),
    ("the item arrived broken", "Returns", "replacement"),
    ("i got the wrong product", "Returns", "correct item"),
    ("can i pay with paypal", "Payment", "PayPal"),
    ("do you accept credit cards", "Payment", "Visa"),
    ("my card was refused", "Payment", "card details"),
    ("my discount code doesnt work", "Payment", "Promo code"),
    ("i forgot my pasword", "Account", "Forgot password"),  # typo on purpose
    ("how do i make an account", "Account", "Sign up"),
    ("can i buy without registering", "Account", "guest"),
    ("is there a size chart", "Products", "Size guide"),
    ("are the products genuine", "Products", "authentic"),
    ("reward points", "Loyalty", "point"),
    ("student discount", "Loyalty", "10%"),
    ("how can i reach customer support", "Contact", "support@shopease.example"),
    ("hello", "small_talk", ""),
    ("thanks a lot", "small_talk", ""),
    ("what is the capital of france", "fallback", ""),
    ("tell me a joke about cats", "fallback", ""),
]


def main():
    bot = FAQChatbot()
    passed = 0
    for question, expected, keyword in TEST_CASES:
        reply = bot.get_response(question)
        got = reply.kind if reply.kind in {"small_talk", "fallback"} else reply.category
        ok = got == expected and keyword in reply.answer
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {reply.score:.2f}  {reply.kind:<10} {question!r:<45} -> {reply.matched_question or got}")
    print(f"\n{passed}/{len(TEST_CASES)} passed ({passed / len(TEST_CASES):.0%})")


if __name__ == "__main__":
    main()
