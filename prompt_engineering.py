
def format_card_info(cards):
    """
    Format card names, orientation, and meanings into a string.
    """
    return "\n".join([
        f"{card['name']} ({card['orientation']}): {card['meanings'][card['orientation']]}"
        for card in cards
    ])

def brief_prompt(question, cards , intent=None):
    card_text = format_card_info(cards)
    return (
        f"Act as a tarot expert.\n\n"
        f"Cards:\n{card_text}\n\n"
        f"User Question: {question}\n"
        f"Reply briefly (2–3 sentences)."
    )


def poetic_prompt(question, cards):
    card_text = format_card_info(cards)
    return (
        f"You are a mystical tarot oracle. Speak in poetic and symbolic language.\n\n"
        f"Cards:\n{card_text}\n\n"
        f"Question: {question}\n"
        f"Give a symbolic , respond in poetic style and intuitive response in 3–5 sentences."
    )


def context_enhanced_prompt(question, cards, intent=None, extra_context=None):
    card_text = format_card_info(cards)
    context = extra_context if extra_context else "No extra context available."
    return (
        f"You are a tarot expert with deep knowledge.\n"
        f"User Intent: {intent}\n\n"
        f"Drawn Cards:\n{card_text}\n\n"
        f"Related Knowledge:\n{context}\n\n"
        f"User Question: {question}\n"
        f"Provide a clear, symbolic answer (max 1000 tokens) and Answer insightfully using the cards and context."
    )
