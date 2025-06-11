# intent_classifier.py
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

intent_examples = {
    "yes_no": [
        "Will I get the job?", "Should I stay in this relationship?",
        "Am I on the right path?", "Is this the right time?",
        "Can I trust them?", "Will it work out?"
    ],
    "timeline": [
        "When will I get married?", "How long until I find success?",
        "When should I make a move?", "How soon will it happen?"
    ],
    "guidance": [
        "What should I do next?", "Where should I focus?",
        "Give me advice for now.", "How can I improve myself?"
    ],
    "insight": [
        "Why do I feel lost?", "What is causing this block?",
        "Explain this confusion.", "What is the root cause?"
    ],
    "general": [
        "Tell me what the universe wants me to know.",
        "Draw a general reading.", "Give me a message for today."
    ]
}

# Pre-compute intent embeddings
intent_embeddings = {
    intent: [model.encode(q) for q in samples]
    for intent, samples in intent_examples.items()
}

def classify_intent(question, threshold=0.5):
    query_emb = model.encode(question)
    avg_scores = {}
    for intent, vectors in intent_embeddings.items():
        sims = [util.cos_sim(query_emb, vec)[0][0].item() for vec in vectors]
        avg_scores[intent] = sum(sims) / len(sims)
    print(avg_scores)  # debug line to see scores
    best_intent = max(avg_scores, key=avg_scores.get)
    if avg_scores[best_intent] >= threshold:
        return best_intent
    return "general"

