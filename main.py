import asyncio
import os
import sys
import random
import sys
import time
from typing import List, Dict

import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from intent_classifier import classify_intent
from prompt_engineering import brief_prompt, poetic_prompt, context_enhanced_prompt, format_card_info

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

tarot_deck: List[Dict[str, Dict[str, str]]] = [
    {
        "name": "The Fool",
        "meanings": {
            "upright": "New beginnings, spontaneity, free spirit, taking a leap of faith.",
            "reversed": "Recklessness, fear of the unknown, foolish behavior, poor judgment.",
        },
    },
    {
        "name": "The Magician",
        "meanings": {
            "upright": "Manifestation, resourcefulness, power, inspired action.",
            "reversed": "Manipulation, deception, untapped potential, illusions.",
        },
    },
    {
        "name": "The High Priestess",
        "meanings": {
            "upright": "Intuition, subconscious, mystery, inner wisdom.",
            "reversed": "Secrets, withdrawal, blocked intuition, hidden motives.",
        },
    },
    {
        "name": "Two of Swords",
        "meanings": {
            "upright": "Indecision, difficult choices, blocked emotions, avoidance.",
            "reversed": "Lies being exposed, confusion, lesser of two evils, no right choice.",
        },
    },
    {
        "name": "Ace of Cups",
        "meanings": {
            "upright": "New emotional beginnings, love, compassion, joy.",
            "reversed": "Emotional loss, emptiness, blocked feelings, repressed emotions.",
        },
    },
    {
        "name": "Ten of Pentacles",
        "meanings": {
            "upright": "Wealth, family, legacy, long-term success, stability.",
            "reversed": "Loss of legacy, family conflict, instability, broken traditions.",
        },
    },
]


def draw_cards(n: int = 3) -> List[Dict]:
    cards = random.sample(tarot_deck, n)
    for card in cards:
        card["orientation"] = random.choice(["upright", "reversed"])
    return cards


def load_documents(filepath: str):
    loader = PyMuPDFReader()
    return loader.load(filepath)


def setup_model():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3")
    llm_wrapped = LangChainLLM(llm)
    Settings.llm = llm_wrapped
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=50)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in .env file")
PINECONE_INDEX_NAME = "llama-index"
PINECONE_ENV = "us-east-1"
pc = Pinecone(api_key=PINECONE_API_KEY)
region_spec = ServerlessSpec(cloud="aws", region="us-east-1")


def get_index(filepath: str, force_rebuild: bool = False) -> VectorStoreIndex:
    persist_dir = "./storage"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=region_spec,
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index)

    if not force_rebuild:
        try:
            print("Trying to load existing index from storage...")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=persist_dir,
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Failed to load index: {e}. Proceeding to rebuild...")

    documents = load_documents(filepath)
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store,
        persist_dir=persist_dir,
    )

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


# Streamlit UI
st.set_page_config(page_title="📄 Document Chatbot", layout="wide")
st.title("📄 Document Q&A Chatbot")
st.markdown(
    "Ask questions about tarot cards. The model uses interpretations from a tarot knowledge base PDF."
)
if "chat_engine" not in st.session_state:
    with st.spinner("Setting up model and index (this may take a minute)..."):
        setup_model()
        index = get_index("sample_tarot_meanings.pdf", force_rebuild=True)
        retriever = index.as_retriever(similarity_top_k=3)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=Settings.llm,
        )
        st.session_state.chat_engine = chat_engine
        st.session_state.retriever = retriever
        st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the document:")
if user_input:
    with st.spinner("Thinking..."):
        intent = classify_intent(user_input)
        st.write(f"🔍 Detected Intent: {intent}")
        user_style = st.selectbox(
            "Choose reading style:", ["brief", "poetic", "contextual"]
        )

        num_cards = 3
        drawn_cards = draw_cards(num_cards)
        drawn_card_names = [f"{c['name']} ({c['orientation']})" for c in drawn_cards]
        st.write("🃏 Drawn Cards:", ", ".join(drawn_card_names))

        card_text = format_card_info(drawn_cards)
        st.markdown(card_text)
        if user_style == "brief":
            prompt = brief_prompt(user_input, drawn_cards)
        elif user_style == "poetic":
            prompt = poetic_prompt(user_input, drawn_cards)
        elif user_style == "contextual":
            retrieved_nodes = st.session_state.retriever.retrieve(user_input)
            context = "\n".join([node.get_content() for node in retrieved_nodes])
            prompt = context_enhanced_prompt(user_input, drawn_cards, intent, context)

       

        response = st.session_state.chat_engine.chat(prompt)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response.response))
        st.write(f"**Bot:** {response.response}")
