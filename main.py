import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import time
import random
import pinecone
import streamlit as st 
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from langchain_ollama import ChatOllama
from llama_index.llms.langchain import LangChainLLM
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    )
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec # Pinecone vector store
# from llama_index.core.node_parser import SimpleNodeParser
from intent_classifier import classify_intent

tarot_deck = [
    {
        "name": "The Fool",
        "meanings": {
            "upright": "New beginnings, spontaneity, free spirit, taking a leap of faith.",
            "reversed": "Recklessness, fear of the unknown, foolish behavior, poor judgment."
        }
    },
    {
        "name": "The Magician",
        "meanings": {
            "upright": "Manifestation, resourcefulness, power, inspired action.",
            "reversed": "Manipulation, deception, untapped potential, illusions."
        }
    },
    {
        "name": "The High Priestess",
        "meanings": {
            "upright": "Intuition, subconscious, mystery, inner wisdom.",
            "reversed": "Secrets, withdrawal, blocked intuition, hidden motives."
        }
    },
    {
        "name": "Two of Swords",
        "meanings": {
            "upright": "Indecision, difficult choices, blocked emotions, avoidance.",
            "reversed": "Lies being exposed, confusion, lesser of two evils, no right choice."
        }
    },
    {
        "name": "Ace of Cups",
        "meanings": {
            "upright": "New emotional beginnings, love, compassion, joy.",
            "reversed": "Emotional loss, emptiness, blocked feelings, repressed emotions."
        }
    },
    {
        "name": "Ten of Pentacles",
        "meanings": {
            "upright": "Wealth, family, legacy, long-term success, stability.",
            "reversed": "Loss of legacy, family conflict, instability, broken traditions."
        }
    }
]


def draw_cards(n=3):
    cards = random.sample(tarot_deck, n)
    for card in cards:
        card['orientation'] = random.choice(["upright", "reversed"])
    return cards

def load_documents(filepath):
    loader = PyMuPDFReader()
    return loader.load(filepath)

    
def Setup_Model():
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
# Pinecone v3 client
pc = Pinecone(api_key=PINECONE_API_KEY)
region_spec = ServerlessSpec(cloud='aws', region='us-east-1')



from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore


def GetIndex(filepath, force_rebuild=False):
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
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Failed to load index: {e}. Proceeding to rebuild...")

    # Explicitly define all storage components
    documents = load_documents(filepath)
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store,
        persist_dir=persist_dir
    )

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    return index



# --- Streamlit UI starts here ---

# Set page title and layout
st.set_page_config(page_title="📄 Document Chatbot", layout="wide")   # <-- NEW: Streamlit page config

st.title("📄 Document Q&A Chatbot")                                  # <-- NEW: Page title
st.markdown("Ask questions about tarot cards. The model uses interpretations from a tarot knowledge base PDF.")  # <-- NEW: Subtitle

# Initialize model and index only once, stored in Streamlit session state
if 'chat_engine' not in st.session_state:                          # <-- NEW: Persistent session state
    with st.spinner("Setting up model and index (this may take a minute)..."):   # <-- NEW: Loading spinner
        Setup_Model()                                                # <-- SAME function call
        index = GetIndex("sample_tarot_meanings.pdf", force_rebuild=True)                   # <-- SAME but you can change file path here
        retriever = index.as_retriever(similarity_top_k=3)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=Settings.llm
        )
        st.session_state.chat_engine = chat_engine                  # <-- NEW: Store chat engine in session state
        st.session_state.chat_history = []                          # <-- NEW: Store chat history


# User text input box for questions
user_input = st.text_input("Ask a question about the document:")   # <-- NEW: Input box for user question

if user_input:
    with st.spinner("Thinking..."):
        # 1. Classify user intent
        intent = classify_intent(user_input)
        st.write(f"🔍 Detected Intent:** {intent}")

        # 2. Draw cards based on intent
        num_cards = 3
        drawn_cards = draw_cards(num_cards)
        drawn_card_names = [f"{c['name']} ({c['orientation']})" for c in drawn_cards]
        st.write("🃏 Drawn Cards:", ", ".join(drawn_card_names))

        # 3. Compose the query to send to the LLM
        card_text = "\n".join([f"{c['name']} ({c['orientation']}): {c['meanings'][c['orientation']]}" for c in drawn_cards])
        prompt = (
    f"Act as a tarot expert. Based on the following cards and meanings:\n"
    f"{card_text}\n\n"
    f"Answer this question: {user_input}\n"
    f"Be concise and direct."
)


        # 4. Send prompt to LLM
        response = st.session_state.chat_engine.chat(prompt)

        # 5. Save conversation history
        st.session_state.chat_history.append(("You", user_input)) 
        st.session_state.chat_history.append(("Bot", response.response))


# Display the chat history below the input box
if st.session_state.chat_history:                                 # <-- NEW: Show conversation history
    for role, text in st.session_state.chat_history:
        st.write(f"{role}:** {text}")