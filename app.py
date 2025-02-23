import openai
import requests
import time
import json
from bs4 import BeautifulSoup
import nltk
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from googlesearch import search
from pathlib import Path
import streamlit as st

# Download tokenizer
nltk.download("punkt")

# Load API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

client = openai.OpenAI(api_key=openai_api_key)

# Initialize OpenAI Client
client = openai.OpenAI(api_key=openai_api_key)

# Initialize Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index
faiss_index = None
sentences = []
metadata = []

# Chat history file
HISTORY_FILE = "chat_history.json"

# Load chat history from file
def load_history():
    if Path(HISTORY_FILE).exists():
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Save chat history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = load_history()

# Rate-limiting function
last_request_time = 0
def rate_limited_request():
    global last_request_time
    elapsed_time = time.time() - last_request_time
    if elapsed_time < 20:
        time.sleep(20 - elapsed_time)
    last_request_time = time.time()

# Google search
def search_google(query, num_results=5):
    return [url for url in search(query, num_results=num_results)]

# Web scraping
def extract_data(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        sentences = nltk.sent_tokenize(" ".join(paragraphs))
        return {"url": url, "sentences": sentences}
    return None

# Build FAISS index
def build_index(urls):
    global faiss_index, sentences, metadata
    all_sentences = []
    metadata = []

    for url in urls:
        if not url.startswith("http"):
            continue
        data = extract_data(url)
        if data:
            all_sentences.extend(data["sentences"])
            metadata.extend([url] * len(data["sentences"]))

    if all_sentences:
        embeddings = model.encode(all_sentences).astype(np.float32)
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings)
        sentences = all_sentences

# Find similar sentences
def find_similar(query, top_k=3):
    if faiss_index is None:
        return []
    query_vector = model.encode([query]).astype(np.float32)
    D, I = faiss_index.search(query_vector, top_k)
    return [(sentences[i], metadata[i]) for i in I[0] if i < len(sentences)]

# OpenAI API call
def chat_with_openai(prompt):
    try:
        rate_limited_request()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("AI-Powered Google Search Chatbot")
st.write("Ask a question, and I'll find the answer from Google and AI!")

st.markdown("""
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
        }
        .user-message {
            background-color: #005eff;
            color: white;
            padding: 10px;
            border-radius: 8px;
            text-align: right;
            margin: 5px 0;
        }
        .bot-message {
            background-color: #2a2a2a;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: left;
            margin: 5px 0;
        }
        .stTextInput>div>div>input {
            background-color: #1e1e1e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Display chat history in correct order (oldest first)
st.subheader("💬 Chat History")
chat_container = '<div class="chat-container">'
for chat in st.session_state.history:  # Now iterating in correct order (first asked first)
    chat_container += f'<div class="user-message"><b>You:</b> {chat["query"]}</div>'
    chat_container += f'<div class="bot-message"><b>Bot:</b>{chat["response"]}</div>'
chat_container += "</div>"
st.markdown(chat_container, unsafe_allow_html=True)

# Chat input BELOW the chat history
user_input = st.text_input("Ask me anything:", key="user_query")
col1, spacer, col2 = st.columns([1, 2.8, 1]) 
with col1:
    if st.button("🔍 Search"):
        if user_input:
            urls = search_google(user_input)
            build_index(urls)
            top_sentences = find_similar(user_input)

            if top_sentences:
                context = " ".join([s[0] for s in top_sentences])
                answer = chat_with_openai(f"Use this context: {context}. Answer: {user_input}")
            else:
                answer = chat_with_openai(user_input)

            # Save to history
            st.session_state.history.append({"query": user_input, "response": answer})
            save_history(st.session_state.history)

# Place "Clear History" button on the right
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        save_history([])
        st.rerun()

st.markdown("<p style='text-align: center;'>Powered by Google</p>", unsafe_allow_html=True)
