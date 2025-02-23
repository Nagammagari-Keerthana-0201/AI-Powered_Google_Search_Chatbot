import openai
import asyncio
import requests
import time
import json
import os
from bs4 import BeautifulSoup
import nltk
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from googlesearch import search
from pathlib import Path

# Download tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")
# Load API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI()

# Initialize Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index
faiss_index = None
sentences = []
metadata = []

# Chat history file
HISTORY_FILE = "chat_history.json"

def load_history():
    if Path(HISTORY_FILE).exists():
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

if "history" not in st.session_state:
    st.session_state.history = load_history()

last_request_time = 0
def rate_limited_request():
    global last_request_time
    elapsed_time = time.time() - last_request_time
    if elapsed_time < 20:
        time.sleep(20 - elapsed_time)
    last_request_time = time.time()

def search_google(query, num_results=5):
    try:
        return [url for url in search(query, num_results=num_results)]
    except Exception as e:
        return [f"Error: {e}"]

def extract_data(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        sentences = nltk.sent_tokenize(" ".join(paragraphs))
        return {"url": url, "sentences": sentences}
    return None

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
    
    if not all_sentences:
        st.error("No data found to build index.")
        return
    
    embeddings = model.encode(all_sentences).astype(np.float32)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    sentences = all_sentences

def find_similar(query, top_k=3):
    if faiss_index is None:
        return []
    query_vector = model.encode([query]).astype(np.float32)
    D, I = faiss_index.search(query_vector, top_k)
    return [(sentences[i], metadata[i]) for i in I[0] if i < len(sentences)]

async def chat_with_openai(prompt):
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def chat_with_openai_sync(prompt):
    return asyncio.run(chat_with_openai(prompt))

# Streamlit UI
st.title("AI-Powered Google Search Chatbot")
st.write("Ask a question, and I'll find the answer from Google and AI!")

st.markdown("""
    <style>
        .chat-container { max-width: 700px; margin: auto; }
        .user-message { background-color: #005eff; color: white; padding: 10px; border-radius: 8px; text-align: right; margin: 5px 0; }
        .bot-message { background-color: #2a2a2a; color: white; padding: 10px; border-radius: 5px; text-align: left; margin: 5px 0; }
        .stTextInput>div>div>input { background-color: #1e1e1e; color: white; }
    </style>
""", unsafe_allow_html=True)

st.subheader("💬 Chat History")
chat_container = '<div class="chat-container">'
for chat in st.session_state.history:
    chat_container += f'<div class="user-message"><b>You:</b> {chat["query"]}</div>'
    chat_container += f'<div class="bot-message"><b>Bot:</b>{chat["response"]}</div>'
chat_container += "</div>"
st.markdown(chat_container, unsafe_allow_html=True)

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
                answer = chat_with_openai_sync(f"Use this context: {context}. Answer: {user_input}")
            else:
                answer = chat_with_openai_sync(user_input)
            
            st.session_state.history.append({"query": user_input, "response": answer})
            save_history(st.session_state.history)

with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        save_history([])
        st.rerun()

st.markdown("<p style='text-align: center;'>Powered by Google</p>", unsafe_allow_html=True)
