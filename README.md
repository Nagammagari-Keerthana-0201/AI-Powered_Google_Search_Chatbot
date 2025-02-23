# AI-Powered Google Search Chatbot

This project is an AI-powered chatbot that searches Google for answers and enhances responses using OpenAI's API. Built with **Streamlit**, it provides an interactive chat interface.

## 🚀 Features
- **Google Search Integration**: Fetches web results for user queries.
- **Web Scraping**: Extracts relevant content from search results.
- **FAISS Indexing**: Finds the most relevant information.
- **OpenAI GPT-4 API**: Generates refined responses.
- **Streamlit UI**: Interactive chatbot experience.
- **Chat History**: Keeps track of previous interactions.

---

## 📌 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies
Make sure you have Python **3.8+** installed.
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up OpenAI API Key
#### Option 1: Use Streamlit Secrets (Recommended for Deployment)
1. **Deploy on Streamlit Cloud** (see below for steps).
2. Go to **Streamlit App Dashboard** → Click **"⋮ More"** → **"Edit Secrets"**.
3. Add the following:
   ```
   OPENAI_API_KEY = "your-openai-api-key"
   ```
4. Click **Save**.

#### Option 2: Use Environment Variables (For Local Setup)
For **Windows (Command Prompt)**:
```sh
set OPENAI_API_KEY=your-secret-key
```
For **Mac/Linux (Terminal)**:
```sh
export OPENAI_API_KEY="your-secret-key"
```

#### Option 3: Store in a `key.txt` File (⚠️ Not Recommended for Public Repos)
Create a file `key.txt` in the project directory and paste your OpenAI API key inside.

---

## 🎯 Run the App Locally
```sh
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501/`.

---

## 🌍 Deploy on Streamlit Cloud (Without Git Commands)

### 1️⃣ Upload the Project to GitHub
- Go to **[GitHub](https://github.com/)** and create a **new repository**.
- Upload your project files manually (or use GitHub Desktop).

### 2️⃣ Deploy on Streamlit Cloud
1. Go to **[Streamlit Cloud](https://share.streamlit.io/)**.
2. Click **"New App"**.
3. Select **GitHub repository**.
4. Choose your repository & branch.
5. Set the **main file path** (e.g., `app.py`).
6. Click **Deploy**.

### 3️⃣ Set Up Secrets (For OpenAI Key)
1. After deployment, go to **Streamlit App Dashboard**.
2. Click **"⋮ More"** → **"Edit Secrets"**.
3. Add:
   ```
   OPENAI_API_KEY = "your-api-key"
   ```
4. Click **Save** and **rerun the app**.

---

## 🛠️ Technologies Used
- **Python** (Backend logic)
- **Streamlit** (UI Framework)
- **OpenAI API** (Chatbot intelligence)
- **Google Search API** (Fetching results)
- **BeautifulSoup** (Web scraping)
- **FAISS** (Efficient similarity search)

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 🤝 Contributions
Feel free to **fork** the repository and submit **pull requests**.

---

## 📩 Contact
For any queries, reach out via GitHub Issues.

---

**🌟 Don't forget to Star the repository if you find this helpful! 🌟**

