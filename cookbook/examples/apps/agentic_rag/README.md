# 💎 Agentic RAG

Agentic RAG is a Streamlit-based application that allows you to explore and question your own documents using powerful LLMs like GPT-4, Claude 3.5, Gemini, and more — powered by AGNO's agentic framework and a persistent PostgreSQL + `pgvector` vector store.

> Upload documents, ask complex questions, and get contextualized responses grounded in your knowledge base.

---

## 🚀 Features

- 📄 Upload PDF, CSV, and TXT files for ingestion
- 🔍 Embedded document search using vector similarity via `pgvector`
- 🧠 Agentic architecture with memory and tool usage
- 🧵 Persistent chat sessions and knowledge state
- 🧰 Choose between multiple LLMs (OpenAI, Anthropic, Google, Groq)
- 🧨 Safe reset mechanism for vector DB cleanup
- 📦 Packaged with Docker Compose for simple setup

---

## 🛠️ Requirements

Before getting started, ensure the following are installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- `git`

---

## 🧾 Installation Instructions

Clone the repository and navigate into it:

```bash
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag
```

🔐 Environment Setup
Create a .env file in the project root and add your API keys:

```bash
cp .env.example .env
```


```
# =======================
# 🔐 API Keys
# =======================
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=your-groq-api-key

# =======================
# 🧠 PostgreSQL + pgvector
# =======================
# This is the default connection string for the internal Docker container.
# You likely don't need to change this unless using an external DB.
DATABASE_URL=postgresql+psycopg://ai:ai@pgvector:5432/ai
```
⚠️ The DATABASE_URL is already configured for the internal Docker PostgreSQL container. You only need to update this if you're running Postgres separately.

🐳 Running the App
Start the system with Docker Compose:

```bash
docker-compose up --build
```

Once it's ready, open your browser and go to:

```html
http://localhost:8501
```

# 📚 Using the App

## Step 1: Select an LLM
From the sidebar, choose your preferred model. Options include:

gpt-4o

claude-3-5-sonnet

gemini-2.0-flash-exp

llama-3.3-70b

o3-mini

## Step 2: Upload Documents
Use the "Add Documents" section in the sidebar to upload .pdf, .csv, or .txt files.

You can upload multiple files at once. The app will parse, chunk, embed, and insert them into a vector database automatically.

## Step 3: Ask Questions
Use the chat interface to ask questions like:

"Summarize the uploaded documents."

"What is the workflow described in this guide?"

"What are the differences between the new and old search processes?"

The assistant will retrieve relevant chunks and answer based on their contents.

### 🧹 Reset Knowledge Base
In the sidebar, under 🚨 Dangerous Actions, you can:

Click 🧨 Clear Knowledge Base

Confirm with ✅ Confirm Clear

This deletes all indexed documents from the vector database.


### 💾 Export Chat
Use the 💾 Export Chat button to download your current conversation as a Markdown file.

### 🧠 How It Works
Documents are processed by reader modules and converted into Document chunks.

These chunks are embedded (using OpenAI/Groq/etc.) and stored in a Postgres table with pgvector.

When you send a question, it’s embedded and matched against stored vectors using cosine similarity.

Top-k results are passed into the agent to generate a coherent, grounded response.

### 🙋 Need Help?
If you're running into errors:

Make sure Docker is running

Check that your .env file has valid API keys

Look at logs in the terminal for traceback errors

For database connection issues, verify that port 5432 is not blocked

### Contributing
Pull requests and issues are welcome. If you're extending it with new features (e.g. vector rerankers, model configs, etc.), be sure to open a discussion first!

Happy hacking, and enjoy your ✨ Agentic RAG ✨ journey!