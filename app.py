from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
import os

# Initializing Flask App
app = Flask(__name__)

# Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Vibhav123%",
    "database": "chatbot_db"
}

VECTORSTORE_PATH = "faiss_index"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "mistral"


# Step 1: Corpus Preparation
def prepare_corpus(corpus_dir):
    print("Loading and preparing corpus...")
    documents = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(corpus_dir, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# Step 2: Embedding and Vector Store
def initialize_vectorstore(chunks):
    print("Initializing vector store...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore


# Load or create vector store
def load_or_create_vectorstore(corpus_dir):
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vector store...")
        return FAISS.load_local(VECTORSTORE_PATH, OllamaEmbeddings(model=EMBEDDING_MODEL),allow_dangerous_deserialization=True)
    else:
        chunks = prepare_corpus(corpus_dir)
        return initialize_vectorstore(chunks)


# Step 3: LLM for RAG
def initialize_rag_pipeline(vectorstore):
    print("Initializing RAG pipeline...")
    llm = OllamaLLM(model="mistral")
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    return qa_chain


# Step 4: Database Setup
def setup_database():
    print("Setting up database...")
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vibhav123%",
        database="chatbot_db"
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role VARCHAR(10),
            content TEXT
        )
        """
    )
    conn.commit()
    cursor.close()
    conn.close()



# Saving chat history
def save_to_database(role, content):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vibhav123%",
        database="chatbot_db"
    )
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (role, content) VALUES (%s, %s)", (role, content)
    )
    conn.commit()
    conn.close()


# Retrieve chat history
def retrieve_history():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vibhav123%",
        database="chatbot_db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_history")
    rows = cursor.fetchall()
    conn.close()
    return rows


# Initialize components
CORPUS_DIR = "C:/Users/deepa/PycharmProjects/ChatbotHistorySQL/corpus"
vectorstore = load_or_create_vectorstore(CORPUS_DIR)
qa_chain = initialize_rag_pipeline(vectorstore)
setup_database()


# Flask Endpoints
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # Generating answer using RAG pipeline
    response = qa_chain.invoke(query)

    # Extracting text response if response is a dictionary
    answer_text = response["result"] if isinstance(response, dict) else str(response)

    # Saving to database
    save_to_database("user", query)
    save_to_database("system", answer_text)

    return jsonify({"answer": answer_text})



@app.route("/history", methods=["GET"])
def history():
    rows = retrieve_history()
    history = [
        {"id": row[0], "timestamp": row[1], "role": row[2], "content": row[3]} for row in rows
    ]
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)
