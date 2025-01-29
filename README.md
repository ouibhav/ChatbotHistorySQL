# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using a vector database for semantic search and storing chat history in a MySQL database. The chatbot is served via a Flask web application.

## Features
- **Document Processing:** Loads and preprocesses text corpus.
- **Vector Store:** Uses FAISS for embedding-based retrieval.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant text chunks and generates responses using an LLM.
- **Chat History Storage:** Stores user queries and bot responses in MySQL.
- **Flask Web Interface:** Provides endpoints for querying the chatbot and retrieving conversation history.




## Installation and  Setup
 **1) Clone the repository:** 
```bash
  git clone https://github.com/ouibhav/ChatbotHistorySQL.git
```
**2) Create a virtual environment and Install Dependencies:** 
  
  ```bash
  python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**3) Set Up MySQL Database:** 

  - Install MySQL and create a database:
  

```bash
  CREATE DATABASE chatbot_db;
```
- Update app.py with your MySQL credentials.


## Running the Chatbot

**1) Start the Flask server:** 
```bash
  python app.py
```
**2) Open your browser and go to http://127.0.0.1:5000/ to access the chatbot UI.**

**3) Testing the Endpoints**

![Image](https://github.com/user-attachments/assets/194585d7-7fdb-411d-ac9b-4c7b63ffe15c)

![Image](https://github.com/user-attachments/assets/77d37c85-692a-43d3-9057-20da580f6923)

