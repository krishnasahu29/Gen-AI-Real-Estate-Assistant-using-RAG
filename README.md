# 🏠 Gen AI Real Estate Assistant
A sophisticated AI-powered real estate assistant built using Retrieval-Augmented Generation (RAG) technology and Streamlit frontend. This application helps to find relevant data from a source website related to 
real estates.
### 🚀 Features

Process URLs: Give 1-3 urls to the model.
Answer relevant questions: It gives relevant answers related to the URLs given as input

### 🛠️ Technology Stack

Frontend: Streamlit
AI Framework: LangChain
Vector Database: Chroma
LLM: Groq/Llama
Embeddings: Sentence Transformers
Data Sources: URLs
Backend: Python, FastAPI (optional)

### 📋 Prerequisites

Python 3.8+
Groq API key (or other LLM provider)
Real estate URLs
Git

### 🔧 Installation

Clone the repository

bashgit clone https://github.com/krishnasahu29/Gen-AI-Real-Estate-Assistant-using-RAG.git
cd Gen-AI-Real-Estate-Assistant-using-RAG
Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Set up environment variables

bashcp .env.example .env
# Edit .env with your API keys and configuration

Initialize vector database

bashpython scripts/setup_vectordb.py
### ⚙️ Configuration
Create a .env file with the following variables:
envOPENAI_API_KEY=your_openai_api_key
STREAMLIT_SERVER_PORT=8501
VECTOR_DB_PATH=./data/vectordb
REAL_ESTATE_API_KEY=your_real_estate_api_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
CHUNK_SIZE=500
CHUNK_OVERLAP=50
### 🚀 Usage

Start the Streamlit application

bashstreamlit run app.py

Access the application
Open your browser and navigate to http://localhost:8501
Upload URLs

### 📱 Application Screenshots

![image](https://github.com/user-attachments/assets/272083d8-4514-4a0d-aacd-2bcab7755800)


### 🤖 RAG Implementation Details
Document Processing Pipeline

Document Ingestion: URLs
Text Chunking: Semantic chunking with overlap for context preservation
Embedding Generation: Convert text chunks to vector embeddings
Vector Storage: Store embeddings in vector database with metadata

Retrieval Strategy

Semantic Search: Find most relevant documents using cosine similarity
Hybrid Search: Combine semantic and keyword-based search
Re-ranking: Post-retrieval re-ranking for improved relevance
Context Window Management: Optimize retrieved context for LLM input

Generation Enhancement

Prompt Engineering: Specialized prompts for real estate domain
Context Integration: Seamlessly blend retrieved information with user queries
Source Attribution: Track and display information sources
Hallucination Mitigation: Grounding responses in retrieved documents

### 📁 Project Structure
gen-ai-real-estate-assistant/
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── 
├── resources/
│   ├── vectorstore
├── 
├── scripts/
│   ├── main.py                # Implementing main streamlit app
|   ├── .env                   # Environment variables template
│   └── rag.py                 # RAG training file
├── 
└── screenshots/               # Application screenshots
    ├── dashboard.png

🔮 Future Enhancements

 Multi-modal support (image analysis for property photos)
 Voice interface integration
 Real-time market data streaming
 Advanced analytics dashboard
 Mobile application development
 Integration with CRM systems
 Multilingual support

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

🙏 Acknowledgments

Groq for providing powerful language models
Streamlit team for the excellent web framework
LangChain/LlamaIndex communities for RAG tools
Real estate data providers and APIs

📞 Support
If you have any questions or need support, please reach out on:
GitHub: @krishnasahu29
LinkedIn: www.linkedin.com/in/krishnasahu29
Email: krishna.sahu.work222@gmail.com
