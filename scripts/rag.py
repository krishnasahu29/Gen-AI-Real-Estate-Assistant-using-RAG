from pathlib import Path
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# === Constants ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # More stable embedding model
COLLECTION_NAME = "mortgage_articles"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
CHUNK_SIZE = 500

# === Globals ===
vector_db = None
groq_model = None


def initialize_components():
    """
    Initialize the components required for the RAG system.
    
    Returns:
        tuple: Contains the vector database and Groq model.
    """
    global vector_db, groq_model

    if vector_db is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True, "device": "cuda" if Path("/dev/nvidia0").exists() else "cpu"},
        )
        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=ef,
        )

    if groq_model is None:
        groq_model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.9, max_tokens=512)


def process_urls(urls):
    """
    Process a list of URLs and store them in a vector database.
    
    Args:
        urls (list): A list of URLs to process.
    """
    yield 'Initializing components...✅'
    initialize_components()
    vector_db.reset_collection()

    yield 'Loading documents from URLs...✅'
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents were loaded from the provided URLs.")

    yield 'Splitting documents into chunks...✅'
    r_split = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = r_split.split_documents(documents)

    if not split_docs:
        raise ValueError("Document splitting failed. No chunks generated.")

    yield 'Adding documents to vector database...✅'
    uuids = [str(uuid4()) for _ in range(len(split_docs))]
    vector_db.add_documents(split_docs, ids=uuids)

    yield "Done adding documents to vector database...✅"

def generate_answer(query):
    """
    Generate an answer to a query using the RAG system.
    
    Args:
        query (str): The query to answer.
        
    Returns:
        tuple: The generated answer and its sources.
    """
    global vector_db, groq_model

    if vector_db is None or groq_model is None:
        raise ValueError("Components not initialized. Call initialize_components() first.")

    print('Generating answer...')
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=groq_model,
        retriever=retriever,
        return_source_documents=True
    )

    res = qa_chain.invoke({"question": query}, return_only_outputs=True)
    answer = res.get("answer", "No answer returned.")
    sources = res.get("sources", "No sources found.")

    return answer, sources


if __name__ == "__main__":
    urls = [
        'https://tradingeconomics.com/india/residential-property-prices',
        'https://www.ceicdata.com/en/indicator/india/house-prices-growth'
    ]

    process_urls(urls)
    question = "Tell me how much indian house prices have grown in 2024?"
    answer, sources = generate_answer(question)

    print(f"\n✅ Answer:\n{answer}")
    print(f"\n📚 Sources:\n{sources}")
