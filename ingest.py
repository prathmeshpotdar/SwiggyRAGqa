from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/Annual-Report-FY-2023-24.pdf"
VECTOR_DB_PATH = "vectorstore"


def ingest_data():

    print("Loading Swiggy Annual Report...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("Generating embeddings...")

    embeddings = OpenAIEmbeddings()

    print("Saving vectors into FAISS...")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_data()
