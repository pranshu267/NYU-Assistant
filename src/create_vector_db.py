from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from data_loader import load_documents, split_text

FAISS_PATH = "faiss_index"
EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)

def save_to_faiss(chunks: list[Document]):

    vectordb = FAISS.from_documents(chunks, EMBEDDINGS)
    vectordb.save_local(FAISS_PATH)

if __name__ == "__main__":
    main()