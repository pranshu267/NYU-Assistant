from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"],
               temperature=1,
               model_name= "llama3-8b-8192")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
FAISS_PATH = "faiss_index"

def get_qa_chain():

    vectordb = FAISS.load_local(FAISS_PATH, EMBEDDINGS)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Answer the question and provide additional helpful information,
        based on the pieces of information, if applicable. Be succinct.

        Responses should be properly formatted to be easily read.
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )

    return chain
