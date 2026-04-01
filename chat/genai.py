import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "paper.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store_db")


def build_and_save_vector_store():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

    print(f"Vector store saved to: {VECTOR_DB_PATH}")
    print(f"Total vectors: {vector_store.index.ntotal}")

    docs = vector_store.similarity_search("test", k=1)
    if docs:
        print(docs[0].page_content)

    #print(vector_store.index.reconstruct(0))
    return vector_store


vector_store = build_and_save_vector_store()

# retriever

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever)
#response=retriever.invoke('What is NSL-KDD dataset?')
#print(response)


# Augmentation

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    provider="auto",
    temperature=0,
    max_new_tokens=256
)

def ask_pdf(question):
    docs = retriever.invoke(question)

    if not docs:
        return "Sorry, the PDF doesn't discuss this."

    context = "\n\n".join(doc.page_content for doc in docs[:3])

    prompt = f"""
You are a strict assistant.
Answer ONLY from the context below.
Do NOT use outside knowledge.
If answer not found, say exactly:
"Sorry, the PDF doesn't discuss this."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break
    print(ask_pdf(q))