import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "paper.pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a strict assistant.
Answer ONLY from the context below.
Do NOT use outside knowledge.
If answer not found, say exactly:
"Sorry, the PDF doesn't discuss this."

Context:
{context}

Question:
{question}"""
)

_vector_store = None
_retriever = None
_main_chain = None


def get_embeddings():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in environment.")

    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=api_token,
        model=EMBEDDING_MODEL,
        task="feature-extraction",
    )


def build_and_save_vector_store():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

    print(f"Vector store saved to: {VECTOR_DB_PATH}")
    print(f"Total vectors: {vector_store.index.ntotal}")
    return vector_store


def load_or_build_vector_store():
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    embeddings = get_embeddings()
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    store_file = os.path.join(VECTOR_DB_PATH, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(store_file):
        _vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        _vector_store = build_and_save_vector_store()

    return _vector_store


def get_retriever():
    global _retriever

    if _retriever is None:
        vector_store = load_or_build_vector_store()
        _retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

    return _retriever


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def get_main_chain():
    global _main_chain

    if _main_chain is None:
        retriever = get_retriever()
        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        _main_chain = parallel_chain | PROMPT_TEMPLATE | get_model() | StrOutputParser()

    return _main_chain


def ask_pdf(question):
    question = question.strip()
    if not question:
        return "Sorry, the PDF doesn't discuss this."

    docs = get_retriever().invoke(question)
    if not docs:
        return "Sorry, the PDF doesn't discuss this."

    response = get_main_chain().invoke(question).strip()
    return response or "Sorry, the PDF doesn't discuss this."


def explain_prediction(prediction, confidence, top_features, input_values):
    model = get_model()

    feature_lines = "\n".join(
        f"- {item['name']}: value={item['value']}, importance={item['importance']:.4f}"
        for item in top_features
    )
    input_lines = "\n".join(
        f"- {key}: {value}" for key, value in input_values.items()
    )

    prompt = f"""You are helping explain an NSL-KDD intrusion prediction.
Give a short explanation in simple language.
Do not claim exact causality.
Clearly say these are the most likely contributing columns based on the model's feature importance and the user's active inputs.
Mention the prediction label and confidence.
Keep the answer under 120 words.

Prediction: {prediction}
Confidence: {confidence:.2%}

Most likely contributing columns:
{feature_lines if feature_lines else "- No strong active columns found."}

User input values:
{input_lines if input_lines else "- No input values provided."}
"""

    response = model.invoke(prompt)
    return getattr(response, "content", str(response)).strip()
