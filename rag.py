from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=500
        )

    if vector_store is None:

        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):

    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector database..."
    vector_store.reset_collection()

    # BETTER LOADER
    yield "Loading website data..."

    loader = WebBaseLoader(urls)

    data = loader.load()

    print("\n================ RAW DATA ================\n")

    for i, doc in enumerate(data):
        print(f"\nDOCUMENT {i+1}\n")
        print(doc.page_content[:1500])

    # BETTER CHUNKING
    yield "Splitting documents into chunks..."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = text_splitter.split_documents(data)

    print(f"\nTOTAL CHUNKS CREATED: {len(docs)}")

    # DEBUG CHUNKS
    for i, doc in enumerate(docs[:5]):
        print(f"\n========== CHUNK {i+1} ==========\n")
        print(doc.page_content[:1000])

    yield "Adding chunks to vector database..."

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(
        documents=docs,
        ids=uuids
    )

    print("\nVECTOR DB COUNT:", vector_store._collection.count())

    yield "Vector database ready!"



def generate_answer(query):

    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    # IMPORTANT: higher k
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    # DEBUG RETRIEVED DOCS
    retrieved_docs = retriever.invoke(query)

    print("\n=========== RETRIEVED DOCS ===========\n")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nRETRIEVED DOC {i+1}\n")
        print(doc.page_content[:1000])

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    result = chain.invoke(
        {"question": query},
        return_only_outputs=True
    )

    return result["answer"], result.get("sources", "")


if __name__ == "__main__":

    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    print("\n================ QUESTION ================\n")

    question = "What was the 30-year fixed mortgage rate and its date?"

    answer, sources = generate_answer(question)

    print("\n================ ANSWER ================\n")

    print(answer)

    print("\n================ SOURCES ================\n")

    print(sources)