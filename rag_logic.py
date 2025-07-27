from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_core.prompts.prompt import PromptTemplate



load_dotenv()


# Setup embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Global Chroma DB variable
db = None

# Create a temp pdf file to apply embedding

def load_vector_db(file_path) -> None:
    global db

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    db.persist()


def query_vector_db(query: str, k=3) -> list:
    global db
    results = db.similarity_search(query, k=3)
    return [
        {"text": doc.page_content, "page": doc.metadata.get("page", -1)}
        for doc in results
    ]


def format_context(docs: list[dict]) -> str:
    return "\n".join(f"{doc['text']} [page: {doc['page']}]" for doc in docs)


def run_rag(query: str) -> list[str, list[int]]:
    retrieved_docs = query_vector_db(query)
    context = format_context(retrieved_docs)
    prompt_template = PromptTemplate.from_template("""
    You are a researcher analyzing expert from documents. Your task is to answer the question using only the provided information in the "context" section below. Respond in clear and simple Arabic suitable for a general audience.

    Instructions:
- إذا ما لقيت إجابة كافية في السياق، قل بصراحة "المعلومة غير متوفرة في الوثيقة"    

    Context: {context}
    Question: {query}
    Answer:
    """)
    prompt = prompt_template.format(context=context, query=query)
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5)
    response = llm.invoke(prompt)

    # Get unique pages and sort them
    pages = [int(doc['page']) + 1 for doc in retrieved_docs if doc['page'] is not None]
    unique_pages = sorted(list(set(pages)))

    return response, unique_pages

