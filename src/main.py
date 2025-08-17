
from itertools import chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def get_api_key(api_key="API_KEY"):
    """
    Pega a chave da api no dotenv.
    """
    load_dotenv()

    api_key = os.getenv(api_key)
    if api_key is None:
        raise ValueError("A chave da API não foi definida no .env")
    
    return api_key

# 1. Leitura
def load_pdfs(folder_path):
    """
    Captura retorna uma variável com o texto de todos os PDFs presentes no diretório específicado
    """
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if "pdf" in file_name:
            loader = PyPDFLoader(file_path) 
        elif "txt" in file_name:
            loader = TextLoader(file_path)
        else:
            pass

        documents.extend(loader.load()) # Une os textos de todos os documentos

    return documents

# 2. Splitting
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Divide o texto do documento em diversos 'chunks' de tamanhos definidos pelo 'chunk_size'.
    """
    # Cria uma classe RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Aplica o split_documents (nativo na classe text_splitter) e retorna uma array com o texto dividido em pedaços (com overlap para manter o contexto)
    return text_splitter.split_documents(documents)

# 3. Chroma DB e Embedding
def create_chroma_db(chunks, db_path="chroma_db"):
    """
    Utilizando um modelo de embedding disponivel pela HuggingFace e armazenando os dados com embedding no Chroma DB
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)
    vector_db.persist()

    return vector_db

# 4. Retornar elementos relevantes por similiaridade
def retrieve_documents(query, vector_db, top_k=3):
    return vector_db.similarity_search(query, k=top_k)

# 5. Retornar resposta
def build_chain():
    modelo = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",     
        #model_name="openai/gpt-oss-20b:free"
        model_name="mistralai/mistral-small-3.2-24b-instruct:free"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é um assistente pessoal. Use o contexto fornecido para responder."),
            ("placeholder", "{historico}"),
            ("human", "Contexto:\n{context}\n\nPergunta: {query}")
        ]
    )

    cadeia = prompt | modelo | StrOutputParser()
    return cadeia

# 6. Armazenar histórico de conversa
memory = {}
def session_history(session: str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
    return memory[session]

def build_chain_with_history():
    chain = build_chain()
    memory_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=session_history,
        input_messages_key="query",
        history_messages_key="historico"
    )
    return memory_chain

if __name__ == "__main__":
    # 1. Pegar chave da API no dotenv
    API_KEY = get_api_key()

    # 2. Pegar texto dos documentos
    documents = load_pdfs('./docs/')

    # 3. Documentos divididos
    splitted_documents = split_documents(documents)

    # 4. Enviando para o chorma DB
    vector_db = create_chroma_db(splitted_documents)

    # 5. Criando sessao para armazenar histórico
    session = "sessao_1"
    memory_chain = build_chain_with_history()
    
    while True:
        query = input("\nPergunta (ou exit para sair): \n: ")

        if query.lower() == "exit":
            break

        retrieved_docs = retrieve_documents(query, vector_db)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        answer = memory_chain.invoke({"query": query, "context": context}, config={"session_id": session})
        print("\nAI Response:\n", answer)