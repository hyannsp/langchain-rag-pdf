from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import ChatOpenAI #, OpenAIEmbeddings pago
from langchain_community.embeddings import HuggingFaceEmbeddings # Gratis
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("A chave da API não foi definida no .env")

modelo = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",     
    model_name="meta-llama/llama-3-8b-instruct"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Caminho dos arquivos
arquivos = [
    "./docs/GTB_standard_Nov23.pdf",
    "./docs/GTB_gold_Nov23.pdf",
    "./docs/GTB_platinum_Nov23.pdf"
]

# Concatenando todos os documentos
documentos = sum(
    [PyPDFLoader(arquivo).load() for arquivo in arquivos], []
)

# Quebrando o texto em pedaços (split)
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Split a cada 1000 char
    chunk_overlap=100 # Sobreposição da chunksize para não tirar do contexto
).split_documents(documentos)

# Utilizando o FAISS, uma base de dados vetorial simples e de uso gratuito
dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":2}) # Recuperar embedding com posição de semelhança como top 2

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta: ")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))