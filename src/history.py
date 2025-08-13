# Imports e API
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("A chave da API não foi definida no .env")

modelo = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",     
    model_name="openai/gpt-oss-20b:free"
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        # Contexto base
        ("system", "Você é um IA que vai me ajudar com o TCC de ciências da computação."),
        # Contexto conversacional
        ("placeholder", "{historico}"),
        # Pergunta/comando
        ("human", "{query}")
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "sessao_1"

# Garantir a memsa memória para cada sessão
def historico_por_sessao(sessao:str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]
"""
lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]
"""

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)
loop = True
while loop:
    pergunta = input("Prompt: \n")
    if pergunta == "0":
        loop = False
        pass
    resposta = cadeia_com_memoria.invoke(
        {"query": pergunta},
        config={"session_id": sessao}
    )
    print("IA: ", resposta, "\n")