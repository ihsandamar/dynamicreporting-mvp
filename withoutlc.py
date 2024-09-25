from sqlalchemy import create_engine
from constants import *
from DatabaseSchema import *
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import chroma as Chroma
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_community.llms import openai as OpenAI
from langchain_experimental.sql.base import SQLDatabaseChain




# SQLAlchemy ile bağlantı oluştur
engine = create_engine(ConnectionString)

# DatabaseSchema(engine).create_database_schema()

# GPT-4 için prompt şablonu
template = """
You are a SQL assistant that helps generate SQL queries. Given the context and question, generate the correct SQL query for MSSQL.
Context: {context}
Question: {question}

SQL Query:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Veritabanı şemasını JSON dosyasından yükle
loader = JSONLoader(
    file_path='database_schema.json',
    jq_schema='.',
    text_content=False)

docs = loader.load()

# Embedding işlemi için OpenAI kullanılıyor
embeddings = OpenAIEmbeddings(openai_api_type=OPENAI_API_TYPE, api_key=OPENAI_API_KEY)

# Chroma ile vektör tabanlı bir veri deposu oluşturuyoruz
vectorstore = Chroma.from_documents(docs, embeddings)


# Vektör tabanlı bir retriever oluşturuyoruz
retriever = VectorStoreRetriever(vectorstore=vectorstore)

# OpenAI GPT-4 LLM yapılandırması
llm = OpenAI(model_name=OPENAI_API_VERSION, temperature=0, openai_api_key=OPENAI_API_KEY)

# Veritabanı zincirini oluşturuyoruz
db_chain = SQLDatabaseChain(llm=llm, database=engine, retriever=retriever, prompt=prompt)

# Kullanıcı sorusuna göre SQL sorgusu üret
def text_to_sql(question: str):
    # Önce vektör tabanlı retriever ile bağlamı alalım
    context = retriever.get_relevant_documents(question)
    context_str = "\n".join([doc.page_content for doc in context])
    
    # Soruya göre SQL sorgusunu oluştur
    sql_query = db_chain.invoke({"context": context_str, "question": question})
    return sql_query

# Kullanıcıdan gelen doğal dili SQL'e çeviriyoruz
question = "List all customers who made a purchase last month."
sql_query = text_to_sql(question)

print(f"Generated SQL Query: {sql_query}")

