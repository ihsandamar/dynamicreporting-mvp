from click import prompt
from sqlalchemy import create_engine

from constants import OPENAI_API_KEY, OPENAI_API_TYPE, ConnectionString
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain.prompts import PromptTemplate
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_experimental.sql.base import  SQLDatabase
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


# OpenAI GPT-4 LLM yapılandırması
llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Veritabanını tanımlayın
db = SQLDatabase.from_uri(database_uri=ConnectionString)  

text_to_sql_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a SQL SERVER expert. Given an input question, create a syntactically correct PostgreSQL query to run and return ONLY the generated Query and nothing else. Unless otherwise specified, do not return more than \
        {top_k} rows.\n\nHere is the relevant table info: {table_info}\
        Pay close attention on which column is in which table. if context contains more than one tables then create a query by performing JOIN operation only using the column unitid for the tables.\
        Follow these Instructions for creating syntactically correct SQL query:\
        - Be sure not to query for columns that do not exist in the tables and use alias only where required.\
        - Always use the column 'instnm' associated with the 'unitid' in the generated query.\
        - Whenever asked for Institute Names, return the institute names using column 'instnm' associated with the 'unitid' in the generated query.\
        - Likewise, when asked about the average (AVG function) or ratio, ensure the appropriate aggregation function is used.\
        - Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.\
        - If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.\
        - When dealing with date or timestamp columns, use appropriate date functions (e.g., DATE_PART, EXTRACT) for extracting specific parts of the date or performing date arithmetic.\
        - If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.\
        - Consider using aliases for tables and columns to improve readability of the query, especially in case of complex joins or subqueries.\
        - If necessary, use subqueries or common table expressions (CTEs) to break down the problem into smaller, more manageable parts."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)


# Embedding işlemi için OpenAI kullanılıyor
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Veritabanı şemasını JSON dosyasından yükle
loader = JSONLoader(
    file_path='database_schema.json',
    jq_schema='.',
    text_content=False)
data = loader.load()
vectorstore = Chroma.from_documents(data, openai_embeddings)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
    {context}
    Search for the table descriptions in the context and accordingly search for column names and associated column description. Include only relevant tables and columns which can be used by the downstream Text-to-SQL Agent to create SQL Queries for generating answer.
    Search for any information performing the following tasks:
    1. Table Names
    2. Table Descriptions
    3. Column Names
    4. Column Descriptions
    5. Encoded Values
    Finally, only return table names, column names and Encoded Values only (if availabe).

    Question: {question}
    """
retriever_prompt = ChatPromptTemplate.from_template(template)
retriever_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
)
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db)

db_chain.invoke({"query": retriever_chain})
