from operator import itemgetter
from click import prompt
from sqlalchemy import Table, create_engine

from constants import OPENAI_API_KEY, OPENAI_API_TYPE, ConnectionString
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_experimental.sql.base import SQLDatabase
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, graph_ascii, graph
from langchain.chains.sql_database.query import create_sql_query_chain

# OpenAI GPT-4 LLM configuration
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define the database
db = SQLDatabase.from_uri(database_uri=ConnectionString)

# Define a text-to-SQL prompt template with instructions
text_to_sql_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a T-SQL expert. Given an input question, create a syntactically correct MSSQL query to run and return ONLY the generated Query and nothing else. Unless otherwise specified, do not return more than \
        {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\n"
         "Instructions for creating syntactically correct SQL query:\n"
         "- Avoid using backticks. Use square brackets for identifiers.\n"
         "- Always use the column 'instnm' associated with the 'unitid'.\n"
         "- For institute names, use 'instnm' with 'unitid'.\n"
         "- Use AVG and other aggregate functions as required.\n"
         "- Apply filtering, GROUP BY with aggregation as needed.\n"
         "- Use readable aliases where necessary.\n"
         "- Return SQL code only in brackets or other delimiters."
         ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

# Embedding using OpenAI
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Load database schema from JSON file
loader = JSONLoader(
    file_path='schooldb.json',
    jq_schema='.',
    text_content=False)
data = loader.load()
vectorstore = Chroma.from_documents(data, openai_embeddings)
retriever = vectorstore.as_retriever()

# Define the retriever prompt
template = """Answer the question based only on the following context:
    {context}
    Search for the table descriptions in the context and search for column names and associated column descriptions. Only include relevant tables and columns to create SQL Queries.
    Tasks:
    1. Table Names
    2. Table Descriptions
    3. Column Names
    4. Column Descriptions
    5. Encoded Values (if available)

    Question: {question}
    """
retriever_prompt = ChatPromptTemplate.from_template(template)
retriever_chain = retriever_prompt.format_prompt(
    context=retriever, 
    question="her eğitmenin verdiği dersleri listele"
)

# Initialize the SQL database chain
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db)

# Run the chain and print results
result = db_chain.invoke({"query": retriever_chain.to_string()})
print(result["result"])
