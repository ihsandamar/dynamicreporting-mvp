from langchain_community.utilities import SQLDatabase
import os

import langchain_openai
from langchain.chains import create_sql_query_chain

# from langchain_openai import ChatOpenAI

import constants



# Çevre değişkenlerini ayarlayın (gerekirse)
os.environ["LANGCHAIN_TRACING_V2"] = constants.LANGCHAIN_TRACING_V2 # Veya uygun bir değer
os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY


# SQLDatabase nesnesini oluşturun
database = SQLDatabase.from_uri(database_uri=constants.ConnectionString)

model = langchain_openai.ChatOpenAI(model=constants.OPENAI_API_VERSION)

tables = database.run("""
SELECT name  
FROM sys.objects  
WHERE type = 'U'; 
""")

for table in tables.split(','):
    print(table)


# chain = create_sql_query_chain(model, database)
# response = chain.invoke({"question": "kaç tablo var ve hangi tabloda siparişler bulunuyor"})
# print(response)
# print(database.run(response).__str__())


