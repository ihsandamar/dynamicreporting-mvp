import chromadb
chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.add(
    documents=["This is a document containing car information",
    "This is a document containing information about dogs", 
    "This document contains four wheeler catalogue"],
    metadatas=[{"source": "Car Book"},{"source": "Dog Book"},{'source':'Vechile Info'}],
    ids=["id1", "id2", "id3"]
)

results = collection.query(
    query_texts=["Car"],
    n_results=2
)


print(results)
