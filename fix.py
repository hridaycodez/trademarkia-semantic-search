with open('src/vector_store.py', 'r') as f:
    content = f.read()

content = content.replace(
    'client.get_or_create_collection(\n        name=COLLECTION_NAME,\n        metadata={"hnsw:space": "cosine"},\n    )',
    'client.get_or_create_collection(\n        name=COLLECTION_NAME,\n        metadata={"hnsw:space": "cosine"},\n        embedding_function=None,\n    )'
)

with open('src/vector_store.py', 'w') as f:
    f.write(content)

print('Fix applied successfully!')
