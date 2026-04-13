try:
    from ragas.embeddings import embedding_factory
    emb = embedding_factory(model="BAAI/bge-m3")
    print("YES, instantiated embedding_factory")
except Exception as e:
    print("ERROR:", e)
