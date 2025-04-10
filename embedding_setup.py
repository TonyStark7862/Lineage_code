from sentence_transformers import SentenceTransformer
print("attribute_matching|Initializing language representation model...")
# Using a different but comparable embedding model
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
print("attribute_matching|Language representation model loaded successfully")
