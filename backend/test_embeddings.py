from sentence_transformers import SentenceTransformer

print("Loading local embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")

text = "Artificial intelligence is transforming scientific research."

embedding = model.encode(text)

print("Embedding created successfully.")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])