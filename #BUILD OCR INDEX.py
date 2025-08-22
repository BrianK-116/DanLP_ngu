import json
import torch
import clip
import numpy as np
from tqdm import tqdm # For a helpful progress bar

# --- Configuration ---
OCR_TEXT_FILE = "F:\AIC25\code\AICute1-main\AICute1-main\id_to_ocr_text.json"
OUTPUT_INDEX_FILE = "F:\AIC25\code\AICute1-main\AICute1-main/faiss_ocr_ViT.bin"
CLIP_MODEL = "ViT-B/32"

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model
print(f"Loading CLIP model: {CLIP_MODEL}")
model, _ = clip.load(CLIP_MODEL, device=device)

# Load your OCR text data
print(f"Loading OCR text from: {OCR_TEXT_FILE}")
with open(OCR_TEXT_FILE, 'r', encoding='utf-8') as f:
    id_to_text = json.load(f)

# Sort by ID to make sure the order of vectors matches the order of IDs
sorted_items = sorted(id_to_text.items(), key=lambda item: int(item[0]))
ocr_texts = [item[1] for item in sorted_items] # Get just the text strings in order
print("Converting text to CLIP vector embeddings...")
all_text_features = []
batch_size = 256 # Process in batches to manage memory

with torch.no_grad(): # Important for performance
    for i in tqdm(range(0, len(ocr_texts), batch_size)):
        batch_texts = ocr_texts[i:i+batch_size]

        # Tokenize the text and move to the active device (GPU or CPU)
        text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)

        # Use the model to encode the text into vectors
        text_features = model.encode_text(text_tokens) #dung voi faiss la model.encode_image
        
        # Move the results to the CPU and store them
        all_text_features.append(text_features.cpu())

# Combine all the batches into one big NumPy array
text_embeddings_np = torch.cat(all_text_features).numpy().astype('float32')

print(f"Successfully created {text_embeddings_np.shape[0]} vectors of dimension {text_embeddings_np.shape[1]}")
import faiss

print("Building the OCR index...")

# Get the dimension of the vectors (e.g., 512 for ViT-B/32)
d = text_embeddings_np.shape[1]

# Create a FAISS index. IndexFlatL2 is a simple but effective one.
index = faiss.IndexFlatL2(d)

# **Crucial Step**: Normalize the vectors. This makes them suitable for cosine similarity searches.
faiss.normalize_L2(text_embeddings_np)

# Add the array of vectors to the FAISS index
index.add(text_embeddings_np)

print(f"The FAISS index now contains {index.ntotal} vectors.")
print(f"Saving the index to disk at: {OUTPUT_INDEX_FILE}")
faiss.write_index(index, OUTPUT_INDEX_FILE)

print("--- Process Complete! ---")
print(f"Your searchable OCR index is ready at '{OUTPUT_INDEX_FILE}'")