import os
import glob
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================================================
# CONFIGURATION
# ==========================================================

PDF_FOLDER = "policies"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ==========================================================
# LOAD PDF DOCUMENTS
# ==========================================================

documents = []

pdf_files = glob.glob(
    os.path.join(PDF_FOLDER, "*.pdf")
)

if len(pdf_files) == 0:

    print("No PDF files found!")

    exit()

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_file in pdf_files:

    print("Reading:", os.path.basename(pdf_file))

    try:

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        documents.append({

            "filename": os.path.basename(pdf_file),

            "text": text

        })

    except Exception as e:

        print("Failed:", pdf_file)

print("\nDocuments Loaded:", len(documents))

# ==========================================================
# CREATE CHUNKS
# ==========================================================

chunks = []

metadata = []

for doc in documents:

    text = doc["text"]

    start = 0

    while start < len(text):

        chunk = text[
            start:start + CHUNK_SIZE
        ]

        if len(chunk.strip()) > 50:

            chunks.append(chunk)

            metadata.append({

                "source": doc["filename"]

            })

        start += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )

print("Chunks Created:", len(chunks))

if len(chunks) == 0:

    print("No chunks were created.")

    exit()

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = embedding_model.encode(

    chunks,

    show_progress_bar=True

)

embeddings = np.array(
    embeddings
).astype("float32")

print("\nEmbedding Shape:", embeddings.shape)

# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

print("\nBuilding FAISS Index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors Indexed:", index.ntotal)

# ==========================================================
# SAVE FILES
# ==========================================================

print("\nSaving files...")

faiss.write_index(
    index,
    "faiss_index.index"
)

np.save(
    "chunks.npy",
    np.array(chunks, dtype=object)
)

np.save(
    "metadata.npy",
    np.array(metadata, dtype=object)
)

np.save(
    "embeddings.npy",
    embeddings
)

print("\n===================================")
print("Index created successfully!")
print("===================================")

print("Saved Files:")
print("✓ faiss_index.index")
print("✓ chunks.npy")
print("✓ metadata.npy")
print("✓ embeddings.npy")

print("\nReady for Streamlit 🚀")