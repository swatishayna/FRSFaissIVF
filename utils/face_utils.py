import os
import time
import logging
import numpy as np

from utils.db_utils import get_connection, fetch_images_from_db
from utils.faiss_utils import (
    create_idmapped_faiss_index,
    load_faiss_index,
    save_faiss_index,
    add_embeddings_with_ids,
)
from utils.embedding_utils import EmbeddingStore, generate_face_embedding
from utils.extra_utils import append_metadata, save_image_locally
from insightface.app import FaceAnalysis

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_DIR = os.path.join(BASE_DIR, "faiss")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "metadata.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

EMBED_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
ID_PATH = os.path.join(EMBED_DIR, "image_ids.npy")
FAISS_PATH = os.path.join(FAISS_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(FAISS_DIR, "id_map.json")

# -----------------------------
# Initialize model and stores
# -----------------------------
logger.info("🧠 Loading face embedding model ...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("✅ Face model loaded successfully.")

embedding_store = EmbeddingStore(EMBED_PATH, ID_PATH)

# -----------------------------
# Ensure FAISS Index
# -----------------------------
def ensure_faiss_index():
    """Ensure FAISS index exists, load or create a new one."""
    if not os.path.exists(FAISS_PATH):
        logger.warning("⚠️ No FAISS index found. Creating new ID-mapped FAISS index.")
        index = create_idmapped_faiss_index(dimension=512)
        save_faiss_index(index, FAISS_PATH)
        return index

    return load_faiss_index(FAISS_PATH)

# -----------------------------
# Process New Images
# -----------------------------
def process_new_images(batch_size=5000, last_id=0):
    """Fetch new images, generate embeddings, and update FAISS & metadata."""
    faiss_index = ensure_faiss_index()
    conn = get_connection()

    results = fetch_images_from_db(conn, batch_size=batch_size, last_id=last_id)
    logger.info(f"🔍 Retrieved {len(results)} records from DB.")

    new_embeddings, new_ids = [], []

    for row in results:
        file_srno = int(row["FILE_SRNO"])

        if embedding_store.exists(file_srno):
            logger.debug(f"⏭️ Skipping existing FILE_SRNO={file_srno}")
            continue

        img_blob = row.get("UPLOADED_FILE")
        if not img_blob:
            logger.warning(f"⚠️ No image blob found for FILE_SRNO={file_srno}")
            continue

        # Save locally
        image_path = save_image_locally(img_blob, file_srno, IMAGE_DIR)

        # Generate embedding
        embedding = generate_face_embedding(app, img_blob)
        if embedding is None:
            logger.warning(f"⚠️ No face detected for FILE_SRNO={file_srno}")
            continue

        # Store embedding
        embedding_store.add(np.array([embedding]), [file_srno])
        new_embeddings.append(embedding)
        new_ids.append(file_srno)

        metadata_row = {
            "FILE_SRNO": file_srno,
            "ACCUSED_SRNO": row.get("ACCUSED_SRNO"),
            "FILE_NAME": row.get("FILE_NAME"),
            "ACCUSED_NAME": row.get("ACCUSED_NAME"),
            "RELATIVE_NAME": row.get("RELATIVE_NAME"),
            "AGE": row.get("AGE"),
            "GENDER": row.get("GENDER"),
            "FIR_REG_NUM": row.get("FIR_REG_NUM"),
            "IMAGE_PATH": image_path,
        }
        append_metadata(metadata_row, METADATA_PATH)

    conn.close()
    logger.info("🔌 Database connection closed.")

    # -----------------------------
    # Update FAISS index
    # -----------------------------
    if new_embeddings:
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
        add_embeddings_with_ids(faiss_index, new_embeddings, new_ids, id_map_path=ID_MAP_PATH)
        save_faiss_index(faiss_index, FAISS_PATH)
        logger.info(f"✅ Added {len(new_embeddings)} new embeddings to FAISS index.")
    else:
        logger.info("ℹ️ No new embeddings generated.")

    logger.info("🎯 Processing complete.")

# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    start = time.time()
    process_new_images(batch_size=5000)
    logger.info(f"⏱️ Total time: {round(time.time() - start, 2)} sec")
