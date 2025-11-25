from fastapi import FastAPI, UploadFile, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np
import uvicorn
import re
import logging

# ---- Your utils ----
from utils.config_utils import load_config
from utils.embedding_utils import EmbeddingStore
from utils.faiss_utils import (
    load_faiss_index,
    search_faiss_index,
    save_faiss_index,
    add_to_faiss_index,
    get_faiss_id_map
)
from utils.db_utils import get_image_row_by_filesrno
from utils.extra_utils import save_image_locally, repair_and_log_corrupted_embeddings
from utils.logger_utils import setup_logger


from insightface.app import FaceAnalysis





# ---- Load config ----
config = load_config()

setup_logger(config)
logger = logging.getLogger(__name__)


# faiss_index_path = os.path.join("data", "faiss","faiss.index")
faiss_index_path = os.path.join("data", "faiss", "faiss_ivf.index")
metadata_csv = os.path.join("data", "metadata","metadata.csv")
threshold = float(config.get("threshold", 0.7))
faiss_cfg = config.get("faiss", {"dimension": 512})
image_folder = os.path.join("data", "images")
os.makedirs(image_folder, exist_ok=True)

# ---- App initialization ----
app = FastAPI(title="Face Recognition (FAISS ANN)")

# ---- Embedding store & paths ----
embedding_path = os.path.join("data","embeddings", "embeddings.npy")
id_path = os.path.join("data", "embeddings","image_ids.npy")
embedding_store = EmbeddingStore(embedding_path, id_path)

def get_metadata(image_id: str, metadata_csv: str):
    import pandas as pd

    if not os.path.exists(metadata_csv):
        return {}

    try:
        # Read raw lines (we will fix each row manually)
        with open(metadata_csv, "r", encoding="utf-8") as f:
            rows = [line.strip() for line in f.readlines()]

        fixed_rows = []
        for line in rows:
            parts = line.split(",")

            # Expected = 10 columns
            if len(parts) == 10:
                fixed_rows.append(parts)
                continue

           
            if len(parts) > 10:
                # Fix logic:
                # FILE_NAME = parts[2]
                # ACCUSED_NAME = join parts[3:-7]
                # RELATIVE_NAME = parts[-7]
                # AGE = parts[-6]
                # GENDER = parts[-5]
                # FIR_REG_NUM = parts[-4]
                # IMAGE_PATH = parts[-3]
                # TIMESTAMP = parts[-2] or [-1]

                fixed = [
                    parts[0],                     # FILE_SRNO
                    parts[1],                     # ACCUSED_SRNO
                    parts[2],                     # FILE_NAME
                    ",".join(parts[3:-7]).strip(),# ACCUSED_NAME (merged)
                    parts[-7],                    # RELATIVE_NAME
                    parts[-6],                    # AGE
                    parts[-5],                    # GENDER
                    parts[-4],                    # FIR_REG_NUM
                    parts[-3],                    # IMAGE_PATH
                    parts[-2] if len(parts) > 10 else parts[-1]  # TIMESTAMP
                ]
                fixed_rows.append(fixed)
                continue

            # If fewer columns? pad with empty strings
            while len(parts) < 10:
                parts.append("")
            fixed_rows.append(parts)

        # Convert to DataFrame
        df = pd.DataFrame(fixed_rows, columns=[
            "FILE_SRNO",
            "ACCUSED_SRNO",
            "FILE_NAME",
            "ACCUSED_NAME",
            "RELATIVE_NAME",
            "AGE",
            "GENDER",
            "FIR_REG_NUM",
            "IMAGE_PATH",
            "TIMESTAMP"
        ])

        # Normalize IDs
        df["FILE_SRNO"] = df["FILE_SRNO"].astype(str).str.strip().str.replace(".0", "")

        clean_id = str(image_id).strip().replace(".0", "")

        row = df[df["FILE_SRNO"] == clean_id]

        if row.empty:
            print("❌ No match found for:", clean_id)
            return {}

        return row.iloc[0].to_dict()

    except Exception as e:
        print("Metadata lookup exception:", e)
        return {}

def extract_multiple_faces(img_bytes: bytes):
    """
    Returns list of dicts:
    [{ 'face_id': i, 'bbox': [x1,y1,x2,y2], 'embedding': np.array }, ...]
    """
    import cv2
    import numpy as np

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    results = []

    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)

        results.append({
            "face_id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "embedding": f.normed_embedding
        })

    return results




# ---- Startup embedding integrity check ----
@app.on_event("startup")
async def check_and_repair_embeddings():
    """
    Run a one-time embedding integrity check and repair before serving requests.
    """
    failed_metadata_path = os.path.join("logs", "metadata_failed_embeddings.csv")
    os.makedirs(os.path.dirname(failed_metadata_path), exist_ok=True)

    try:
        repaired = repair_and_log_corrupted_embeddings(
            embedding_path=embedding_path,
            id_path=id_path,
            metadata_path=metadata_csv,
            failed_metadata_path=failed_metadata_path,
            embedding_dim=faiss_cfg.get("dim", 512),
        )
        if repaired > 0:
            print(f"🧹 Repaired {repaired} corrupted embeddings before server startup.")
        else:
            print("✅ Embeddings verified clean at startup.")
    except Exception as e:
        print(f"⚠️ Warning: failed to verify embeddings — {e}")

# ---- Load face model (replaces FaceProcessor) ----
print("🧠 Loading face model ...")
face_app = FaceAnalysis(name=config.get("face_model", "buffalo_l"))
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Face model loaded successfully.")


def get_embedding_from_bytes(img_bytes: bytes) -> Optional[np.ndarray]:
    """
    Generate a 512-d face embedding from image bytes.
    Returns None if no face detected.
    """
    import cv2
    import numpy as np

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return None

    return faces[0].normed_embedding


# ---- FAISS Index Management ----
# def get_or_build_faiss_index():
#     """Load existing FAISS index or build a new one from embeddings."""
#     if os.path.exists(faiss_index_path):
#         return load_faiss_index(
#             faiss_index_path, faiss_config={"dimension": faiss_cfg.get("dim", 512)}
#         )

#     if embedding_store.embeddings is None or embedding_store.embeddings.shape[0] == 0:
#         return None  # No embeddings yet

#     index = load_faiss_index(
#         faiss_index_path, faiss_config={"dimension": faiss_cfg.get("dim", 512)}
#     )
#     emb = np.array(embedding_store.embeddings, dtype=np.float32)
#     add_to_faiss_index(index, emb)
#     save_faiss_index(index, faiss_index_path)
#     return index

def get_or_build_faiss_index():
    """Load or create FAISS IVF index, and train if needed."""
    from utils.faiss_utils import load_faiss_index, add_to_faiss_index, save_faiss_index
    import numpy as np

    if os.path.exists(faiss_index_path):
        return load_faiss_index(faiss_index_path, dimension=faiss_cfg.get("dim", 512), nlist=1000)

    if embedding_store.embeddings is None or embedding_store.embeddings.shape[0] == 0:
        return None

    index = load_faiss_index(faiss_index_path, dimension=faiss_cfg.get("dim", 512), nlist=1000)
    embeddings = np.array(embedding_store.embeddings, dtype=np.float32)

    if not index.is_trained:
        print("🧠 Training IVF index before adding embeddings...")
        index.train(embeddings)
        print("✅ IVF training complete.")

    add_to_faiss_index(index, embeddings)
    save_faiss_index(index, faiss_index_path)
    return index

faiss_index = get_or_build_faiss_index()

# ---- Response Models ----
class MatchItem(BaseModel):
    # image_id: str
    # image_path: str
    # distance: float
    # score: Optional[float] = None
    # below_threshold: bool
    
    FILE_SRNO: Optional[int] = None
    ACCUSED_SRNO: Optional[int] = None
    FILE_NAME: Optional[str] = None
    ACCUSED_NAME: Optional[str] = None
    RELATIVE_NAME: Optional[str] = None
    AGE: Optional[int] = None
    GENDER: Optional[str] = None
    FIR_REG_NUM: Optional[str] = None
    IMAGE_PATH: Optional[str] = None
    IMAGE_BASE64: Optional[str] = None

    image_id: str
    image_path: str
    distance: float
    score: Optional[float] = None
    below_threshold: bool

class MultiMatchItem(BaseModel):
    # image_id: str
    # image_path: str
    # distance: float
    # score: Optional[float] = None
    # below_threshold: bool

    face_id: Optional[int] = None
    FILE_SRNO: Optional[int] = None
    ACCUSED_SRNO: Optional[int] = None
    FILE_NAME: Optional[str] = None
    ACCUSED_NAME: Optional[str] = None
    RELATIVE_NAME: Optional[str] = None
    AGE: Optional[int] = None
    GENDER: Optional[str] = None
    FIR_REG_NUM: Optional[str] = None
    IMAGE_PATH: Optional[str] = None
    IMAGE_BASE64: Optional[str] = None

    image_id: str
    image_path: str
    distance: float
    score: Optional[float] = None
    below_threshold: bool



class RecognitionResponse(BaseModel):
    query_image: Optional[str]
    matches: List[MatchItem]

class FaceSearchResult(BaseModel):
    face_id: int
    bbox: List[int]
    matches: List[MultiMatchItem]


class MultiFaceRecognitionResponse(BaseModel):
    total_faces: int
    results: List[FaceSearchResult]

# ---- Helper Functions ----
def id_to_image_path(image_id: str) -> str:
    """Resolve image_id to actual image path."""
    for ext in [".jpg", ".png"]:
        candidate = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_folder, f"{image_id}.jpg")  # fallback


def normalize_distance_to_score(dist: float) -> float:
    """Convert L2 distance to similarity score [0, 1]."""
    max_dist = 4.0
    s = 1.0 - min(dist, max_dist) / max_dist
    return float(max(0.0, min(1.0, s)))


# ---- Routes ----
@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(file: UploadFile, top_k: int = Query(5, ge=1, le=50)):
    """
    Upload a face image and get top_k nearest matches from existing data/images.
    """
    global faiss_index

    contents = await file.read()
    print(f"📸 Received file: {file.filename}, size={len(contents)} bytes")
    embedding = get_embedding_from_bytes(contents)
    if embedding is None:
        print("🚫 No face detected in the image.")
        return RecognitionResponse(query_image=None, matches=[])

    emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
    print(f"✅ Embedding shape: {emb.shape}")

    if faiss_index is None:
        print("⚠️ FAISS index not loaded. Rebuilding ...")
        faiss_index = get_or_build_faiss_index()

    if faiss_index is None or embedding_store.image_ids.size == 0:
        print("🚫 No embeddings or IDs found in the store.")
        return RecognitionResponse(query_image=None, matches=[])

    distances, indices = search_faiss_index(faiss_index, emb, top_k)
    print(f"🔍 Search returned distances: {distances}, indices: {indices}")
    distances, indices = distances[0].tolist(), indices[0].tolist()

    matches = []
    # for idx, dist in zip(indices, distances):
    #     if idx < 0 or idx >= embedding_store.image_ids.shape[0]:
    #         continue
    #     image_id = str(embedding_store.image_ids[idx])
    #     path = id_to_image_path(image_id)
    id_map = get_faiss_id_map("data/faiss/id_map.json")

    for idx, dist in zip(indices, distances):
        image_id = id_map.get(str(idx))  
        if not image_id:
            continue  

        path = id_to_image_path(image_id)
        score = normalize_distance_to_score(float(dist))
        below_threshold = score >= (1.0 - threshold)
        # matches.append(
        #     MatchItem(
        #         image_id=image_id,
        #         image_path=path,
        #         distance=float(dist),
        #         score=score,
        #         below_threshold=below_threshold,
        #     )
        # )

        meta = get_image_row_by_filesrno(file_srno = int(image_id), image_base_path = "temp_images")

        # meta = get_metadata(image_id, metadata_csv)
      

        matches.append(
            MatchItem(
                FILE_SRNO=meta.get("FILE_SRNO"),
                ACCUSED_SRNO=meta.get("ACCUSED_SRNO"),
                FILE_NAME=meta.get("FILE_NAME"),
                ACCUSED_NAME=meta.get("ACCUSED_NAME"),
                RELATIVE_NAME=meta.get("RELATIVE_NAME"),
                AGE=meta.get("AGE"),
                GENDER=meta.get("GENDER"),
                FIR_REG_NUM=meta.get("FIR_REG_NUM"),
                IMAGE_PATH=meta.get("image_path"),
                IMAGE_BASE64= meta.get("image_base64"),

                image_id=image_id,
                image_path=path,
                distance=float(dist),
                score=score,
                below_threshold=below_threshold,
            )
        )


    return RecognitionResponse(query_image=None, matches=matches)

@app.post("/multi-face-search", response_model=MultiFaceRecognitionResponse)
async def multi_face_search(file: UploadFile, top_k: int = Query(5, ge=1, le=50)):
    global faiss_index

    contents = await file.read()

    # Extract all faces
    faces = extract_multiple_faces(contents)
    if not faces:
        logger.info("No face detected")
        return MultiFaceRecognitionResponse(total_faces=0, results=[])

    if faiss_index is None:
        faiss_index = get_or_build_faiss_index()

    id_map = get_faiss_id_map("data/faiss/id_map.json")

    all_results = []

    for face in faces:
        emb = np.array(face["embedding"], dtype=np.float32).reshape(1, -1)

        # Search
        distances, indices = search_faiss_index(faiss_index, emb, top_k)
        distances, indices = distances[0].tolist(), indices[0].tolist()

        matches = []

        # -----------------------------------------
        # 1️⃣ FIRST PASS — apply threshold normally
        # -----------------------------------------
        for idx, dist in zip(indices, distances):

            image_id = id_map.get(str(idx))
            if not image_id:
                logger.warning(f"⚠️ Missing id_map entry for FAISS index {idx}")
                continue  # skip safely

            meta = get_image_row_by_filesrno(
                file_srno=int(image_id),
                image_base_path="temp_images"
            )

            # If DB returns None → skip safely
            if not meta:
                logger.warning(f"⚠️ Metadata missing for image_id {image_id}")
                continue

            path = id_to_image_path(image_id)

            score = normalize_distance_to_score(dist)
            below_threshold = score >= (1.0 - threshold)

            # Only add if above threshold
            if below_threshold:
                matches.append(
                    MultiMatchItem(
                        FILE_SRNO=meta.get("FILE_SRNO"),
                        ACCUSED_SRNO=meta.get("ACCUSED_SRNO"),
                        FILE_NAME=meta.get("FILE_NAME"),
                        ACCUSED_NAME=meta.get("ACCUSED_NAME"),
                        RELATIVE_NAME=meta.get("RELATIVE_NAME"),
                        AGE=meta.get("AGE"),
                        GENDER=meta.get("GENDER"),
                        FIR_REG_NUM=meta.get("FIR_REG_NUM"),
                        IMAGE_PATH=meta.get("image_path"),
                        IMAGE_BASE64=meta.get("image_base64"),

                        image_id=image_id,
                        image_path=path,
                        distance=float(dist),
                        score=score,
                        below_threshold=below_threshold
                    )
                )

        # -----------------------------------------
        # 2️⃣ SECOND PASS — If matches < top_k → fill by LOWEST threshold
        # -----------------------------------------
        if len(matches) < top_k:
            need = top_k - len(matches)

            backup_results = []
            for idx, dist in zip(indices, distances):
                if len(backup_results) >= need:
                    break

                image_id = id_map.get(str(idx))
                if not image_id:
                    continue

                meta = get_image_row_by_filesrno(
                    file_srno=int(image_id),
                    image_base_path="temp_images"
                )
                if not meta:
                    continue

                path = id_to_image_path(image_id)
                score = normalize_distance_to_score(dist)

                backup_results.append(
                    MultiMatchItem(
                        FILE_SRNO=meta.get("FILE_SRNO"),
                        ACCUSED_SRNO=meta.get("ACCUSED_SRNO"),
                        FILE_NAME=meta.get("FILE_NAME"),
                        ACCUSED_NAME=meta.get("ACCUSED_NAME"),
                        RELATIVE_NAME=meta.get("RELATIVE_NAME"),
                        AGE=meta.get("AGE"),
                        GENDER=meta.get("GENDER"),
                        FIR_REG_NUM=meta.get("FIR_REG_NUM"),
                        IMAGE_PATH=meta.get("image_path"),
                        IMAGE_BASE64=meta.get("image_base64"),

                        image_id=image_id,
                        image_path=path,
                        distance=float(dist),
                        score=score,
                        below_threshold=False  # low-confidence
                    )
                )

            matches.extend(backup_results)

        # -----------------------------------------
        # LABEL EACH FACE AS Person1, Person2, etc.
        # -----------------------------------------
        person_label = f"Person{face['face_id']}"

        enriched_results = []
        for m in matches:
            m.face_id = person_label
            enriched_results.append(m)

        all_results.append(
            FaceSearchResult(
                face_id=face["face_id"],
                bbox=face["bbox"],
                matches=enriched_results
            )
        )

    return MultiFaceRecognitionResponse(
        total_faces=len(faces),
        results=all_results
    )



@app.post("/add_image")
async def add_image(file: UploadFile, image_id: str):
    """
    Add an image (upload) to the embedding store and FAISS index.
    """
    global faiss_index
    contents = await file.read()

    if embedding_store.exists(image_id):
        return {"status": "fail", "reason": "Image already exists"}

    embedding = get_embedding_from_bytes(contents)
    if embedding is None:
        return {"status": "fail", "reason": "No face detected"}

    emb_arr = np.array([embedding], dtype=np.float32)

    # Save image
    save_image_locally(contents, image_id, output_dir=image_folder)

    # Add to store and FAISS
    embedding_store.add(emb_arr, [image_id])

    if faiss_index is None:
        faiss_index = get_or_build_faiss_index()
    if faiss_index is None:
        faiss_index = load_faiss_index(
            faiss_index_path, faiss_config={"dimension": faiss_cfg.get("dim", 512)}
        )

    add_to_faiss_index(faiss_index, emb_arr)
    save_faiss_index(faiss_index, faiss_index_path)
    return {"status": "success", "image_id": image_id}


@app.get("/info")
def info():
    """Return quick info about loaded embeddings / index."""
    n_emb = (
        int(embedding_store.embeddings.shape[0])
        if embedding_store.embeddings is not None
        else 0
    )
    n_ids = (
        int(embedding_store.image_ids.shape[0])
        if embedding_store.image_ids is not None
        else 0
    )
    has_index = os.path.exists(faiss_index_path)
    return {
        "num_embeddings": n_emb,
        "num_image_ids": n_ids,
        "faiss_index_exists": has_index,
        "faiss_index_path": faiss_index_path,
    }


# ---- Run server ----
if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    uvicorn.run("app:app", host=host, port=port, reload=True)
