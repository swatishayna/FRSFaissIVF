from fastapi import FastAPI, UploadFile, Query
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
import numpy as np
import uvicorn
import re
import logging
import cv2
from insightface.app import FaceAnalysis

from utils.config_utils import load_config
from utils.embedding_utils import EmbeddingStore
from utils.faiss_utils import (
    load_faiss_index,
    search_faiss_index,
    save_faiss_index,
    add_to_faiss_index,
    get_faiss_id_map
)
from utils.db_utils import get_image_row_by_filesrno_check, get_image_row_by_filesrno
from utils.extra_utils import save_image_locally, repair_and_log_corrupted_embeddings
from utils.logger_utils import setup_logger


# ---- Load config ----
config = load_config()

setup_logger(config)
logger = logging.getLogger(__name__)

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


def extract_multiple_faces_bbox(img_bytes: bytes):
    """
    Detects multiple faces, draws bounding boxes + face_id,
    saves image as bbox.jpg, and returns:
    - results list
    - base64 annotated image
    """

    # -----------------------------
    # 1️⃣ Delete existing bbox.jpg
    # -----------------------------
    output_path = "bbox.jpg"
    if os.path.exists(output_path):
        os.remove(output_path)

    # -----------------------------
    # 2️⃣ Read image from bytes
    # -----------------------------
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    results = []

    # If no faces, return immediately
    if not faces:
        return [], None

    # -----------------------------
    # 3️⃣ Draw bounding boxes + face_id
    # -----------------------------
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)

        # Append result structure
        results.append({
            "face_id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "embedding": f.normed_embedding
        })

        # Draw bounding box (green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put face_id text at top-right
        text = f"ID: {i+1}"

        # Ensure text is inside image
        text_x = max(x2 - 70, 0)
        text_y = max(y1 + 20, 20)

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # 4️⃣ Save annotated image
    # -----------------------------
    cv2.imwrite(output_path, img)

    # -----------------------------
    # 5️⃣ Convert annotated image → base64
    # -----------------------------
    _, buffer = cv2.imencode('.jpg', img)
    bbox_base64 = base64.b64encode(buffer).decode("utf-8")

    return results, bbox_base64


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
class Base64Response(BaseModel):
    filename: Optional[str] = None
    base64_string: Optional[str] = None

class MatchItem(BaseModel):
    FILE_SRNO: Optional[int] = None
    ACCUSED_SRNO: Optional[int] = None
    FILE_NAME: Optional[str] = None
    ACCUSED_NAME: Optional[str] = None
    RELATIVE_NAME: Optional[str] = None
    AGE: Optional[int] = None
    GENDER: Optional[str] = None
    FIR_REG_NUM: Optional[str] = None
    IMAGE_PATH: Optional[str] = None
    DISTRICT: Optional[str] = None
    POLICE_STATION: Optional[str] = None

    image_id: str
    image_path: str
    distance: float
    score: Optional[float] = None
    below_threshold: bool
    IMAGE_BASE64: Optional[str] = None

class MultiMatchItem(BaseModel):
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
    DISTRICT: Optional[str] = None
    POLICE_STATION: Optional[str] = None
    
    image_id: str
    image_path: str
    distance: float
    score: Optional[float] = None
    below_threshold: bool
    IMAGE_BASE64: Optional[str] = None

class Base64ImageInput(BaseModel):
    image_base64: str
    top_k: int = 5

class RecognitionResponse(BaseModel):
    query_image: Optional[str]
    matches: List[MatchItem]

class FaceSearchResult(BaseModel):
    face_id: int
    bbox: List[int]
    matches: List[MultiMatchItem]

class MultiFaceRecognitionResponse(BaseModel):
    total_faces: int
    bbox_base64: Optional[str]=None
    results: List[FaceSearchResult]

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

@app.post("/file-to-base64", response_model=Base64Response)
async def file_to_base64(file: UploadFile):
    """
    Accepts an image file and converts it to Base64 string.
    Works for JPG, PNG, JPEG.
    """

    # Read file bytes
    contents = await file.read()

    # Convert to Base64
    base64_str = base64.b64encode(contents).decode("utf-8")

    # Optionally add header (remove if not needed)
    mime_type = file.content_type  # e.g., image/jpeg
    base64_with_header = f"data:{mime_type};base64,{base64_str}"

    return Base64Response(
        filename=file.filename,

        base64_string=base64_str
    )

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(payload: Base64ImageInput):
    """
    Upload a face image base64 and get top_k nearest matches from existing data/images.
    """
    global faiss_index

    
    # 1️⃣ Extract & decode Base64
    base64_string = payload.image_base64
    top_k = payload.top_k

    if "," in base64_string:  
        base64_string = base64_string.split(",")[1]  

    try:
        contents = base64.b64decode(base64_string)
    except Exception:
        print("🚫 Invalid Base64 format")
        return RecognitionResponse(query_image=None, matches=[])

    print(f"📸 Received base64 image, size={len(contents)} bytes")

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
    
    id_map = get_faiss_id_map("data/faiss/id_map.json")

    for idx, dist in zip(indices, distances):
        image_id = id_map.get(str(idx))  
        if not image_id:
            continue  

        path = id_to_image_path(image_id)
        score = normalize_distance_to_score(float(dist))
        below_threshold = score >= (1.0 - threshold)
       
        print("IDX from FAISS:", idx)
        print("IMAGE_ID from ID_MAP:", image_id)

        meta = get_image_row_by_filesrno_check(file_srno = int(image_id), image_base_path = "temp_images")
        print(meta)

        
      

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
                DISTRICT = meta.get("DISTRICT"),
                POLICE_STATION = meta.get("POLICE_STATION"),

                

                image_id=image_id,
                image_path=path,
                distance=float(dist),
                score=score,
                below_threshold=below_threshold,
                IMAGE_BASE64= meta.get("image_base64")
            )
        )


    return RecognitionResponse(query_image=None, matches=matches)

@app.post("/multi-face-search", response_model=MultiFaceRecognitionResponse)
async def multi_face_search(payload: Base64ImageInput):
    global faiss_index

    # contents = await file.read()
    # 1️⃣ Extract & decode Base64
    base64_string = payload.image_base64
    top_k = payload.top_k

    if "," in base64_string:  
        base64_string = base64_string.split(",")[1]  

    try:
        contents = base64.b64decode(base64_string)
    except Exception:
        print("🚫 Invalid Base64 format")
        return RecognitionResponse(query_image=None, matches=[])

    print(f"📸 Received base64 image, size={len(contents)} bytes")

    # Extract all faces
    faces, bbox_base64 = extract_multiple_faces_bbox(contents)
    if not faces:
        logger.info("No face detected")
        return MultiFaceRecognitionResponse(total_faces=0, results=[])

    if faiss_index is None:
        faiss_index = get_or_build_faiss_index()

    id_map = get_faiss_id_map("data/faiss/id_map.json")

    all_results = []

    for face in faces:
        emb = np.array(face["embedding"], dtype=np.float32).reshape(1, -1)

        distances, indices = search_faiss_index(faiss_index, emb, top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        matches = []

        # -----------------------------------------
        # STRICT THRESHOLD FILTERING
        # -----------------------------------------
        for idx, dist in zip(indices, distances):

            image_id = id_map.get(str(idx))
            if not image_id:
                logger.warning(f"Missing id_map entry for FAISS index {idx}")
                continue

            meta = get_image_row_by_filesrno_check(
                file_srno=int(image_id),
                image_base_path="temp_images"
            )
            if not meta:
                logger.warning(f"No metadata for image_id {image_id}")
                continue

            path = id_to_image_path(image_id)
            score = normalize_distance_to_score(dist)

            # Only allow above threshold
            if score < threshold:
                continue

            matches.append(
                MultiMatchItem(
                    face_id=face["face_id"],

                    FILE_SRNO=meta.get("FILE_SRNO"),
                    ACCUSED_SRNO=meta.get("ACCUSED_SRNO"),
                    FILE_NAME=meta.get("FILE_NAME"),
                    ACCUSED_NAME=meta.get("ACCUSED_NAME"),
                    RELATIVE_NAME=meta.get("RELATIVE_NAME"),
                    AGE=meta.get("AGE"),
                    GENDER=meta.get("GENDER"),
                    FIR_REG_NUM=meta.get("FIR_REG_NUM"),
                    IMAGE_PATH=meta.get("image_path"),
                    DISTRICT = meta.get("DISTRICT"),
                    POLICE_STATION = meta.get("POLICE_STATION"),
                    

                    image_id=image_id,
                    image_path=path,
                    distance=float(dist),
                    score=score,
                    below_threshold=False,
                    IMAGE_BASE64=meta.get("image_base64")
                    
                )
            )

        # -----------------------------------------
        # EVEN IF ZERO MATCHES → RETURN EMPTY FOR THIS FACE
        # -----------------------------------------
        all_results.append(
            FaceSearchResult(
                face_id=face["face_id"],
                bbox=face["bbox"],
                matches=matches
            )
        )

    return MultiFaceRecognitionResponse(
        total_faces=len(faces),
        bbox_base64 = bbox_base64,
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
