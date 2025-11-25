import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss

from utils.config_utils import load_config
from utils.db_utils import fetch_images_from_db, get_connection
from utils.embedding_utils import EmbeddingStore
from utils.faiss_utils import save_faiss_index
from utils.extra_utils import append_metadata, save_image_locally, drop_duplicate_rows, load_already_processed
from utils.logger_utils import setup_logger

# ---------------------------------------
# Initialize Configuration and Logger
# ---------------------------------------
config = load_config()
setup_logger(config)
logger = logging.getLogger(__name__)

# ---------------------------------------
# Paths and Constants
# ---------------------------------------
BASE_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_DIR = os.path.join(BASE_DIR, "faiss")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

METADATA_PATH = os.path.join(METADATA_DIR, "metadata.csv")
FAILED_METADATA_PATH = os.path.join(METADATA_DIR, "metadata_failed_embeddings.csv")
EMBED_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
ID_PATH = os.path.join(EMBED_DIR, "image_ids.npy")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.json")
FAISS_IVF_PATH = os.path.join(FAISS_DIR, "faiss_ivf.index")
ID_MAP_PATH = os.path.join(FAISS_DIR, "id_map.json")

for p in [IMAGE_DIR, EMBED_DIR, FAISS_DIR, METADATA_DIR]:
    os.makedirs(p, exist_ok=True)


# ---------------------------------------
# Initialize Components
# ---------------------------------------
embedding_store = EmbeddingStore(embedding_path=EMBED_PATH, id_path=ID_PATH)
db_conn = get_connection()

# ---------------------------------------
# Load Face Model
# ---------------------------------------
from insightface.app import FaceAnalysis
logger.info("🧠 Loading face embedding model ...")
app = FaceAnalysis(name=config.get("face_model", "buffalo_l"))
app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("✅ Face model loaded successfully.")


# ---------------------------------------
# Helper: Checkpoint Management
# ---------------------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
            return data.get("last_processed_id", 0)
    return 0


def save_checkpoint(last_processed_id):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"last_processed_id": last_processed_id, "timestamp": str(datetime.now())}, f)
    logger.info(f"💾 Checkpoint updated: last_processed_id = {last_processed_id}")


# ---------------------------------------
# Helper: Build FAISS IVF Index
# ---------------------------------------
def build_ivf_index_from_embeddings(embedding_path, faiss_output_path, dim=512, nlist=1000):
    """Train and save FAISS IVF index from embeddings."""
    if not os.path.exists(embedding_path):
        logger.error(f"❌ Embedding file not found at {embedding_path}")
        return

    embeddings = np.load(embedding_path).astype(np.float32)
    logger.info(f"📦 Loaded {len(embeddings)} embeddings for IVF training.")

    if len(embeddings) < nlist:
        nlist = max(8, len(embeddings) // 2)
        logger.warning(f"⚠️ Not enough embeddings ({len(embeddings)}). Reducing nlist → {nlist}")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    logger.info(f"🔧 Training IVF index with nlist={nlist} ...")
    index.train(embeddings)
    index.add(embeddings)
    if os.path.exists(ID_PATH):
        ids = np.load(ID_PATH, allow_pickle=True)
        id_map = {str(i): str(ids[i]) for i in range(len(ids))}
        with open(ID_MAP_PATH, "w") as f:
            json.dump(id_map, f, indent=4)
        logger.info(f"💾 Saved FAISS id_map.json → {ID_MAP_PATH}")
    else:
        logger.error("❌ Cannot create id_map.json because image_ids.npy not found!")

    save_faiss_index(index, faiss_output_path)
    logger.info(f"✅ Saved FAISS IVF index to {faiss_output_path}")


# ---------------------------------------
# Helper: Process Single Image
# ---------------------------------------
def process_single_row(row):
    try:
        # file_srno = int(row["FILE_SRNO"])
        file_srno = str(row["FILE_SRNO"]).strip()


        if embedding_store.exists(file_srno):
            return None

        img_blob = row.get("UPLOADED_FILE")
        if not img_blob:
            return None

        image_array = np.frombuffer(img_blob, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None

        faces = app.get(image)
        if not faces:
            return None

        embedding = faces[0].normed_embedding.astype(np.float32)
        image_path = save_image_locally(img_blob, file_srno, IMAGE_DIR)

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

        return (file_srno, embedding, metadata_row)
    except Exception as e:
        logger.error(f"❌ Error generating embedding for {row.get('FILE_SRNO', 'Unknown')}: {e}")
        return None


# ---------------------------------------
# Main Processing Function
# ---------------------------------------
# def process_images_continuously(batch_size=1000):
#     """Generate embeddings and resume from last checkpoint."""
    
#     logger.info("📌 Loading already processed FILE_SRNO values ...")
#     already_processed = load_already_processed(METADATA_PATH, FAILED_METADATA_PATH)
#     logger.info(f"🔹 Found {len(already_processed)} previously processed images")

#     last_processed_id = load_checkpoint()
#     logger.info(f"🚀 Resuming from FILE_SRNO > {last_processed_id}")

#     total_processed = 0
#     batch_num = 1

#     while True:
#         rows = fetch_images_from_db(db_conn, batch_size=batch_size, last_id=last_processed_id)
#         if not rows:
#             logger.info("✅ No new records found. Stopping.")
#             break

#         new_embeddings, new_ids, batch_metadata, failed_records = [], [], [], []
#         max_id_in_batch = last_processed_id

#         with ThreadPoolExecutor(max_workers=6) as executor:
#             futures = {executor.submit(process_single_row, row): row for row in rows}
#             for future in tqdm(as_completed(futures), total=len(rows), desc=f"Batch #{batch_num}", unit="img"):
#                 row = futures[future]
#                 file_srno = row.get("FILE_SRNO")

#                 if str(file_srno) in already_processed:
#                     continue
            


#                 try:
#                     result = future.result()
#                     if result:
#                         file_srno, embedding, metadata_row = result
#                         new_embeddings.append(embedding)
#                         new_ids.append(file_srno)
#                         batch_metadata.append(metadata_row)
#                         failed_records.append({"FILE_SRNO": file_srno, "STATUS": "success"})
#                         total_processed += 1
#                         max_id_in_batch = str(max(int(max_id_in_batch), int(file_srno)))
#                         #max_id_in_batch = max(str(max_id_in_batch), str(file_srno))

#                     else:
#                         failed_records.append({"FILE_SRNO": file_srno, "STATUS": "fail"})
#                 except Exception as e:
#                     failed_records.append({"FILE_SRNO": file_srno, "STATUS": f"error: {e}"})

#         # Save failed embeddings metadata
#         if failed_records:
#             df_failed = pd.DataFrame(failed_records)
#             if os.path.exists(FAILED_METADATA_PATH):
#                 old_failed = pd.read_csv(FAILED_METADATA_PATH)
#                 df_failed = pd.concat([old_failed, df_failed], ignore_index=True)
#                 df_failed.drop_duplicates(subset=["FILE_SRNO"], keep="last", inplace=True)
#             df_failed.to_csv(FAILED_METADATA_PATH, index=False)
#             logger.info(f"💾 Updated failed log: {FAILED_METADATA_PATH}")

#         # Save successful embeddings
#         if new_embeddings:
#             embeddings_np = np.array(new_embeddings, dtype=np.float32)
#             embedding_store.add(embeddings_np, new_ids)
#             for m in batch_metadata:
#                 append_metadata(METADATA_PATH, m)
#             logger.info(f"✅ Saved {len(new_embeddings)} embeddings.")
#         else:
#             logger.warning("⚠️ No valid embeddings in this batch.")

    
#         try:
#             if os.path.exists(METADATA_PATH):
#                 df_meta = pd.read_csv(METADATA_PATH)
#                 total_rows = len(df_meta)

#                 if total_rows >= 2000:
#                     logger.info(
#                         f"🛑 Metadata has {total_rows} rows. Stopping generation and moving to FAISS creation."
#                     )
#                     break
#         except Exception as e:
#             logger.error(f"Error checking metadata count: {e}")


#         # Save checkpoint
#         save_checkpoint(max_id_in_batch)
#         last_processed_id = max_id_in_batch
#         batch_num += 1

#     logger.info(f"🎯 Total processed embeddings: {total_processed}")



def process_images_continuously(batch_size=1000):
    """Generate embeddings and resume from last checkpoint."""
    
    logger.info("📌 Loading already processed FILE_SRNO values ...")
    already_processed = load_already_processed(METADATA_PATH, FAILED_METADATA_PATH)
    logger.info(f"🔹 Found {len(already_processed)} previously processed images")

    last_processed_id = load_checkpoint()
    logger.info(f"🚀 Resuming from FILE_SRNO > {last_processed_id}")

    total_processed = 0
    batch_num = 1

    while True:
        rows = fetch_images_from_db(db_conn, batch_size=batch_size, last_id=last_processed_id)
        if not rows:
            logger.info("✅ No new records found. Stopping.")
            break

        new_embeddings, new_ids, batch_metadata, failed_records = [], [], [], []
        max_id_in_batch = last_processed_id  # still string

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_single_row, row): row for row in rows}

            for future in tqdm(as_completed(futures), total=len(rows), desc=f"Batch #{batch_num}", unit="img"):
                row = futures[future]
                file_srno = row.get("FILE_SRNO")

                # Skip already processed
                if str(file_srno) in already_processed:
                    continue

                try:
                    result = future.result()
                    if result:
                        file_srno, embedding, metadata_row = result
                        new_embeddings.append(embedding)
                        new_ids.append(file_srno)
                        batch_metadata.append(metadata_row)
                        failed_records.append({"FILE_SRNO": file_srno, "STATUS": "success"})
                        total_processed += 1

                        # CORRECT FIX — numeric compare, store as string
                        try:
                            max_id_in_batch = str(max(int(max_id_in_batch), int(file_srno)))
                        except:
                            pass

                    else:
                        failed_records.append({"FILE_SRNO": file_srno, "STATUS": "fail"})

                except Exception as e:
                    failed_records.append({"FILE_SRNO": file_srno, "STATUS": f"error: {e}"})

        # Save failed log
        if failed_records:
            df_failed = pd.DataFrame(failed_records)
            if os.path.exists(FAILED_METADATA_PATH):
                old_failed = pd.read_csv(FAILED_METADATA_PATH)
                df_failed = pd.concat([old_failed, df_failed], ignore_index=True)
                df_failed.drop_duplicates(subset=["FILE_SRNO"], keep="last", inplace=True)
            df_failed.to_csv(FAILED_METADATA_PATH, index=False)
            logger.info(f"💾 Updated failed log: {FAILED_METADATA_PATH}")

        # Save embeddings
        if new_embeddings:
            embeddings_np = np.array(new_embeddings, dtype=np.float32)
            embedding_store.add(embeddings_np, new_ids)
            for m in batch_metadata:
                append_metadata(METADATA_PATH, m)
            logger.info(f"✅ Saved {len(new_embeddings)} embeddings.")
        else:
            logger.warning("⚠️ No valid embeddings in this batch.")

        # STOP CONDITION — metadata row count
        # try:
        #     if os.path.exists(METADATA_PATH):
        #         df_meta = pd.read_csv(METADATA_PATH)
        #         total_rows = len(df_meta)

        #         if total_rows >= 1500:
        #             logger.info(
        #                 f"🛑 Metadata has {total_rows} rows. Stopping generation and moving to FAISS creation."
        #             )
        #             break
        # except Exception as e:
        #     logger.error(f"Error checking metadata count: {e}")

        # SAFETY SKIP → If batch contains rows but no new embeddings
        # this prevents infinite looping when DB contains old FILE_SRNO values.
        if len(new_embeddings) == 0 and len(rows) > 0:
            logger.warning("⚠️ No new embeddings created in this batch. Advancing FILE_SRNO to continue...")
            last_processed_id = rows[-1]["FILE_SRNO"]
            continue

        # SAVE CHECKPOINT
        save_checkpoint(max_id_in_batch)
        last_processed_id = max_id_in_batch
        batch_num += 1

    logger.info(f"🎯 Total processed embeddings: {total_processed}")



# ---------------------------------------
# Main Entrypoint
# ---------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    try:
        # Step 1: Generate embeddings (resumable)
        process_images_continuously(batch_size=500)

        # Step 2: Drop duplicates from metadata
        # drop_duplicate_rows(filepath=METADATA_PATH, unique_cols=["FILE_SRNO", "ACCUSED_SRNO"])

        # Step 3: Build FAISS IVF index
        logger.info("🏗️ Building FAISS IVF index ...")
        build_ivf_index_from_embeddings(
            embedding_path=EMBED_PATH,
            faiss_output_path=FAISS_IVF_PATH,
            dim=config.get("faiss", {}).get("dim", 512),
            nlist=config.get("faiss", {}).get("nlist", 1000), #change here
        )

    finally:
        db_conn.close()

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Completed in {elapsed} seconds.")


# nohup python run_add_new_embeddings.py > embeddings_log.out 2>&1 &
# tail -f embeddings_log.out
