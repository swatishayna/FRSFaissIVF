import os
import csv
from datetime import datetime
import pandas as pd
import shutil
import numpy as np

def repair_and_log_corrupted_embeddings(
    embedding_path,
    id_path,
    metadata_path,
    failed_metadata_path,
    embedding_dim=512,
    image_dir=None
):
    """
    Detect and remove corrupted or incomplete embeddings (e.g., due to interrupted process).

    Performs:
    - Detects size mismatch between embeddings and image IDs.
    - Trims corrupted embeddings and IDs.
    - Updates metadata.csv to remove those entries.
    - Appends removed rows to metadata_failed_embeddings.csv for regeneration.
    - Deletes corresponding image files from project-level data/images.
    - Backs up all files before modifying.

    Returns:
        int: Number of corrupted embeddings removed.
    """
    # --- Resolve project root image directory ---
    if image_dir is None:
        # resolve ../../data/images from utils/extra_utils.py
        image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "data", "images"))

    if not os.path.exists(embedding_path) or not os.path.exists(id_path):
        logger.error("❌ Embedding or ID file not found.")
        return 0

    try:
        embeddings = np.load(embedding_path, allow_pickle=False)
        image_ids = np.load(id_path, allow_pickle=False)
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings or IDs: {e}")
        return 0

    # --- Detect mismatch ---
    total_ids = len(image_ids)
    total_embeddings = embeddings.size // embedding_dim

    if total_embeddings == total_ids:
        logger.info("✅ No corruption detected in embeddings.")
        return 0

    corrupted_count = abs(total_ids - total_embeddings)
    logger.warning(f"⚠️ Detected {corrupted_count} corrupted entries. Repairing...")

    # --- Backup originals ---
    backup_dir = os.path.join(os.path.dirname(embedding_path), "backup")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(embedding_path, os.path.join(backup_dir, "embeddings_backup.npy"))
    shutil.copy2(id_path, os.path.join(backup_dir, "image_ids_backup.npy"))
    if os.path.exists(metadata_path):
        shutil.copy2(metadata_path, os.path.join(backup_dir, "metadata_backup.csv"))
    logger.info(f"💾 Backup saved to {backup_dir}")

    # --- Trim corrupted data ---
    embeddings = embeddings.reshape(-1, embedding_dim)[:total_embeddings]
    valid_ids = image_ids[:total_embeddings]
    corrupted_ids = image_ids[total_embeddings:]

    np.save(embedding_path, embeddings)
    np.save(id_path, valid_ids)
    logger.info(f"✅ Trimmed corrupted entries. New shape: {embeddings.shape}")

    # --- Update metadata ---
    if os.path.exists(metadata_path):
        df_meta = pd.read_csv(metadata_path)
        df_meta["FILE_SRNO"] = df_meta["FILE_SRNO"].astype(str)
        corrupted_ids_str = set(map(str, corrupted_ids))

        # Separate corrupted and clean rows
        df_failed = df_meta[df_meta["FILE_SRNO"].isin(corrupted_ids_str)]
        df_clean = df_meta[~df_meta["FILE_SRNO"].isin(corrupted_ids_str)]

        # Overwrite metadata with clean entries
        df_clean.to_csv(metadata_path, index=False)
        logger.info(f"🧹 Removed {len(df_failed)} corrupted entries from metadata.csv")

        # Append failed entries to failed_metadata.csv
        if len(df_failed) > 0:
            if os.path.exists(failed_metadata_path):
                df_prev = pd.read_csv(failed_metadata_path)
                df_failed = pd.concat([df_prev, df_failed], ignore_index=True)
                df_failed.drop_duplicates(subset=["FILE_SRNO"], keep="last", inplace=True)
            df_failed.to_csv(failed_metadata_path, index=False)
            logger.info(f"📁 Appended {len(df_failed)} rows to metadata_failed_embeddings.csv")

            # --- Delete corresponding image files ---
            deleted_images = 0
            for fid in corrupted_ids_str:
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_path = os.path.join(image_dir, f"{fid}{ext}")
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted_images += 1
                        break  # stop after first match

            logger.info(f"🗑️ Deleted {deleted_images} corrupted image files from {image_dir}")

    else:
        logger.warning("⚠️ metadata.csv not found — skipping metadata cleanup.")

    logger.info("🧠 Embedding repair complete.")
    return corrupted_count


def save_image_locally(image_bytes, file_srno, output_dir="data/images"):
    """
    Save image bytes as a .jpg file to the specified output directory.
    Args:
        image_bytes (bytes): The image blob (from DB or upload)
        file_srno (str/int): Unique ID or filename
        output_dir (str): Directory to save images
    Returns:
        str: Path to the saved image file
    """
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{file_srno}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    print(f"✅ Image saved at: {image_path}")
    return image_path


def append_metadata(metadata_path, record_dict, header=None):
    """
    Append a metadata row to metadata.csv.
    Args:
        metadata_path (str): CSV path (e.g., data/metadata.csv)
        record_dict (dict): Key-value pairs to append
        header (list, optional): Column names (written if file doesn't exist)
    """
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    file_exists = os.path.exists(metadata_path)

    with open(metadata_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header or record_dict.keys())

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        record_dict["created_at"] = datetime.now().isoformat()
        writer.writerow(record_dict)

    print(f"📝 Metadata appended for FILE_SRNO={record_dict.get('FILE_SRNO')}")


def drop_duplicate_rows(filepath, unique_cols):
    """
    Remove duplicate rows from a CSV file based on a combination of columns.

    Args:
        filepath (str): Path to the CSV file.
        unique_cols (list): List of column names to consider as a unique key.

    Example:
        drop_duplicate_rows("data/metadata.csv", ["FILE_SRNO", "ACCUSED_SRNO"])
    """
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath, on_bad_lines="skip", engine="python")

        before = len(df)

        # Drop duplicates based on unique column combination
        df.drop_duplicates(subset=unique_cols, keep="last", inplace=True)

        after = len(df)
        df.to_csv(filepath, index=False)

        print(f"✅ Cleaned {filepath}: Removed {before - after} duplicates. Final rows: {after}")
    except Exception as e:
        print(f"❌ Error cleaning {filepath}: {e}")

import os
import pandas as pd

def load_already_processed(METADATA_PATH, FAILED_METADATA_PATH):
    processed = set()

    def safe_load(path):
        """Load CSV even if some rows are corrupted."""
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception:
            print(f"⚠️ Warning: Corrupted CSV detected in {path}, loading with on_bad_lines='skip'")
            return pd.read_csv(path, on_bad_lines="skip")

    # Load metadata safely
    df1 = safe_load(METADATA_PATH)
    if "FILE_SRNO" in df1.columns:
        processed.update(df1["FILE_SRNO"].dropna().astype(int).tolist())

    # Load failed metadata safely
    df2 = safe_load(FAILED_METADATA_PATH)
    if "FILE_SRNO" in df2.columns:
        processed.update(df2["FILE_SRNO"].dropna().astype(int).tolist())

    print(f"✔ Loaded processed FILE_SRNO count: {len(processed)}")
    return processed
