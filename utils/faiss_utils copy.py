import os
import faiss
import numpy as np
import json


def load_faiss_index(index_path, faiss_config=None):
    """
    Load an existing FAISS index if it exists, otherwise create a new one.
    Args:
        index_path (str): Path to the FAISS index file.
        faiss_config (dict): Optional FAISS configuration (e.g., dimension, metric type).
    Returns:
        faiss.Index: The loaded or newly created FAISS index.
    """
    try:
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            print(f"✅ Loaded existing FAISS index from: {index_path}")
        else:
            dim = faiss_config.get("dimension", 512) if faiss_config else 512
            metric_type = faiss.METRIC_L2
            index = faiss.IndexFlatL2(dim)
            print(f"🆕 Created new FAISS index with dimension={dim}.")
        return index
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        raise e


def add_to_faiss_index(index, embeddings):
    """
    Add embeddings to an existing FAISS index.
    Args:
        index (faiss.Index): The FAISS index.
        embeddings (np.ndarray): A numpy array of shape (N, D) containing embeddings.
    Returns:
        faiss.Index: Updated index.
    """
    try:
        if embeddings is None or len(embeddings) == 0:
            print("⚠️ No embeddings to add to FAISS index.")
            return index

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        index.add(embeddings)
        print(f"✅ Added {embeddings.shape[0]} embeddings to FAISS index.")
        return index
    except Exception as e:
        print(f"❌ Error adding embeddings to FAISS index: {e}")
        raise e


def save_faiss_index(index, index_path):
    """
    Save FAISS index to disk.
    Args:
        index (faiss.Index): The FAISS index object.
        index_path (str): File path to save index.
    """
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        print(f"💾 FAISS index saved at: {index_path}")
    except Exception as e:
        print(f"❌ Error saving FAISS index: {e}")
        raise e


def search_faiss_index(index, query_embedding, top_k=5):
    """
    Search for similar embeddings in the FAISS index.
    Args:
        index (faiss.Index): The FAISS index.
        query_embedding (np.ndarray): Single embedding vector (1, D).
        top_k (int): Number of nearest neighbors to return.
    Returns:
        distances (np.ndarray), indices (np.ndarray)
    """
    try:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = index.search(query_embedding, top_k)
        return distances, indices
    except Exception as e:
        print(f"❌ Error searching FAISS index: {e}")
        raise e


def create_idmapped_faiss_index(dimension=512):
    """
    Create a FAISS index that supports mapping embeddings to IDs.
    """
    base_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap2(base_index)
    print(f"🆕 Created ID-mapped FAISS index (dim={dimension})")
    return index


def add_embeddings_with_ids(faiss_index, embeddings, image_ids, id_map_path="data/faiss/id_map.json"):
    """
    Add embeddings to FAISS with corresponding string-based IDs.
    - image_ids: list of strings
    - embeddings: np.array of shape (n, 512)
    """
    os.makedirs(os.path.dirname(id_map_path), exist_ok=True)

    # Convert string IDs to unique int64 IDs
    int_ids = np.array([abs(hash(i)) % (2**63) for i in image_ids], dtype=np.int64)

    # Add to FAISS
    faiss_index.add_with_ids(embeddings.astype("float32"), int_ids)
    print(f"✅ Added {len(image_ids)} embeddings with mapped IDs.")

    # Save ID ↔ name map
    id_map = {str(i): sid for i, sid in zip(int_ids, image_ids)}
    if os.path.exists(id_map_path):
        with open(id_map_path, "r") as f:
            existing_map = json.load(f)
        existing_map.update(id_map)
        id_map = existing_map
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)

    print(f"💾 Saved FAISS ID ↔ image_id map to {id_map_path}")


def get_faiss_id_map(id_map_path="data/faiss/id_map.json"):
    """Load the FAISS ID ↔ image_id mapping."""
    if os.path.exists(id_map_path):
        with open(id_map_path, "r") as f:
            return json.load(f)
    return {}


def get_faiss_image_ids(faiss_index, id_map_path="data/faiss/id_map.json"):
    """
    Retrieve all FAISS IDs mapped to image IDs.
    """
    id_map = get_faiss_id_map(id_map_path)
    ids = faiss_index.id_map if hasattr(faiss_index, "id_map") else []
    return [id_map.get(str(i), str(i)) for i in ids]


def force_rebuild_faiss_index(
    embedding_path="data/embeddings/embeddings.npy",
    id_path="data/embeddings/image_ids.npy",
    faiss_index_path="data/faiss/faiss.index",
    id_map_path="data/faiss/id_map.json",
    dimension=512
):
    """
    Force rebuild the FAISS index from embeddings and image IDs.

    Steps:
    1. Loads embeddings.npy and image_ids.npy
    2. Normalizes embeddings (L2)
    3. Rebuilds a fresh ID-mapped FAISS index
    4. Saves the rebuilt index and updated ID map
    5. Backs up the old FAISS index if it exists

    Returns:
        int: Number of embeddings successfully indexed
    """
    import shutil

    try:
        if not os.path.exists(embedding_path) or not os.path.exists(id_path):
            print(f"❌ Missing embeddings or image_ids file. Cannot rebuild FAISS index.{embedding_path}")
            return 0

        embeddings = np.load(embedding_path).astype("float32")
        image_ids = np.load(id_path)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, dimension)

        if embeddings.shape[0] == 0:
            print("⚠️ No embeddings found. Skipping rebuild.")
            return 0

        # Normalize for L2 distance consistency
        faiss.normalize_L2(embeddings)

        # Trim mismatched lengths
        if len(embeddings) != len(image_ids):
            print(f"⚠️ Length mismatch — embeddings: {len(embeddings)}, IDs: {len(image_ids)}. Truncating to smallest.")
            min_len = min(len(embeddings), len(image_ids))
            embeddings = embeddings[:min_len]
            image_ids = image_ids[:min_len]

        # Backup old FAISS index
        if os.path.exists(faiss_index_path):
            backup_dir = os.path.join(os.path.dirname(faiss_index_path), "backup")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(faiss_index_path, os.path.join(backup_dir, "faiss_backup.index"))
            print(f"💾 Backed up old FAISS index to {backup_dir}")

        # Create a new ID-mapped index
        index = create_idmapped_faiss_index(dimension=dimension)

        # Add embeddings with mapped IDs
        add_embeddings_with_ids(index, embeddings, image_ids, id_map_path=id_map_path)

        # Save the rebuilt index
        save_faiss_index(index, faiss_index_path)

        print(f"✅ Rebuilt FAISS index with {len(image_ids)} embeddings.")
        return len(image_ids)

    except Exception as e:
        print(f"❌ Error rebuilding FAISS index: {e}")
        raise e

force_rebuild_faiss_index()