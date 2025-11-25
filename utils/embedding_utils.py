import numpy as np
import os

import cv2
from insightface.app import FaceAnalysis


class EmbeddingStore:
    def __init__(self, embedding_path="data/embeddings.npy", id_path="data/image_ids.npy"):
        self.embedding_path = embedding_path
        self.id_path = id_path
        self.embeddings = None
        self.image_ids = None
        self.load()

    def load(self):
        if os.path.exists(self.embedding_path) and os.path.exists(self.id_path):
            self.embeddings = np.load(self.embedding_path)
            self.image_ids = np.load(self.id_path)
        else:
            self.embeddings = np.zeros((0, 512), dtype='float32')
            self.image_ids = np.array([], dtype=str)

    def save(self):
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        np.save(self.embedding_path, self.embeddings)
        np.save(self.id_path, self.image_ids)

    def add(self, new_embeddings, new_ids):
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.image_ids = np.concatenate([self.image_ids, np.array(new_ids, dtype=str)])
        self.save()

    def exists(self, image_id):
        return image_id in self.image_ids




# -----------------------------
# Face Embedding Generator
# -----------------------------
def generate_face_embedding(app, img_blob):
    """
    Given an image BLOB and an InsightFace app instance,
    returns a 512-d normalized face embedding.
    """
    try:
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
        return embedding

    except Exception as e:
        logger.error(f"❌ Error generating face embedding: {e}")
        return None

