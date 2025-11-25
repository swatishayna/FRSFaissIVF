import os
import json
import logging

# Path for the cache file
CACHE_FILE = os.path.join("data", "cache.json")

# Ensure parent folder exists
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

logger = logging.getLogger(__name__)

def get_cache():
    """Read cache.json and return its contents as a dict."""
    if not os.path.exists(CACHE_FILE):
        logger.info("🗂️ No cache file found, starting fresh.")
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"⚠️ Failed to read cache file: {e}")
        return {}

def set_cache(data: dict):
    """Write the given dictionary to cache.json."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"💾 Cache updated: {data}")
    except Exception as e:
        logger.error(f"⚠️ Failed to write cache file: {e}")
