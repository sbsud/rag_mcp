import os
import socket
import logging
import requests

logger = logging.getLogger(__name__)

CONSUL_URL = os.getenv("CONSUL_URL") 

def get_kv(key: str, fallback: str = None) -> str:
    if not CONSUL_URL:
        return fallback
    try:
        r = requests.get(f"{CONSUL_URL}/v1/kv/{key}?raw", timeout=3)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("Consul KV read failed for '%s': %s — using fallback", key, e)
        return fallback