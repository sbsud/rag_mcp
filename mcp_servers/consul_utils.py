import os
import socket
import logging
import requests

logger = logging.getLogger(__name__)

CONSUL_URL = os.getenv("CONSUL_URL")  # None in local dev — registration skipped

def register_service(name: str, port: int):
    if not CONSUL_URL:
        logger.info("CONSUL_URL not set — skipping Consul registration for '%s'", name)
        return

    address = socket.gethostname()
    payload = {
        "Name": name,
        "ID": f"{name}-1",
        "Address": address,
        "Port": port,
        "Check": {
            "HTTP": f"http://{address}:{port}/health",
            "Interval": "10s",
            "Timeout": "2s",
            "DeregisterCriticalServiceAfter": "30s"
        }
    }

    for attempt in range(3):
        try:
            r = requests.put(
                f"{CONSUL_URL}/v1/agent/service/register",
                json=payload,
                timeout=3
            )
            r.raise_for_status()
            logger.info("Registered '%s' with Consul at %s:%d", name, address, port)
            return
        except Exception as e:
            logger.warning("Consul registration attempt %d/3 failed for '%s': %s", attempt + 1, name, e)
            if attempt < 2:
                import time; time.sleep(2)

    logger.warning("Consul registration failed for '%s' after 3 attempts — continuing without it", name)

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