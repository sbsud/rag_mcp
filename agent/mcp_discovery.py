import os
import socket
import logging
import requests

logger = logging.getLogger(__name__)

CONSUL_URL = os.getenv("CONSUL_URL") 


def discover_services(names: list[str]) -> list[dict]:
    servers = []
    for name in names:
        try:
            r = requests.get(f"{CONSUL_URL}/v1/catalog/service/{name}", timeout=3)
            r.raise_for_status()
            instances = r.json()
            if not instances:
                raise ValueError(f"No instances of '{name}' in Consul")
            svc = instances[0]
            # Use ServiceName for DNS resolution, not ServiceAddress (container hostname)
            url = f"http://{svc['ServiceName']}:{svc['ServicePort']}/sse"
            servers.append({"name": name, "url": url})
            logger.info("Discovered '%s' at %s", name, url)
        except Exception as e:
            logger.warning("Consul discovery failed for '%s': %s", name, e)
    return servers