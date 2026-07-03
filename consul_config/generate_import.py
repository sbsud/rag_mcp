# consul_config/generate_import.py
import json, base64, sys

with open("ecommerce_config_plaintext.json") as f:
    plain = json.load(f)

output = [
    {"key": k, "value": base64.b64encode(v.encode()).decode()}
    for k, v in plain.items()
]

with open("ecommerce_config.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(output)} keys.")