#!/bin/sh

consul agent -server -bootstrap-expect=1 -ui -client=0.0.0.0 -data-dir=/consul/data &

RETRIES=0
until curl -sf http://localhost:8500/v1/status/leader > /dev/null 2>&1; do
  RETRIES=$((RETRIES + 1))
  if [ $RETRIES -gt 30 ]; then
    echo "Consul failed to start after 60s — exiting"
    exit 1
  fi
  echo "Waiting for Consul... ($RETRIES/30)"
  sleep 2
done

echo "Consul ready — loading KV config..."

jq -c '.[]' /consul_config/ecommerce_config_plaintext.json | \
while read -r entry; do
  key=$(echo "$entry" | jq -r '.key')
  value=$(echo "$entry" | jq -r '.value')
  curl -sf -X PUT "http://localhost:8500/v1/kv/${key}" --data "${value}"
  echo "  ✓ ${key}"
done

echo "Bootstrap complete."
wait