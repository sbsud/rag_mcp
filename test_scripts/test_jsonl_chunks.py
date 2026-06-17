import json
from pathlib import Path

path = Path("../data_dump/amazon_Appliances_complaints.jsonl")
sizes = []
split_count = 0
CHUNK_SIZE = 400

with open(path) as f:
    for line in f:
        record = json.loads(line.strip())
        text_len = len(record["text"])
        sizes.append(text_len)
        if text_len > CHUNK_SIZE:
            split_count += 1

sizes.sort()
print(f"Records:           {len(sizes)}")
print(f"Records > 400:     {split_count}  ({100*split_count/len(sizes):.1f}%)")
print(f"Min text length:   {sizes[0]}")
print(f"Median:            {sizes[len(sizes)//2]}")
print(f"90th percentile:   {sizes[int(len(sizes)*0.9)]}")
print(f"Max text length:   {sizes[-1]}")

oversized = []
with open(path) as f:
    for line in f:
        record = json.loads(line.strip())
        text_len = len(record["text"])
        if text_len > CHUNK_SIZE:
            oversized.append((text_len, record["metadata"]["brand"], record["metadata"]["rating"]))

oversized.sort(reverse=True)
for size, brand, rating in oversized[:20]:
    print(f"{size:5d} chars  brand={brand}  rating={rating}")