import os
import hashlib
from tqdm import tqdm

base_path = "master_dataset"

def get_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

benign_hashes = set()

# Hash benign
for file in tqdm(os.listdir(os.path.join(base_path, "benign")), desc="Hashing benign"):
    path = os.path.join(base_path, "benign", file)
    benign_hashes.add(get_hash(path))

cross_duplicates = []

# Check malicious
for file in tqdm(os.listdir(os.path.join(base_path, "malicious")), desc="Checking malicious"):
    path = os.path.join(base_path, "malicious", file)
    if get_hash(path) in benign_hashes:
        cross_duplicates.append(file)

print("Cross-class identical images:", len(cross_duplicates))