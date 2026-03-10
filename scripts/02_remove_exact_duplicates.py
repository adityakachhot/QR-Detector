import os
import hashlib
from tqdm import tqdm

base_path = "master_dataset"
hashes = {}
duplicates = []

def get_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

for label in ["benign", "malicious"]:
    folder = os.path.join(base_path, label)
    files = os.listdir(folder)

    for file in tqdm(files, desc=f"Checking {label}"):
        path = os.path.join(folder, file)
        file_hash = get_hash(path)

        if file_hash in hashes:
            duplicates.append(path)
        else:
            hashes[file_hash] = path

print("Total exact duplicates found:", len(duplicates))

# Remove duplicates
for dup in duplicates:
    os.remove(dup)

print("Exact duplicate removal complete.")