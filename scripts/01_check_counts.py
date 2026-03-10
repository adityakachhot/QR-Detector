import os

base_path = "master_dataset"

benign_count = len(os.listdir(os.path.join(base_path, "benign")))
malicious_count = len(os.listdir(os.path.join(base_path, "malicious")))

print(f"Benign images: {benign_count}")
print(f"Malicious images: {malicious_count}")
print(f"Total images: {benign_count + malicious_count}")