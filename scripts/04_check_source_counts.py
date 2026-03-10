import os

base_path = "master_dataset"

for source in os.listdir(base_path):
    source_path = os.path.join(base_path, source)
    if os.path.isdir(source_path):
        benign = len(os.listdir(os.path.join(source_path, "benign")))
        malicious = len(os.listdir(os.path.join(source_path, "malicious")))

        print(f"{source.upper()}")
        print(f"  Benign: {benign}")
        print(f"  Malicious: {malicious}")
        print(f"  Total: {benign + malicious}")
        print("-" * 30)