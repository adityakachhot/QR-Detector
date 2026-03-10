import os
import csv

base_path = "master_dataset"
output_file = "dataset_metadata.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filepath", "label", "source"])

    for source in os.listdir(base_path):
        source_path = os.path.join(base_path, source)

        if os.path.isdir(source_path):
            for label in ["benign", "malicious"]:
                label_path = os.path.join(source_path, label)

                for img in os.listdir(label_path):
                    full_path = os.path.join(label_path, img)
                    writer.writerow([full_path, label, source])

print("Metadata CSV created successfully.")