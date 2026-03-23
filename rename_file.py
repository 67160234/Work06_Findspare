import os

DATASET_PATH = "dataset"

for part_folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, part_folder)

    if not os.path.isdir(folder_path):
        continue

    count = 1

    for filename in os.listdir(folder_path):

        old_path = os.path.join(folder_path, filename)

        ext = os.path.splitext(filename)[1]

        new_name = f"{part_folder}_{count}{ext}"

        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

        print(f"{filename} -> {new_name}")

        count += 1

print("Rename finished")