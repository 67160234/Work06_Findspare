import os
import clip
import torch
import pickle
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device)

DATASET_PATH = "dataset"

vectors = []

for part_name in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, part_name)

    if not os.path.isdir(folder):
        continue

    for img in os.listdir(folder):

        path = os.path.join(folder, img)

        try:

            image = preprocess(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                vec = model.encode_image(image)

            vec = vec / vec.norm(dim=-1, keepdim=True)

            vectors.append({
                "part_name": part_name,
                "vector": vec.cpu().numpy()[0]
            })

        except:
            pass

# save pkl
os.makedirs("embeddings", exist_ok=True)

with open("embeddings/dataset_vectors.pkl","wb") as f:
    pickle.dump(vectors,f)

print("Dataset embeddings saved")