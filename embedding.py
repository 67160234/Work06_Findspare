import pymysql
import clip
import torch
from PIL import Image
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

SHOP_IMAGE_PATH = "shop_parts"

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="findspares",
    cursorclass=pymysql.cursors.DictCursor
)

cursor = conn.cursor()

cursor.execute("SELECT id, image FROM shop_parts")

parts = cursor.fetchall()

print("Total parts:", len(parts))

count = 0

for part in parts:

    img_path = os.path.join(SHOP_IMAGE_PATH, part["image"])

    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        continue

    try:

        image = Image.open(img_path).convert("RGB")

        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image)

        features = features / features.norm(dim=-1, keepdim=True)

        vector = features.cpu().numpy().tolist()[0]

        cursor.execute("""
        REPLACE INTO part_embeddings (part_id, embedding)
        VALUES (%s,%s)
        """,(part["id"], json.dumps(vector)))

        count += 1

        print("Embedded:", part["image"])

    except Exception as e:

        print("Error with image:", part["image"], e)

conn.commit()
conn.close()

print("Embedding created:", count)