import streamlit as st
import pymysql
import torch
import clip
import numpy as np
import json
from PIL import Image
import faiss
import math
import os

st.set_page_config(page_title="FindSpares AI", layout="wide")

# ---------------------------
# LOAD MODEL
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------------------
# DATABASE (รองรับทั้งการรัน Local และบน Cloud (Hugging Face / Streamlit Community))
# - รัน Local: ระบบจะอ่านค่าจากโฟลเดอร์ .streamlit/secrets.toml
# - รัน Cloud: ระบบจะอ่านค่าจาก Settings -> Variables and secrets ของโฮสต์
# ---------------------------

try:
    conn = pymysql.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"],
        port=int(st.secrets.get("DB_PORT", 3306)),
        cursorclass=pymysql.cursors.DictCursor
    )
except FileNotFoundError:
    st.error('ไม่พบไฟล์ตั้งค่ารหัสผ่าน! หากรันในเครื่องตัวเอง กรุณาสร้างโฟลเดอร์ ".streamlit" และไฟล์ "secrets.toml" ภายในโฟลเดอร์โปรเจกต์')
    st.stop()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {e}")
    st.stop()

cursor = conn.cursor()

# ---------------------------
# DISTANCE FUNCTION
# ---------------------------

def distance(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)

    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R*c


# ---------------------------
# TRANSLATE KEYWORD
# ---------------------------

def translate_keyword(keyword):

    cursor.execute("""
    SELECT part_name
    FROM part_synonyms
    WHERE synonym LIKE %s
    """, ("%"+keyword+"%",))

    r = cursor.fetchone()

    if r:
        return r["part_name"]

    return keyword


# ---------------------------
# LOAD EMBEDDINGS
# ---------------------------

def load_vectors():

    cursor.execute("""
    SELECT
    sp.id,
    sp.part_name,
    sp.image,
    s.shop_name,
    s.latitude,
    s.longitude,
    s.google_map_link,
    pe.embedding
    FROM part_embeddings pe
    JOIN shop_parts sp ON pe.part_id=sp.id
    JOIN shops s ON sp.shop_id=s.id
    """)

    data = cursor.fetchall()

    vectors = []
    items = []

    for d in data:

        vec = np.array(json.loads(d["embedding"])).astype("float32")
        vec = vec / np.linalg.norm(vec)

        vectors.append(vec)
        items.append(d)

    vectors = np.array(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, items


index, items = load_vectors()

# ---------------------------
# ENCODE TEXT
# ---------------------------

def encode_text(text):

    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        vec = model.encode_text(text)

    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy().astype("float32")


# ---------------------------
# ENCODE IMAGE
# ---------------------------

def encode_image(img):

    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model.encode_image(img)

    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy().astype("float32")


# ---------------------------
# SEARCH
# ---------------------------

def search(query_vec, user_lat, user_lng, query=None):

    D, I = index.search(query_vec, 200)

    results = []
    seen = set()

    for score, i in zip(D[0], I[0]):

        item = items[i]

        key = item["id"]   # แก้ตรงนี้

        if key in seen:
            continue

        seen.add(key)

        dist = distance(
            user_lat,
            user_lng,
            item["latitude"],
            item["longitude"]
        )

        score = float(score)
        score = max(0, min(1, score))

        results.append({
            "part_name": item["part_name"],
            "image": item["image"],
            "shop_name": item["shop_name"],
            "distance": dist,
            "score": score,
            "map": item["google_map_link"]
        })

    # keyword filter
    if query:

        keyword = query.lower()

        filtered = [
            r for r in results
            if keyword in r["part_name"].lower()
        ]

        if len(filtered) > 0:
            results = filtered

    return results

# ---------------------------
# UI
# ---------------------------

st.title("🔧 FindSpares AI")

user_lat = 13.2839215
user_lng = 100.9289055

st.success("📍 Location detected")

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Search spare part")

with col2:
    upload = st.file_uploader("Upload image")

if "page" not in st.session_state:
    st.session_state.page = 1

per_page = 10

# ---------------------------
# SEARCH BUTTON
# ---------------------------

if st.button("AI Search"):

    st.session_state.page = 1

    if upload:

        img = Image.open(upload)
        query_vec = encode_image(img)
        query_text = None

    else:

        query_text = translate_keyword(query)
        query_vec = encode_text(query_text)

    results = search(query_vec, user_lat, user_lng, query_text)

    results = sorted(
        results,
        key=lambda x: (-x["score"], x["distance"])
    )

    st.session_state.results = results


# ---------------------------
# SHOW RESULT
# ---------------------------

if "results" in st.session_state:

    results = st.session_state.results
    page = st.session_state.page

    start = (page-1)*per_page
    end = start+per_page

    page_results = results[start:end]

    cols = st.columns(3)

    for i, r in enumerate(page_results):

        with cols[i % 3]:

            img_path = f"shop_parts/{r['image']}"

            if os.path.exists(img_path):
                st.image(img_path, width=220)
            else:
                st.image("https://via.placeholder.com/220x160")

            st.markdown(f"""
            **{r['part_name']}**

            🏪 {r['shop_name']}

            📍 {r['distance']:.2f} km

            ⭐ AI match {r['score']:.2f}

            [🗺 Open Map]({r['map']})
            """)

    total_pages = math.ceil(len(results)/per_page)

    if total_pages > 1:

        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            if st.button("⬅ Prev") and page > 1:
                st.session_state.page -= 1
                st.rerun()

        with col3:
            if st.button("Next ➡") and page < total_pages:
                st.session_state.page += 1
                st.rerun()

        st.write(f"Page {page} / {total_pages}")