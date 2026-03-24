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
# LOAD MODEL (Cached to save memory and startup time)
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

model, preprocess = load_clip_model()

# ---------------------------
# DATABASE SECRETS & CONNECTION
# ---------------------------

def get_db_connection():
    try:
        # Check if secrets are available
        if "DB_HOST" not in st.secrets:
            st.error("❌ ไม่พบข้อมูลการเชื่อมต่อฐานข้อมูลใน Secrets! กรุณาเพิ่ม DB_HOST, DB_USER, DB_PASSWORD, DB_NAME ในหน้า Dashboard Settings")
            st.stop()
            
        return pymysql.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"],
            port=int(st.secrets.get("DB_PORT", 3306)),
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10  # ป้องกันแอปค้างหาก Firewall บล็อก
        )
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {e}")
        st.info("💡 คำแนะนำ: ตรวจสอบว่าโฮสต์ฐานข้อมูลพะยอมรับการเชื่อมต่อจากภายนอก (Streamlit Cloud) หรือไม่")
        st.stop()

# ---------------------------
# LOAD VECTORS (Cached to prevent OOM and redundant DB calls)
# ---------------------------

@st.cache_resource
def load_vectors_cached():
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
    conn.close()

    if not data:
        return None, []

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

# Initialize Index
with st.spinner("📦 กำลังโหลดข้อมูลอะไหล่..."):
    index, items = load_vectors_cached()

if index is None:
    st.warning("⚠️ ไม่พบข้อมูลเวกเตอร์ในฐานข้อมูล")
    st.stop()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT part_name 
    FROM part_synonyms 
    WHERE synonym LIKE %s
    """, ("%"+keyword+"%",))
    r = cursor.fetchone()
    conn.close()
    if r:
        return r["part_name"]
    return keyword

# ---------------------------
# ENCODE FUNCTIONS
# ---------------------------

def encode_text(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        vec = model.encode_text(text_tokens)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")

def encode_image(img):
    img_processed = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(img_processed)
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
        key = item["id"]
        if key in seen:
            continue
        seen.add(key)

        dist = distance(user_lat, user_lng, item["latitude"], item["longitude"])
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

    if query:
        keyword = query.lower()
        results = [r for r in results if keyword in r["part_name"].lower()]

    return results

# ---------------------------
# UI - FRONTEND
# ---------------------------

st.title("🔧 FindSpares AI")

user_lat = 13.2839215
user_lng = 100.9289055

st.info("📍 ระบบจำลองตำแหน่งผู้ใช้งาน (Chonburi, Thailand)")

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("ค้นหาอะไหล่ (ระบุคำค้นหา)")

with col2:
    upload = st.file_uploader("หรืออัปโหลดรูปภาพเพื่อค้นหา")

if "results" not in st.session_state:
    st.session_state.results = None

if "page" not in st.session_state:
    st.session_state.page = 1

per_page = 9

if st.button("🚀 ค้นหาด้วย AI"):
    st.session_state.page = 1
    with st.spinner("🔎 กำลังประมวลผลการค้นหา..."):
        if upload:
            img = Image.open(upload)
            query_vec = encode_image(img)
            query_text = None
        else:
            query_text = translate_keyword(query)
            query_vec = encode_text(query_text)

        results = search(query_vec, user_lat, user_lng, query_text)
        results = sorted(results, key=lambda x: (-x["score"], x["distance"]))
        st.session_state.results = results

# ---------------------------
# DISPLAY RESULTS
# ---------------------------

if st.session_state.results:
    results = st.session_state.results
    page = st.session_state.page
    total_pages = math.ceil(len(results) / per_page)
    
    start = (page - 1) * per_page
    end = start + per_page
    page_results = results[start:end]

    cols = st.columns(3)
    for i, r in enumerate(page_results):
        with cols[i % 3]:
            img_path = f"shop_parts/{r['image']}"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x200?text=No+Image", use_container_width=True)
            
            st.markdown(f"### {r['part_name']}")
            st.write(f"🏪 **ร้าน:** {r['shop_name']}")
            st.write(f"📍 **ระยะห่าง:** {r['distance']:.2f} กม.")
            st.write(f"⭐ **AI Match:** {r['score']:.2f}")
            st.markdown(f"[🗺️ ดูแผนที่ร้าน]({r['map']})")
            st.divider()

    # Pagination UI
    if total_pages > 1:
        p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
        with p_col1:
            if st.button("⬅️ ก่อนหน้า") and page > 1:
                st.session_state.page -= 1
                st.rerun()
        with p_col2:
            st.write(f"หน้า {page} จาก {total_pages}")
        with p_col3:
            if st.button("ถัดไป ➡️") and page < total_pages:
                st.session_state.page += 1
                st.rerun()