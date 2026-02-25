import numpy as np
import cv2
import insightface
from fastapi import FastAPI, UploadFile, File
import pymysql
import faiss

app = FastAPI()

# 初始化模型
model = insightface.app.FaceAnalysis(
    name='buffalo_l',
    root='/root/.insightface'
)
model.prepare(ctx_id=0)  # CPU

# 数据库配置
DB_CONFIG = {
    "host": "192.168.3.241",
    "user": "root",
    "password": "123456",
    "database": "testwm",
    "port": 3306
}

def get_all_embeddings():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id,person_name, face_embedding FROM t_person WHERE face_embedding IS NOT NULL")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    ids = []
    vectors = []

    for row in rows:
	    # ids放入name
        ids.append(row[1])
        vectors.append(np.frombuffer(row[2], dtype=np.float32))

    return ids, np.array(vectors).astype('float32')

@app.post("/extract")
async def extract_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = model.get(img)

    if len(faces) != 1:
        return {"success": False, "msg": "必须且只能检测一张人脸"}

    embedding = faces[0].embedding.astype('float32')
    return {
        "success": True,
        "embedding": embedding.tolist()
    }

@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = model.get(img)

    if len(faces) != 1:
        return {"success": False, "msg": "必须且只能检测一张人脸"}

    query_embedding = faces[0].embedding.astype('float32')

    ids, vectors = get_all_embeddings()

    if len(vectors) == 0:
        return {"success": False, "msg": "数据库没数据"}

    index = faiss.IndexFlatIP(512)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    scores, result_ids = index.search(query_embedding, 1)

    best_score = float(scores[0][0])
    best_id = ids[result_ids[0][0]]

    return {
        "success": True,
        "person_id": best_id,
        "score": best_score
    }