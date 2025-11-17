# animegan.py
import cv2
import numpy as np
import os, requests, base64

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")  # 可选：如果配置了 HF token，可以调用 HF 的模型推理

def local_anime_style(img):
    """轻量化二次元效果：保留色块和边缘，方便后续抽线"""
    # 使用边缘保护的双边滤波 + 色彩量化
    img_small = cv2.pyrDown(img)
    for _ in range(2):
        img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)
    img_up = cv2.pyrUp(img_small)
    # 色彩量化
    Z = img_up.reshape((-1,3))
    Z = np.float32(Z)
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result = res.reshape((img_up.shape))
    return result

def apply_anime(img):
    """对外接口：如果设置 HF_TOKEN，则可调用 HuggingFace 的轻量模型（可选）；否则用本地简化风格"""
    if HF_TOKEN:
        # 使用 HF Inference API 做二次元风（这里示例调用一个通用 model，你可以改为偏好的模型）
        # 注意：HF 推理输出格式不同，下面是一个简易实现（按需改）
        url = "https://api-inference.huggingface.co/models/akhaliq/animegan3-shinkai-512"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        data = {"inputs": b64}
        r = requests.post(url, headers=headers, json=data, timeout=60)
        if r.status_code == 200:
            out_b64 = r.json().get("image_base64") or r.json().get("data")
            if out_b64:
                out = base64.b64decode(out_b64)
                arr = np.frombuffer(out, dtype=np.uint8)
                img2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img2
    # fallback to local stylization
    try:
        return local_anime_style(img)
    except Exception:
        return img
