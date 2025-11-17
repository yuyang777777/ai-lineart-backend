# server.py
# FastAPI 多风格图像处理后端（支持 lineart / comic / sketch / anime）
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, base64, cv2, numpy as np, os, io
from pydantic import BaseModel
from typing import Optional
from lineart import lineart
from sketch import pencil
from comic import comic_effect
from animegan import apply_anime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产请限定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ProcessReq(BaseModel):
    image: str  # base64 without data:prefix
    style: Optional[str] = "lineart"

def b64_to_cv2_img(b64: str):
    b = base64.b64decode(b64)
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_img_to_b64(img) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode()

@app.post("/process")
async def process(req: ProcessReq):
    try:
        img = b64_to_cv2_img(req.image)
    except Exception as e:
        return {"error": "invalid image", "detail": str(e)}

    style = req.style or "lineart"
    style = style.lower()

    if style == "lineart":
        out = lineart(img)
    elif style == "comic":
        out = comic_effect(img)
    elif style == "sketch":
        out = pencil(img)
    elif style == "anime":
        # anime 走轻量化路径：如果配置了 HF_TOKEN 则可扩展调用 HF API（可选）
        out = apply_anime(img)
    else:
        out = lineart(img)

    b64 = cv2_img_to_b64(out)
    return {"image": b64}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
