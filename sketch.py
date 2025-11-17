# sketch.py
import cv2
import numpy as np

def pencil(img):
    """铅笔素描风（基于色彩反转 + 高斯模糊）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch_color = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_color
