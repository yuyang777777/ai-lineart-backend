# lineart.py
import cv2
import numpy as np

def lineart(img):
    """生成黑白线稿（适合漫画线稿的拔线版）"""
    # 输入 BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 降噪
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # 边缘检测（Canny + 膨胀 + 二值化）
    edges = cv2.Canny(blur, 60, 160)
    # 膨胀使线条更连贯
    kernel = np.ones((2,2), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    # 反色，线为黑底白背景 -> 我们想要黑线白底
    line = 255 - dil
    # 保证单通道 uint8
    line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    return line
