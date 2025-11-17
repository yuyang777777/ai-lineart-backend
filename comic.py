# comic.py
import cv2
import numpy as np

def comic_effect(img):
    """漫画化：保留强线 + 局部二值化来加强漫画感"""
    # 先做边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
    # 细化一些
    kernel = np.ones((1,1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_color
