import cv2
import numpy as np
from PIL import Image
import os


def binarize_image(image):
    """
    图像二值化
    :param image: 输入图像
    :return: 二值化图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Otsu二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def denoise_image(image):
    """
    图像去噪
    :param image: 输入图像
    :return: 去噪后的图像
    """
    # 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return denoised


def extract_skeleton(image):
    """
    提取图像骨架
    :param image: 输入图像
    :return: 骨架图像
    ""
    # 使用形态学操作提取骨架
    skeleton = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, opening)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = opening.copy()
        
        if cv2.countNonZero(image) == 0:
            break
    
    return skeleton


def segment_characters(image):
    """
    字符分割
    :param image: 输入图像
    :return: 分割后的字符图像列表
    """
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取字符图像
    characters = []
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小的区域
        if w * h > 100:
            # 提取字符区域
            char_img = image[y:y+h, x:x+w]
            characters.append(char_img)
    
    return characters


def normalize_character(image, size=(64, 64)):
    """
    标准化字符图像大小
    :param image: 输入字符图像
    :param size: 目标大小
    :return: 标准化后的图像
    """
    # 调整图像大小
    normalized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    return normalized


def preprocess_pipeline(image_path):
    """
    完整的预处理流水线
    :param image_path: 图像路径
    :return: 预处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 二值化
    binary = binarize_image(image)
    
    # 去噪
    denoised = denoise_image(binary)
    
    # 提取骨架
    skeleton = extract_skeleton(denoised)
    
    # 分割字符
    characters = segment_characters(skeleton)
    
    # 标准化字符
    normalized_chars = [normalize_character(char) for char in characters]
    
    return normalized_chars


def test_preprocess():
    """测试预处理函数"""
    print("预处理工具测试")
    print("注意：需要实际图像文件来运行完整测试")


if __name__ == "__main__":
    test_preprocess()