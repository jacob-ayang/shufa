"""
图像预处理模块
实现论文中提到的图像预处理步骤：灰度化、二值化、去噪、骨架提取等
"""

import cv2
import numpy as np
from scipy import ndimage


def grayscale_conversion(image):
    """
    将彩色图像转换为灰度图像
    
    Args:
        image (numpy.ndarray): 输入的彩色图像
        
    Returns:
        numpy.ndarray: 灰度图像
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarization(image, method='otsu'):
    """
    图像二值化处理
    
    Args:
        image (numpy.ndarray): 输入的灰度图像
        method (str): 二值化方法 ('otsu', 'adaptive', 'global')
        
    Returns:
        numpy.ndarray: 二值化后的图像
    """
    if method == 'otsu':
        # Otsu二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # 自适应二值化
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    else:
        # 全局阈值二值化
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    return binary


def denoising(image, kernel_size=3):
    """
    图像去噪处理
    
    Args:
        image (numpy.ndarray): 输入的二值化图像
        kernel_size (int): 去噪核大小
        
    Returns:
        numpy.ndarray: 去噪后的图像
    """
    # 中值滤波去噪
    denoised = cv2.medianBlur(image, kernel_size)
    return denoised


def skeleton_extraction(image):
    """
    骨架提取
    使用距离变换和局部最大值法提取图像骨架
    
    Args:
        image (numpy.ndarray): 输入的二值化图像
        
    Returns:
        numpy.ndarray: 骨架图像
    """
    # 距离变换
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    
    # 归一化距离变换结果
    normalized_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 局部最大值法提取骨架
    local_max = ndimage.maximum_filter(normalized_dist, size=3)
    skeleton = (normalized_dist == local_max) & (normalized_dist > 0)
    
    return skeleton.astype(np.uint8) * 255


def remove_burr(skeleton):
    """
    去毛刺处理
    
    Args:
        skeleton (numpy.ndarray): 输入的骨架图像
        
    Returns:
        numpy.ndarray: 去毛刺后的骨架图像
    """
    # 形态学操作去毛刺
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
    return cleaned


def preprocess_image(image, denoise_kernel=3):
    """
    完整的图像预处理流程
    
    Args:
        image (numpy.ndarray): 输入的原始图像
        denoise_kernel (int): 去噪核大小
        
    Returns:
        dict: 包含各个处理步骤结果的字典
    """
    results = {}
    
    # 1. 灰度化
    gray = grayscale_conversion(image)
    results['gray'] = gray
    
    # 2. 二值化
    binary = binarization(gray)
    results['binary'] = binary
    
    # 3. 去噪
    denoised = denoising(binary, denoise_kernel)
    results['denoised'] = denoised
    
    # 4. 骨架提取
    skeleton = skeleton_extraction(denoised)
    results['skeleton'] = skeleton
    
    # 5. 去毛刺
    cleaned_skeleton = remove_burr(skeleton)
    results['cleaned_skeleton'] = cleaned_skeleton
    
    return results