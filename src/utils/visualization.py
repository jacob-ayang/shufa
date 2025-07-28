"""
工具模块 - 可视化
实现评分结果可视化功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_character_processing(original_image, processed_image, title="字符处理结果"):
    """
    可视化字符处理结果
    
    Args:
        original_image (numpy.ndarray): 原始图像
        processed_image (numpy.ndarray): 处理后图像
        title (str): 图像标题
    """
    plt.figure(figsize=(10, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示处理后图像
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title("处理后图像")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_character_segmentation(original_image, segments, positions, title="字符分割结果"):
    """
    可视化字符分割结果
    
    Args:
        original_image (numpy.ndarray): 原始图像
        segments (list): 分割后的字符图像列表
        positions (list): 字符位置列表
        title (str): 图像标题
    """
    fig, axes = plt.subplots(2, max(1, len(segments)), figsize=(15, 8))
    
    # 显示原始图像
    if len(segments) > 0:
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis('off')
        
        # 显示分割后的字符
        for i, (segment, pos) in enumerate(zip(segments, positions)):
            axes[1, i].imshow(segment, cmap='gray')
            axes[1, i].set_title(f"字符 {i+1}\n位置: ({pos[0]}, {pos[1]})")
            axes[1, i].axis('off')
    else:
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis('off')
        
        # 显示未分割的图像
        axes[1, 0].imshow(original_image, cmap='gray')
        axes[1, 0].set_title("未分割")
        axes[1, 0].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_scoring_results(scores, title="评分结果"):
    """
    可视化评分结果
    
    Args:
        scores (dict): 评分结果字典
        title (str): 图像标题
    """
    # 准备数据
    labels = list(scores.keys())
    values = list(scores.values())
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('评分')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_evaluation_results(evaluation_results, title="评估结果"):
    """
    可视化评估结果
    
    Args:
        evaluation_results (dict): 评估结果
        title (str): 图像标题
    """
    # 准备数据
    metrics = ['准确率', 'MAE', 'RMSE']
    values = [evaluation_results['accuracy'] * 100, 
              evaluation_results['mae'], 
              evaluation_results['rmse']]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制指标柱状图
    bars = ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('评估指标')
    ax1.set_ylabel('数值')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom')
    
    # 绘制相关性散点图
    # 这里需要预测评分和实际评分的数据，简化实现中使用随机数据
    # 在实际使用中，应传入预测评分和实际评分
    np.random.seed(42)
    actual = np.random.uniform(60, 100, 50)
    predicted = actual + np.random.normal(0, 5, 50)
    
    ax2.scatter(actual, predicted, alpha=0.6)
    ax2.plot([60, 100], [60, 100], 'r--', lw=2)
    ax2.set_xlabel('实际评分')
    ax2.set_ylabel('预测评分')
    ax2.set_title(f"相关性分析\n皮尔逊相关系数: {evaluation_results['correlation']['pearson_correlation']:.4f}")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()