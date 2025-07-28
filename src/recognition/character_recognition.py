"""
单字识别模块
实现论文中提到的CIRM框架和ResNet50特征提取
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity


class CIRMSNN:
    """
    基于孪生神经网络的单字识别模型 (CIRM-SNN)
    """
    
    def __init__(self, input_shape=(64, 64, 1), embedding_dim=128):
        """
        初始化CIRM-SNN模型
        
        Args:
            input_shape (tuple): 输入图像形状
            embedding_dim (int): 嵌入向量维度
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """
        构建孪生神经网络模型
        
        Returns:
            tensorflow.keras.Model: 构建的模型
        """
        # 使用ResNet50作为特征提取器
        base_model = ResNet50(weights=None, include_top=False, input_shape=self.input_shape)
        
        # 添加全局平均池化层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # 添加全连接层作为嵌入层
        embeddings = Dense(self.embedding_dim, activation='relu')(x)
        
        # 创建模型
        model = Model(inputs=base_model.input, outputs=embeddings)
        return model
    
    def extract_features(self, images):
        """
        提取图像特征
        
        Args:
            images (numpy.ndarray): 输入图像数组
            
        Returns:
            numpy.ndarray: 特征向量
        """
        # 确保输入图像有正确的形状
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # 归一化图像
        images = images.astype(np.float32) / 255.0
        
        # 提取特征
        features = self.model.predict(images)
        return features
    
    def calculate_similarity(self, feature1, feature2):
        """
        计算两个特征向量之间的相似度
        
        Args:
            feature1 (numpy.ndarray): 第一个特征向量
            feature2 (numpy.ndarray): 第二个特征向量
            
        Returns:
            float: 余弦相似度
        """
        # 计算余弦相似度
        similarity = cosine_similarity([feature1], [feature2])[0][0]
        return similarity
    
    def recognize_character(self, input_image, template_images, template_labels):
        """
        识别输入字符
        
        Args:
            input_image (numpy.ndarray): 输入字符图像
            template_images (list): 模板字符图像列表
            template_labels (list): 模板字符标签列表
            
        Returns:
            tuple: (预测标签, 相似度)
        """
        # 提取输入图像特征
        input_features = self.extract_features(input_image)
        
        # 提取模板图像特征
        template_features = self.extract_features(np.array(template_images))
        
        # 计算相似度
        similarities = []
        for template_feature in template_features:
            similarity = self.calculate_similarity(input_features, template_feature)
            similarities.append(similarity)
        
        # 找到最相似的模板
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        predicted_label = template_labels[best_match_idx]
        
        return predicted_label, best_similarity
    
    def compile_model(self, optimizer='adam', loss='mse'):
        """
        编译模型
        
        Args:
            optimizer (str): 优化器
            loss (str): 损失函数
        """
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """
        训练模型
        
        Args:
            x_train (numpy.ndarray): 训练数据
            y_train (numpy.ndarray): 训练标签
            epochs (int): 训练轮数
            batch_size (int): 批次大小
        """
        # 归一化图像
        x_train = x_train.astype(np.float32) / 255.0
        
        # 如果输入是单通道图像，需要转换为三通道
        if x_train.shape[-1] == 1:
            x_train = np.repeat(x_train, 3, axis=-1)
        
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


def load_template_dataset(template_data_path, template_labels_path):
    """
    加载模板数据集
    
    Args:
        template_data_path (str): 模板数据文件路径
        template_labels_path (str): 模板标签文件路径
        
    Returns:
        tuple: (模板图像, 模板标签)
    """
    template_images = np.load(template_data_path)
    template_labels = np.load(template_labels_path)
    return template_images, template_labels


def recognize_characters(segmented_chars, template_data_path, template_labels_path):
    """
    识别分割后的字符
    
    Args:
        segmented_chars (list): 分割后的字符图像列表
        template_data_path (str): 模板数据文件路径
        template_labels_path (str): 模板标签文件路径
        
    Returns:
        list: 识别结果列表
    """
    # 加载模板数据集
    template_images, template_labels = load_template_dataset(template_data_path, template_labels_path)
    
    # 初始化识别模型
    recognizer = CIRMSNN()
    
    # 识别每个字符
    results = []
    for char_image in segmented_chars:
        predicted_label, similarity = recognizer.recognize_character(
            char_image, template_images, template_labels)
        results.append({
            'predicted_label': predicted_label,
            'similarity': similarity
        })
    
    return results