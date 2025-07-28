import numpy as np
from src.preprocessing.image_preprocessing import preprocess_image
from src.segmentation.character_segmentation import segment_characters
from src.recognition.character_recognition import create_cirm_model

# 硬笔书法评分系统训练示例
# 演示如何训练字符识别模型

def load_and_prepare_data():
    """
    加载并准备数据用于训练
    """
    print("正在加载数据...")
    # 这里应该加载实际的书法数据集
    # 为了演示目的，我们使用随机数据
    # 实际应用中应该从data/processed/目录加载处理后的数据
    
    # 模拟加载数据
    X = np.random.rand(1000, 64, 64)  # 1000个64x64的字符图像
    y = np.random.randint(0, 100, 1000)  # 1000个随机标签
    
    return X, y

def train_character_recognition_model(X_train, y_train):
    """
    训练字符识别模型
    
    Args:
        X_train (numpy.ndarray): 训练数据
        y_train (numpy.ndarray): 训练标签
        
    Returns:
        model: 训练好的模型
    """
    # 创建CIRM模型
    model = create_cirm_model(input_shape=(64, 64, 1), num_classes=100)
    
    # 准备数据
    X_train = X_train.reshape(-1, 64, 64, 1).astype('float32') / 255.0
    y_train_categorical = np.eye(100)[y_train]  # 转换为one-hot编码
    
    # 训练模型
    print("正在训练字符识别模型...")
    model.fit(X_train, y_train_categorical, epochs=5, batch_size=32, verbose=1)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test (numpy.ndarray): 测试数据
        y_test (numpy.ndarray): 测试标签
        
    Returns:
        float: 准确率
    """
    # 准备测试数据
    X_test = X_test.reshape(-1, 64, 64, 1).astype('float32') / 255.0
    y_test_categorical = np.eye(100)[y_test]  # 转换为one-hot编码
    
    # 评估模型
    print("正在评估模型性能...")
    loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"测试准确率: {accuracy:.4f}")
    
    return accuracy

def main():
    """
    主函数
    """
    # 加载数据
    X, y = load_and_prepare_data()
    
    # 分割训练集和测试集
    # 为了演示目的，我们使用前800个样本作为训练集，后200个作为测试集
    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:1000]
    y_test = y[800:1000]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练字符识别模型
    print("\n正在训练字符识别模型...")
    model = train_character_recognition_model(X_train, y_train)
    
    # 评估模型性能
    print("\n正在评估模型性能...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()