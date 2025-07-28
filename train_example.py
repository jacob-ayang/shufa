import numpy as np

# 注意：这个示例仅用于演示目的
# 实际的汉字识别任务需要更复杂的预处理和模型

def load_and_prepare_data():
    """
    加载并准备数据用于训练
    """
    print("正在加载数据...")
    vectors = np.load('all_parsed_data.npy')
    labels = np.load('all_parsed_data_labels.npy', allow_pickle=True)
    
    # 为了演示目的，我们只使用前1000个样本
    # 实际训练应该使用完整数据集
    X = vectors[:1000]
    y = labels[:1000]
    
    # 简单的标签编码（实际应用中需要更复杂的编码方式）
    # 这里我们只使用标签的哈希值作为类别标识
    y_encoded = [hash(label) % 100 for label in y]  # 简化的编码方式
    
    return X, y_encoded

def simple_knn_predict(X_train, y_train, X_test, k=3):
    """
    简单的K近邻预测实现
    """
    predictions = []
    for test_sample in X_test:
        # 计算与所有训练样本的距离
        distances = np.sqrt(np.sum((X_train - test_sample) ** 2, axis=1))
        
        # 找到最近的k个邻居
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_labels = [y_train[i] for i in nearest_indices]
        
        # 投票决定预测标签
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        predictions.append(prediction)
    
    return predictions

def evaluate_predictions(y_true, y_pred):
    """
    评估预测结果
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    print(f"准确率: {accuracy:.4f}")
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
    
    # 为了演示目的，我们只使用前50个训练样本和前10个测试样本
    # 因为完整的KNN计算会很慢
    X_train_small = X_train[:50]
    y_train_small = y_train[:50]
    X_test_small = X_test[:10]
    y_test_small = y_test[:10]
    
    print(f"\n简化训练集大小: {X_train_small.shape}")
    print(f"简化测试集大小: {X_test_small.shape}")
    
    # 使用简单的KNN进行预测
    print("\n正在使用K近邻算法进行预测...")
    predictions = simple_knn_predict(X_train_small, y_train_small, X_test_small, k=3)
    
    # 评估预测结果
    print("\n正在评估预测结果...")
    accuracy = evaluate_predictions(y_test_small, predictions)
    
    print("\n预测结果示例:")
    for i in range(min(5, len(predictions))):
        print(f"  真实标签: {y_test_small[i]}, 预测标签: {predictions[i]}")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()