import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'raw_data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, 'notebooks')

# 数据集路径
HWDB1_0_TRAIN = os.path.join(DATA_DIR, 'HWDB1.0trn')
HWDB1_0_TEST = os.path.join(DATA_DIR, 'HWDB1.0tst')
HWDB1_1_TRAIN = os.path.join(DATA_DIR, 'HWDB1.1trn')
HWDB1_1_TEST = os.path.join(DATA_DIR, 'HWDB1.1tst')
OLHWDB1_0_TRAIN = os.path.join(DATA_DIR, 'OLHWDB1.0trn')
OLHWDB1_0_TEST = os.path.join(DATA_DIR, 'OLHWDB1.0tst')
Pot1_2_TRAIN = os.path.join(DATA_DIR, 'Pot1.2Train')
Pot1_2_TEST = os.path.join(DATA_DIR, 'Pot1.2Test')

# 模型参数
NUM_CLASSES = 3755  # HWDB1.1字符类数
IMAGE_SIZE = (64, 64)  # 输入图像大小
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# 设备配置
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 艺术性模型参数
FONT_FEATURE_DIM = 128
EMOTION_FEATURE_DIM = 128
STYLE_FEATURE_DIM = 768  # BERT输出维度
HIDDEN_SIZE = 256

# 预处理参数
THRESHOLD_METHOD = 'otsu'  # 二值化方法