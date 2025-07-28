import struct
import numpy as np
import re
import os

class MPFParser:
    """
    MPF文件解析器
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.header_size = 0
        self.format_code = ""
        self.illustration_text = ""
        self.code_type = ""
        self.code_length = 0
        self.data_type = ""
        self.sample_number = 0
        self.dimensionality = 0
        self.samples = []
    
    def parse_header(self, f):
        """
        解析MPF文件头
        """
        # 读取文件头大小（4字节，小端序）
        header_size_data = f.read(4)
        if len(header_size_data) < 4:
            raise ValueError("无法读取文件头大小")
        self.header_size = struct.unpack('<i', header_size_data)[0]
        print(f"DEBUG: 文件头大小: {self.header_size}")
        
        # 读取格式代码（8字节）
        format_code_data = f.read(8)
        self.format_code = format_code_data.decode('ascii', errors='ignore').rstrip('\x00')
        print(f"DEBUG: 格式代码: {self.format_code}")
        
        # 读取说明文本
        # 说明文本大小 = 文件头总大小 - (文件头大小字段4字节 + 格式代码字段8字节 + 代码类型20字节 + 代码长度2字节 + 数据类型20字节 + 样本数量4字节 + 维度4字节)
        illustration_text_size = self.header_size - 4 - 8 - 20 - 2 - 20 - 4 - 4
        illustration_text_data = f.read(illustration_text_size)
        self.illustration_text = illustration_text_data.decode('ascii', errors='ignore').rstrip('\x00')
        print(f"DEBUG: 说明文本: {self.illustration_text}")
        
        # 读取代码类型（20字节）
        code_type_data = f.read(20)
        self.code_type = code_type_data.decode('ascii', errors='ignore').rstrip('\x00')
        print(f"DEBUG: 代码类型: {self.code_type}")
        
        # 读取代码长度（2字节，小端序）
        code_length_data = f.read(2)
        self.code_length = struct.unpack('<h', code_length_data)[0]
        print(f"DEBUG: 代码长度: {self.code_length}")
        
        # 读取数据类型（20字节）
        data_type_data = f.read(20)
        self.data_type = data_type_data.decode('ascii', errors='ignore').rstrip('\x00')
        # 如果无法从固定位置正确读取，则尝试从说明文本中提取
        if not self.data_type or len(self.data_type) == 0:
            data_type_match = re.search(r'#ftrtype=(\w+)', self.illustration_text)
            if data_type_match:
                extracted_type = data_type_match.group(1)
                if extracted_type == 'ncg':
                    self.data_type = 'unsigned char'
                elif extracted_type == 'fcg':
                    self.data_type = 'float'
                elif extracted_type == 'scg':
                    self.data_type = 'short'
                else:
                    self.data_type = 'unsigned char'  # 默认值
            else:
                self.data_type = 'unsigned char'  # 默认值
        print(f"DEBUG: 数据类型: {self.data_type}")
        
        # 读取样本数量（4字节，小端序）
        sample_count_data = f.read(4)
        self.sample_number = struct.unpack('<i', sample_count_data)[0]
        print(f"DEBUG: 样本数量: {self.sample_number}")
        
        # 读取维度（4字节，小端序）
        dimension_data = f.read(4)
        self.dimensionality = struct.unpack('<i', dimension_data)[0]
        print(f"DEBUG: 维度: {self.dimensionality}")
    
    def get_data_type_info(self):
        """
        获取数据类型信息
        """
        if self.data_type == 'unsigned char':
            return np.uint8, 1
        elif self.data_type == 'float':
            return np.float32, 4
        elif self.data_type == 'short':
            return np.int16, 2
        else:
            raise ValueError(f"不支持的数据类型: {self.data_type}")
    
    def parse_samples(self, f):
        """
        解析样本记录
        """
        # 获取数据类型信息
        dtype, element_size = self.get_data_type_info()
        
        # 计算每个样本的大小
        sample_size = self.code_length + self.dimensionality * element_size
        
        # 读取所有样本
        self.samples = []
        for i in range(self.sample_number):
            # 读取标签
            label_data = f.read(self.code_length)
            if not label_data or len(label_data) < self.code_length:
                print(f"警告: 在读取第{i+1}个样本时文件结束")
                break
            
            # 读取向量数据
            vector_data = f.read(self.dimensionality * element_size)
            if not vector_data or len(vector_data) < self.dimensionality * element_size:
                print(f"警告: 在读取第{i+1}个样本的向量数据时文件结束")
                break
            
            # 解析向量数据
            vector = np.frombuffer(vector_data, dtype=dtype)
            
            self.samples.append({
                'label': label_data,
                'vector': vector
            })
    
    def parse(self):
        """
        解析MPF文件
        """
        with open(self.file_path, 'rb') as f:
            # 解析文件头
            self.parse_header(f)
            
            # 解析样本记录
            self.parse_samples(f)
        
        return self
    
    def get_header_info(self):
        """
        获取文件头信息
        """
        return {
            'header_size': self.header_size,
            'format_code': self.format_code,
            'illustration_text': self.illustration_text,
            'code_type': self.code_type,
            'code_length': self.code_length,
            'data_type': self.data_type,
            'sample_number': self.sample_number,
            'dimensionality': self.dimensionality
        }
    
    def get_samples(self):
        """
        获取解析后的样本数据
        """
        return self.samples
    
    def get_sample_count(self):
        """
        获取样本数量
        """
        return len(self.samples)
    
    def get_sample(self, index):
        """
        获取指定索引的样本
        """
        if 0 <= index < len(self.samples):
            return self.samples[index]
        else:
            raise IndexError("样本索引超出范围")

def parse_mpf_file(file_path):
    """
    解析MPF文件
    :param file_path: MPF文件路径
    :return: 解析后的数据
    """
    parser = MPFParser(file_path)
    parser.parse()
    return parser.get_samples()

def main():
    # 创建解析器实例
    parser = MPFParser('data/HWDB1.0trn/HWDB1.0trn/001.mpf')
    
    # 解析文件
    parser.parse()
    
    # 显示文件头信息
    header_info = parser.get_header_info()
    print("MPF文件头信息:")
    print(f"  文件头大小: {header_info['header_size']}")
    print(f"  格式代码: {header_info['format_code']}")
    # 分行显示说明文本，每行不超过80字符
    illustration_lines = [header_info['illustration_text'][i:i+80] for i in range(0, len(header_info['illustration_text']), 80)]
    for line in illustration_lines:
        print(f"  说明文本: {line}")
    print(f"  代码类型: {header_info['code_type']}")
    print(f"  代码长度: {header_info['code_length']}")
    print(f"  数据类型: {header_info['data_type']}")
    print(f"  样本数量: {header_info['sample_number']}")
    print(f"  维度: {header_info['dimensionality']}")
    
    # 显示前5个样本的信息
    samples = parser.get_samples()
    print(f"\n成功解析 {len(samples)} 个样本")
    
    for i in range(min(5, len(samples))):
        sample = samples[i]
        label = sample['label']
        vector = sample['vector']
        print(f"\n第{i+1}个样本:")
        print(f"  标签: {label}")
        print(f"  向量形状: {vector.shape}")
        print(f"  向量前10个元素: {vector[:10]}")
    
    # 保存解析结果到NumPy文件
    import numpy as np
    
    # 提取所有样本的向量数据
    vectors = np.array([sample['vector'] for sample in samples])
    
    # 保存到文件
    np.save('parsed_data.npy', vectors)
    print(f"\n已将向量数据保存到 'parsed_data.npy' 文件中，形状为: {vectors.shape}")

if __name__ == "__main__":
    main()