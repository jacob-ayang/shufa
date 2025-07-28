import os

def show_directory_tree(startpath, max_depth=2):
    """
    显示目录树结构
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level <= max_depth:
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            if level < max_depth:
                for file in files:
                    print(f'{subindent}{file}')

def main():
    """
    主函数
    """
    print("项目结构:")
    print("=")
    show_directory_tree('.', max_depth=2)
    
    print("\n数据目录结构:")
    print("=")
    show_directory_tree('./data', max_depth=2)

if __name__ == "__main__":
    main()