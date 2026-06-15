import os
import glob
import codecs

def convert_gbk_to_utf8(filepath):
    # 先尝试用 utf-8 读取，如果成功说明本来就是 utf-8
    try:
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            f.read()
        return False, "Already UTF-8"
    except UnicodeDecodeError:
        pass

    # 尝试用 gbk 读取
    try:
        with codecs.open(filepath, 'r', encoding='gbk') as f:
            content = f.read()
            
        # 写回 utf-8
        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Converted from GBK to UTF-8"
    except UnicodeDecodeError:
        return False, "Not UTF-8 and Not GBK, ignored"
    except Exception as e:
        return False, f"Error: {e}"

if __name__ == "__main__":
    print("开始扫描代码文件，自动将 GBK 转换为 UTF-8...")
    
    extensions = ['*.py', '*.sh', '*.md', '*.txt']
    files_to_check = []
    
    # 遍历当前目录及所有子目录
    for root, _, files in os.walk('.'):
        if '.git' in root or '__pycache__' in root or 'wandb' in root:
            continue
        for ext in extensions:
            for file in glob.glob(os.path.join(root, ext)):
                files_to_check.append(file)
                
    converted_count = 0
    for filepath in files_to_check:
        success, msg = convert_gbk_to_utf8(filepath)
        if success:
            print(f"✅ [已转换] {filepath}")
            converted_count += 1
            
    print(f"\n🎉 扫描完成！共转换了 {converted_count} 个文件。")
