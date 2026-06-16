import os

def convert_to_clean_utf8(filepath):
    """
    三阶段强力转换函数：
    1. 尝试正常 UTF-8 读取（若有旧的 gbk 声明则清除并重写）
    2. 尝试 GBK 读取并转换
    3. 终极兜底：强行以二进制读取，忽略掉所有无法识别的乱码字节，强转为纯净 UTF-8
    """
    try:
        # 阶段 1：尝试标准 UTF-8 读取
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 顺便检查并清理掉之前可能误加的 gbk 编码声明
        if '# -*- coding: gbk -*-' in content:
            content = content.replace('# -*- coding: gbk -*-', '')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Normalized (Removed GBK header)"
        return False, "Already Clean UTF-8"
        
    except UnicodeDecodeError:
        pass

    try:
        # 阶段 2：尝试用 GBK 读取并转存为 UTF-8
        with open(filepath, 'r', encoding='gbk') as f:
            content = f.read()
        if '# -*- coding: gbk -*-' in content:
            content = content.replace('# -*- coding: gbk -*-', '')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Converted from GBK to UTF-8"
    except UnicodeDecodeError:
        pass

    try:
        # 阶段 3：终极兜底（处理包含混杂乱码、非法字节的文件）
        # 以二进制模式读取，强行用 utf-8 解码并无视非法字节
        with open(filepath, 'rb') as f:
            raw_data = f.read()
        
        content = raw_data.decode('utf-8', errors='ignore')
        if '# -*- coding: gbk -*-' in content:
            content = content.replace('# -*- coding: gbk -*-', '')
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Forcefully cleaned up invalid bytes to UTF-8"
    except Exception as e:
        return False, f"Error: {e}"

if __name__ == "__main__":
    print("🚀 开始全盘扫描代码文件，自动修复并统一为 UTF-8 编码...")
    
    # 定义需要扫描的后缀名
    valid_extensions = ('.py', '.sh', '.md', '.txt')
    # 定义需要绝对跳过的文件夹名称
    skip_dirs = {'.git', '__pycache__', 'wandb', '.idea', '.vscode', 'build', 'dist'}
    
    files_to_check = []
    
    # 遍历当前目录及所有子目录
    for root, dirs, files in os.walk('.'):
        # 优化：通过修改 dirs 切片，让 os.walk 彻底不进入这些大文件夹，大幅提升扫描速度
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(valid_extensions):
                files_to_check.append(os.path.join(root, file))
                
    converted_count = 0
    print(f"📊 找到待检查文件共计: {len(files_to_check)} 个\n" + "-"*50)
    
    for filepath in files_to_check:
        success, msg = convert_to_clean_utf8(filepath)
        if success:
            print(f"✅ [{msg}] -> {filepath}")
            converted_count += 1
            
    print("-"*50)
    print(f"🎉 扫描全面完成！本次共强力修复/转换了 {converted_count} 个文件的编码。")