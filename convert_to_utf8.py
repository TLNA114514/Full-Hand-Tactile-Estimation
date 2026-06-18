import os

def sanitize_file_content(filepath):
    """
    用安全且不丢失字符的方式读取文件，清洗缩进杀手（\xa0 和 \t），并规范化为 UTF-8。
    """
    try:
        # 1. 以二进制模式读取原始字节，防止直接触发 UnicodeDecodeError 崩溃
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
        
        if not raw_bytes:
            return False, "Empty file"

        # 2. 尝试用 utf-8-sig (处理可能带 BOM 的 UTF-8) 解码
        # 如果失败，则退回到 latin1。latin1 可以无损地映射任意字节（0-255），保证绝对不会丢失任何代码字符
        try:
            content = raw_bytes.decode('utf-8-sig')
            encoding_used = "UTF-8"
        except UnicodeDecodeError:
            content = raw_bytes.decode('latin1')
            encoding_used = "Latin1/GBK Fallback"

        original_content = content

        # 3. 精准替换“缩进杀手”
        # \xa0 是网页或聊天软件复制时最常混入的“不换行空格”（Non-breaking space）
        if '\xa0' in content:
            content = content.replace('\xa0', ' ')
        
        # 4. 统一将硬 Tab 键转换为 4 个标准半角空格
        if '\t' in content:
            content = content.replace('\t', '    ')

        # 5. 如果文件被污染了或者原本不是标准 UTF-8，则将其强制重写为纯净的 UTF-8
        if content != original_content or encoding_used != "UTF-8":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fixed spacing ({encoding_used})"
        
        return False, "Already Clean UTF-8"

    except Exception as e:
        return False, f"Error: {str(e)}"


if __name__ == "__main__":
    print("🚀 开始全盘扫描代码文件，自动拦截并修复隐藏乱码与缩进错误...")
    
    # 定义需要扫描的文本/代码文件后缀
    valid_extensions = ('.py', '.sh', '.md', '.txt')
    # 严格跳过不需要扫描的底层或大资产文件夹，大幅度提升扫描效率
    skip_dirs = {'.git', '__pycache__', 'wandb', '.idea', '.vscode', 'build', 'dist', 'data'}
    
    files_to_check = []
    
    # 递归遍历当前目录
    for root, dirs, files in os.walk('.'):
        # 优化：直接修改 dirs 切片，让 os.walk 彻底不进入这些无需检查的文件夹
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(valid_extensions):
                files_to_check.append(os.path.join(root, file))
                
    print(f"📂 找到待检查文本文件共计: {len(files_to_check)} 个")
    print("-" * 60)
    
    fixed_count = 0
    for filepath in files_to_check:
        success, msg = sanitize_file_content(filepath)
        if success:
            print(f"✨ [已修复] -> {filepath} ({msg})")
            fixed_count += 1
            
    print("-" * 60)
    print(f"🎉 扫描全面完成！本次共强力修复/规范化了 {fixed_count} 个文件的编码与缩进。")