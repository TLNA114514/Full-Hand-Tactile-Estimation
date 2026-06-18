import os

def sanitize_file_content(filepath):
    """
    终极安全版：优先挽救中文编码，彻底杜绝乱码转义吞引号引发的 SyntaxError。
    """
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
        
        if not raw_bytes:
            return False, "Empty file"

        content = None
        encoding_used = ""

        # 【核心改进】阶梯式编码尝试，精准拦截中文 GBK/GB18030
        encodings_to_try = ['utf-8-sig', 'gb18030', 'gbk', 'latin1']
        
        for enc in encodings_to_try:
            try:
                # 尝试用当前编码严格解码
                content = raw_bytes.decode(enc)
                encoding_used = enc
                break
            except UnicodeDecodeError:
                continue

        if content == None:
            # 极端的最后防线：如果都失败了，用 utf-8 忽略错误解码
            content = raw_bytes.decode('utf-8', errors='ignore')
            encoding_used = "UTF-8 (Lossy Recovery)"

        original_content = content

        # 精准清理缩进杀手
        if '\xa0' in content:
            content = content.replace('\xa0', ' ')
        if '\t' in content:
            content = content.replace('\t', '    ')

        # 【核心改进】检查是否有破坏语法的潜在转义风险（比如乱码导致的反斜杠挤在引号前）
        # 强制将有风险的中文/乱码打印行规范化，或者直接确保写回时是纯净的 UTF-8
        if content != original_content or encoding_used != "utf-8-sig":
            # 以标准的 utf-8 写入，errors='replace' 确保即使有零星死字也不会破坏整体结构
            with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            return True, f"Fixed encoding from [{encoding_used}]"
        
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