import os
import re

def sanitize_file_content(filepath):
    """
    终极安全版：
    1. 优先尝试 utf-8，失败再尝试 gbk。
    2. 解决"吞掉引号/SyntaxError"的根本原因：如果文件被转成 utf-8，但头部还保留着 `# -*- coding: gbk -*-`，
       Python 解释器会用 gbk 去解析 utf-8 的字节，导致中文字符的字节序列把后面的引号吞并。
       因此，脚本会自动修正 Python 文件头部的 coding 声明。
    3. 移除粗暴的全局 replace('\t', '    ')，因为它会破坏字符串内部的转义符 "\t"，导致意外的缩进或字符串解析错误。
    """
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
        
        if not raw_bytes:
            return False, "Empty file"

        content = None
        encoding_used = ""
        is_modified = False

        # 1. 尝试解码
        try:
            content = raw_bytes.decode('utf-8')
            encoding_used = "utf-8"
        except UnicodeDecodeError:
            try:
                content = raw_bytes.decode('gbk')
                encoding_used = "gbk"
                is_modified = True # 编码从 gbk 变为了 utf-8
            except UnicodeDecodeError:
                try:
                    content = raw_bytes.decode('gb18030')
                    encoding_used = "gb18030"
                    is_modified = True
                except UnicodeDecodeError:
                    return False, "Failed to decode with utf-8, gbk, and gb18030."

        # 2. 修正 Python 文件的 coding 声明 (解决"吞引号"的核心)
        if filepath.endswith('.py'):
            # 匹配 # -*- coding: gbk -*- 或者 # coding=gbk
            coding_pattern = re.compile(r'^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)', re.IGNORECASE | re.MULTILINE)
            match = coding_pattern.search(content)
            if match:
                declared_encoding = match.group(1).lower()
                if declared_encoding != 'utf-8':
                    # 将声明替换为 utf-8
                    content = content[:match.start(1)] + 'utf-8' + content[match.end(1):]
                    is_modified = True

        # 3. 修复特殊空格（不替换全局的 \t，避免破坏 "\t" 字符串）
        if '\xa0' in content:
            content = content.replace('\xa0', ' ')
            is_modified = True

        if is_modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fixed encoding from [{encoding_used}] and sanitized"
        
        return False, "Already Clean UTF-8"

    except Exception as e:
        return False, f"Error: {str(e)}"


if __name__ == "__main__":
    print("🚀 开始全盘扫描代码文件，自动修复编码、消除头声明冲突与不可见空格...")
    
    valid_extensions = ('.py', '.sh', '.md', '.txt')
    skip_dirs = {'.git', '__pycache__', 'wandb', '.idea', '.vscode', 'build', 'dist', 'data', '_DATA'}
    
    files_to_check = []
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith(valid_extensions):
                files_to_check.append(os.path.join(root, file))
                
    print(f"📂 找到待检查文件共计: {len(files_to_check)} 个")
    print("-" * 60)
    
    fixed_count = 0
    for filepath in files_to_check:
        success, msg = sanitize_file_content(filepath)
        if success:
            print(f"✨ [已修复] -> {filepath} ({msg})")
            fixed_count += 1
            
    print("-" * 60)
    print(f"🎉 扫描全面完成！本次共修复了 {fixed_count} 个文件。")