import os

# ==========================================
# 自动路径替换配置 (你可以自由修改和添加)
# 格式: "本地路径": "远端服务器路径"
# ==========================================
PATH_MAPPINGS = {
    # 替换代码根目录
    "/code/users/jiangrui/Full-Hand-Tactile-Estimation": "/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation",
    
    # 替换数据集目录（请把下面的远端路径换成你实际在 ModelArts 上的数据集绝对路径）
    "/data1/jiangrui/OpenTouch Data": "/home/ma-user/work/cfzhao/OpenTouch Data", 
    "/data1/jiangrui/EgoTouch/": "/home/ma-user/work/cfzhao/EgoTouch/",
    "/data1/jiangrui/EgoTactile/": "/home/ma-user/work/cfzhao/EgoTactile/"
    # 你可以继续在这里无限添加你需要自动替换的字符串
}

def auto_replace_paths():
    """
    遍历当前目录所有的 Python 和 Shell 脚本，并自动替换其中的绝对路径。
    """
    print("🚀 开始扫描并自动替换远端服务器独有的路径...")
    
    valid_extensions = ('.py', '.sh')
    skip_dirs = {'.git', '__pycache__', 'wandb', 'logs', 'lightning_logs'}
    
    adapted_count = 0
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if not file.endswith(valid_extensions):
                continue
                
            filepath = os.path.join(root, file)
            # 排除本文件，防止把映射字典本身给替换乱了
            if file == "auto_adapt_paths.py":
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                for local_path, remote_path in PATH_MAPPINGS.items():
                    new_content = new_content.replace(local_path, remote_path)
                
                # 如果内容有变化，则重新写入
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"✨ [已适配远端路径] -> {filepath}")
                    adapted_count += 1
            except Exception as e:
                # 忽略非文本或编码无法读取的文件
                pass

    print("-" * 50)
    print(f"🎉 路径适配完成！共修改了 {adapted_count} 个文件。")

if __name__ == '__main__':
    auto_replace_paths()
