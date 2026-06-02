# US.T 统一环境搭建与安装指南 (INSTALLATION)

本指南整合了 `US.T` 目录下三大核心项目（`HaWoR`、`HaMeR`、`OpenTouch`）的环境依赖，提供了一套统一且经过兼容性设计的环境搭建方案。

---

## 🛠 一、 统一环境创建

由于各子项目均使用 **Python 3.10** 开发，并且依赖相同或相近版本的 PyTorch 与 CUDA 环境，强烈推荐您创建一个统一的 Conda 虚拟环境来管理依赖：

```bash
# 1. 创建并激活 Conda 环境
conda create --name ust_env python=3.10 -y
conda activate ust_env

# 2. 安装适配您 GPU 的 PyTorch 与 Torchvision 
#（推荐 PyTorch 2.1.0 或 1.13.0 + CUDA 11.8 / 12.1。以下以 CUDA 11.8 为例）
pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 二、 一键安装核心依赖 (requirements.txt)

激活环境并确保 PyTorch 安装成功后，运行以下命令一键安装常规 Python 包：

```bash
# 确保在 US.T 根目录下
pip install -r requirements.txt
```

---

## ⛓ 三、 特殊三维/编译依赖手动安装 (重要)

部分依赖直接托管于 GitHub，或需要本地编译（C++ / CUDA 扩展）。请按照以下顺序手动构建：

### 1. 安装 PyTorch3D
PyTorch3D 建议根据您的 PyTorch 和 CUDA 版本直接下载预编译的 wheel，或者从 Git 下载特定版本编译：
```bash
# 推荐采用编译安装（可能需要较长时间，确保本地已配置 g++ / nvcc）
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 2. 安装 torch-scatter
```bash
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# 请根据实际安装的 Torch 和 CUDA 版本调整 URL
```

### 3. 安装 DROID-SLAM (HaWoR 第三方库)
`HaWoR` 的世界坐标手部重建依赖 DROID-SLAM。请进入相应目录并执行编译安装：
```bash
cd HaWoR/thirdparty/DROID-SLAM
python setup.py install
cd ../../../
```

### 4. 安装 EasyMocap (OpenTouch 渲染子模块)
如需使用 OpenTouch 的渲染脚本，请执行：
```bash
cd opentouch/EasyMocap
pip install -e .
cd ../../
```

---

## 🗃 四、 外部预训练权重与 3D 手部模型 (MANO) 配置

要在本地完整运行所有演示及评估脚本，需要手动下载以下模型权重，并将其放置在指定的绝对路径中：

### 1. MANO 3D 手部模板 (关键)
请先前往 [MANO 官网](https://mano.is.tue.mpg.de/) 注册账号并下载 `mano_v1_2.zip`。
解压后，请分别复制以下文件到对应的子项目路径下：

- **对于 OpenTouch:**
  放置在 `opentouch/preprocess/scratch/MANO_RIGHT.pkl`
- **对于 HaWoR:**
  放置在 `HaWoR/_DATA/data/mano/MANO_RIGHT.pkl` 和 `HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl`

---

### 2. 外部预训练权重下载

#### (1) HaWoR 所需权重
您可以通过以下命令下载 `HaWoR` 重建所需的深度图估计与姿态回归模型：

```bash
# 下载 DROID-SLAM 官方权重并放入 external 目录
# 链接: https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view
# 存放位置: HaWoR/weights/external/droid.pth

# 下载 Metric3D 官方深度估计权重
# 链接: https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view
# 存放位置: HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth

# 下载 HaWoR 与 WiLoR 权重到指定文件夹
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./HaWoR/weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./HaWoR/weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./HaWoR/weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./HaWoR/weights/hawor/
```

#### (2) HaMeR 所需权重
`hamer` 可自动在首次运行时通过 `gdown` 下载其模型。如果遇到网络问题，请使用：
```bash
cd hamer
bash fetch_demo_data.sh
cd ..
```

---

## 🏃‍♀️ 五、 简易运行验证

成功配置环境与权重后，您可以通过运行以下各项目的 Demo 脚本来验证安装：

- **验证 HaWoR 世界空间手部估计：**
  ```bash
  cd HaWoR
  python demo.py --video_path ./example/video_0.mp4 --vis_mode world
  ```

- **验证 HaMeR 图像级手部 mesh 重建：**
  ```bash
  cd hamer
  python demo.py --img example_data/images/im1012.jpg
  ```

- **验证 OpenTouch 数据集渲染可视化：**
  ```bash
  cd opentouch
  python preprocess/build_demo.py --hdf5 data/fablab_ml_p1.hdf5 --demo-id demo_05 --fps 30
  ```
