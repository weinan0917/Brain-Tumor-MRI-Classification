# Brain Tumor MRI Classification

基于多源 MRI 影像数据的脑肿瘤四分类项目，涵盖数据合并去重、预处理、手工特征分析，以及多种深度学习模型的训练与可解释性分析。

**分类类别（4 类）：** `glioma`（胶质瘤）、`meningioma`（脑膜瘤）、`notumor`（无肿瘤）、`pituitary`（垂体瘤）

---

## 目录

- [项目结构](#项目结构)
- [数据说明](#数据说明)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [数据处理流程](#数据处理流程)
- [模型训练](#模型训练)
- [可解释性分析](#可解释性分析)
- [配置说明](#配置说明)
- [输出目录](#输出目录)
- [注意事项](#注意事项)

---

## 项目结构

```
Brain Tumor MRI Classification/
├── config.py                  # 全局配置（路径、超参数、随机种子）
├── data_mix.py                # 四数据集合并与去重
├── Dataset_Load_Initial.py    # 数据加载、EDA、预处理、划分
├── feature_analysis.py        # 手工特征提取与降维可视化
├── requirements.txt           # Python 依赖
├── data/
│   ├── processed_data/        # 预处理 + 特征提取结果
│   └── plots/                 # EDA 与特征分析图表
├── models/
│   ├── ResNet50V2/            # TensorFlow/Keras 两阶段迁移学习
│   ├── DeepSCNN/              # 双视图 Siamese + 注意力机制
│   ├── GKLNN/                 # 核原型 + 液态神经网络
│   ├── ViT-B/                 # PyTorch Vision Transformer
│   └── ConvNeXtTiny/          # PyTorch ConvNeXt-Tiny
└── reports/                   # 全局报告输出
```

---



## 数据说明

所有路径均为**相对项目根目录**的相对路径。

### `data/processed_data/` 子目录

```
processed_data/
├── train/ val/ test/          # 按 70% / 15% / 15% 划分的预处理 PNG 图像
├── arrays/                    # X_train.npy, y_train.npy 等 NumPy 数组
├── features/                  # feat_train.npy, feat_test.npy（HOG/LBP/GLCM）
└── metadata/                  # dataset_info.csv, label_mapping.json 等
```

---



## 环境配置



### 系统要求

- Python 3.10+
- 推荐 16 GB 以上内存（完整数据集预处理）
- 默认使用 CPU 训练（`config.py` 中 `CUDA_VISIBLE_DEVICES=-1`）；PyTorch 模型可按需启用 GPU



### 安装依赖

```powershell
# 创建虚拟环境（推荐）
python -m venv .venv
.\.venv\Scripts\activate

# 安装全部依赖（清华镜像）
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

若 PyTorch 需指定 CUDA 版本，请先到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 wheel，再安装其余依赖。

### 主要依赖


| 框架               | 包                                             | 用途                        |
| ---------------- | --------------------------------------------- | ------------------------- |
| TensorFlow/Keras | `tensorflow`, `scikit-learn`                  | ResNet50V2、DeepSCNN、GKLNN |
| PyTorch          | `torch`, `torchvision`, `timm`                | ViT-B、ConvNeXtTiny        |
| 图像处理             | `Pillow`, `scikit-image`, `imagehash`, `h5py` | 预处理、去重、`.mat` 读取          |
| 可解释性             | `lime`, `shap`                                | XAI 分析                    |


---



## 快速开始

在项目根目录下依次执行：

```powershell
# 1. 合并四个原始数据集并去重
python data_mix.py

# 2. 加载混合数据集、EDA、预处理、划分 train/val/test
python Dataset_Load_Initial.py

# 3. 提取手工特征（HOG / LBP / GLCM）并做 PCA、t-SNE 可视化
python feature_analysis.py

# 4. 训练模型（以 ResNet50V2 为例）
python models/ResNet50V2/ResNet50V2.py --dataset processed
```

> 若 `data/mixed_data/` 与 `data/processed_data/` 已存在，可跳过对应步骤。

---



## 数据处理流程



### Step 1：数据集合并与去重

```powershell
python data_mix.py
python data_mix.py --force-export   # 强制重新展开 BrainTumorDataPublic 的 .mat 文件
```

**处理步骤：**

1. 将 `BrainTumorDataPublic` 的 `.mat` 切片展开为 JPG，缓存至 `data/ori_data/_expanded/BrainTumorDataPublic/`
2. 合并四个数据集，统一标签（如 `no_tumor` → `notumor`）
3. SHA-256 精确去重 + pHash 感知哈希近重复去除（Hamming 距离 ≤ 3）
4. 输出至 `data/mixed_data/{glioma,meningioma,notumor,pituitary}/`

**输出文件：** `manifest.json`、`manifest.csv`、`DATA_MIX_REPORT.md`

### Step 2：预处理与划分

```powershell
python Dataset_Load_Initial.py
```

**处理内容：**

- 类别分布、分辨率等 EDA 分析（图表保存至 `data/plots/`）
- 中值滤波去噪 → CLAHE 增强 → 高斯平滑 → 224×224 归一化
- Isolation Forest 异常值检测
- 分层划分 train / val / test（70% / 15% / 15%）
- 保存预处理 PNG、NumPy 数组及元数据

**快速调试模式：** 在 `config.py` 中设置 `QUICK_RUN = True`，每类仅采样 60 张。

### Step 3：手工特征分析

```powershell
python feature_analysis.py
```

从 `data/processed_data/arrays/` 加载数据，提取 HOG、LBP 直方图、GLCM 纹理特征（共 406 维），输出 PCA / t-SNE 可视化及特征统计报告。

---



## 模型训练

所有 TensorFlow 模型支持 `--dataset processed`（预处理数组）或 `--dataset ori`（从混合数据集实时加载）。

### ResNet50V2（TensorFlow / Keras）

ImageNet 预训练 ResNet50V2，两阶段迁移学习（冻结骨干 → 微调顶层）。

```powershell
python models/ResNet50V2/ResNet50V2.py --dataset processed
python models/ResNet50V2/ResNet50V2.py --dataset processed --batch-size 16 --phase1-epochs 12 --phase2-epochs 20
```



### DeepSCNN-Brain Attention（TensorFlow / Keras）

双视图 Siamese 架构 + 通道/空间双重注意力 + 残差编码器，ImageNet 预训练 ResNet50V2 骨干。

```powershell
python models/DeepSCNN/DeepSCNN.py --dataset processed
python models/DeepSCNN/DeepSCNN.py --dataset processed --batch-size 16 --phase1-epochs 15 --phase2-epochs 30
```



### GKLNN（TensorFlow / Keras）

双特征流（PCA 降维特征 + 手工特征）+ K-Means 原型选择 + 液态核学习网络。

```powershell
python models/GKLNN/GKLNN.py --dataset processed
python models/GKLNN/GKLNN.py --dataset processed --epochs 180 --batch-size 48 --refresh-cache
```



### ViT-B（PyTorch / timm）

基于 `vit_base_patch16_224`，使用 `data/processed_data/` 中 train/val/test 目录的 ImageFolder 加载。

```powershell
# ImageNet-21k 预训练微调
python models/ViT-B/train_vit_processed_imagenet21k.py

# 从零训练（无预训练）
python models/ViT-B/train_vit_processed_no_imagenet21k.py

# 评估
python models/ViT-B/evaluate_vit_processed.py --split test
```

配置文件：`models/ViT-B/config_vit_processed.yaml`（默认 `data/processed_data`，输出至 `models/ViT-B/outputs/`）

### ConvNeXt-Tiny（PyTorch / timm）

```powershell
python models/ConvNeXtTiny/train_convnext_tiny.py
python models/ConvNeXtTiny/train_convnext_tiny.py --epochs 30 --batch-size 16 --lr 0.0001
```

配置文件：`models/ConvNeXtTiny/config_convnext_tiny.yaml`

> 本地预训练权重路径：`models/ConvNeXtTiny/weights/convnext_tiny_22k_224.pth`（需自行下载放置）



### CIFAR-100 基准实验（ViT-B，可选）

```powershell
python models/ViT-B/train_vit_cifar100_imagenet21k.py
python models/ViT-B/train_vit_cifar100_no_imagenet21k.py
python models/ViT-B/evaluate_vit_cifar100.py --split test
```

数据路径配置见 `models/ViT-B/config_vit_cifar100.yaml`（默认 `data/cifar-100-python-train450-val50`）。

---



## 可解释性分析

对训练好的 ResNet50V2 进行 XAI 分析（Grad-CAM、Grad-CAM++、LIME、SHAP、MC Dropout、集成不确定性、误差分析）：

```powershell
python models/ResNet50V2/XAI.py --dataset processed
python models/ResNet50V2/XAI.py --dataset processed --quick
```

**依赖：** 需先完成 ResNet50V2 训练并生成 checkpoint；LIME 与 SHAP 需已安装（见 `requirements.txt`）。

**输出目录：** `models/ResNet50V2/output/xai_reports/` 与 `data/plots/`

---



## 配置说明

全局配置集中在 `config.py`：


| 参数                                     | 默认值                  | 说明        |
| -------------------------------------- | -------------------- | --------- |
| `SEED`                                 | `42`                 | 随机种子      |
| `TARGET_SIZE`                          | `(224, 224)`         | 输入图像尺寸    |
| `TRAIN_RATIO / VAL_RATIO / TEST_RATIO` | `0.70 / 0.15 / 0.15` | 数据集划分比例   |
| `QUICK_RUN`                            | `False`              | 快速调试子采样模式 |
| `OUTLIER_CONTAMINATION`                | `0.03`               | 异常值检测比例   |


路径相关常量：

```python
MIXED_DATASET_PATH    = "data/mixed_data"
PROCESSED_DATASET_PATH = "data/processed_data"
DATASET_1_DIR         = "data/ori_data/Brain Tumor MRI Dataset"
DATASET_2_DIR         = "data/ori_data/BRISC 2025 Dataset"
```

路径解析工具：

- `resolve_project_path(path)` — 相对路径 → 绝对路径
- `to_relative_path(path)` — 绝对路径 → 相对路径（POSIX 格式）

---



## 输出目录


| 模块             | 输出位置                                      |
| -------------- | ----------------------------------------- |
| 数据 EDA / 特征可视化 | `data/plots/`                             |
| 预处理元数据         | `data/processed_data/metadata/`           |
| ResNet50V2     | `models/ResNet50V2/outputs/{processed     |
| DeepSCNN       | `models/DeepSCNN/output/`                 |
| GKLNN          | `models/GKLNN/outputs/`                   |
| ViT-B          | `models/ViT-B/outputs/{run_name}/`        |
| ConvNeXtTiny   | `models/ConvNeXtTiny/outputs/{run_name}/` |
| XAI 报告         | `models/ResNet50V2/output/xai_reports/`   |


各模型输出通常包含：

- `checkpoints/` — 最优模型权重
- `plots/` — 训练曲线、混淆矩阵、ROC/PR 曲线
- `reports/` — `metrics.csv`、`training_history.json` 等

---



## 注意事项

1. **路径规范：** 代码与配置文件中一律使用相对项目根目录的路径，避免硬编码绝对路径。
2. **运行目录：** 所有命令均需在项目根目录（`1111Final/`）下执行。
3. **数据顺序：** 首次运行须先执行 `data_mix.py`，再执行 `Dataset_Load_Initial.py`；`feature_analysis.py` 依赖预处理数组。
4. **BrainTumor 展开缓存：** 首次合并时会将 `.mat` 展开至 `data/ori_data/_expanded/BrainTumorDataPublic/`，耗时较长，后续运行会跳过已缓存文件。
5. **CPU 模式：** TensorFlow 模型默认强制 CPU（`config.py` 中 `CUDA_VISIBLE_DEVICES=-1`），如需 GPU 请修改该配置。
6. **ConvNeXt 预训练权重：** 需手动下载 `convnext_tiny_22k_224.pth` 并放置于 `models/ConvNeXtTiny/weights/`。

---



## 推荐工作流

```
原始数据 (data/ori_data/)
        │
        ▼  data_mix.py
混合去重集 (data/mixed_data/)
        │
        ▼  Dataset_Load_Initial.py
预处理数据 (data/processed_data/)
        │
        ├──▶ feature_analysis.py  →  手工特征 + 降维可视化
        │
        └──▶ 模型训练
               ├── ResNet50V2 / DeepSCNN / GKLNN  (TensorFlow)
               ├── ViT-B / ConvNeXtTiny             (PyTorch)
               └── XAI.py                           (可解释性)
```

