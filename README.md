# Product Image Cleaner | 商品图片净化工具

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Flask-2.3+-green.svg" alt="Flask 2.3+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/ONNX-Runtime-orange.svg" alt="ONNX Runtime">
</p>

<p align="center">
  <b>智能消除图片文字，一键净化商品图片</b><br>
  <b>Intelligent text removal for product images</b>
</p>

---

## 🌟 功能特性 | Features

| 功能 | 描述 |
|------|------|
| 🔴 **智能消除** | 自动识别并消除营养成分表、配料表等文字 |
| 🟣 **文字替换** | 将商标文字智能替换为"xxx"，保留背景纹理 |
| 🎯 **主体识别** | 自动识别商品主体，一键抠图去背景 |
| ✏️ **精确标注** | 支持框选和画笔涂抹，精确控制处理区域 |
| 🚀 **双模型修复** | MI-GAN + LaMa 双模型，智能选择最优方案 |
| ⚡ **边缘羽化** | 修复区域边缘平滑过渡，消除拼接痕迹 |

---

## 📸 效果展示 | Demo

```
┌─────────────────┐    ┌─────────────────┐
│  原图：带文字    │ →  │  处理后：干净    │
│  的商品图片      │    │  的商品图片      │
└─────────────────┘    └─────────────────┘
```

### 处理示例
- **营养成分表消除** → 背景智能填充
- **商标文字替换** → "xxx" 替换，保持风格
- **配料表清除** → 完全无痕消除

---

## 🏗️ 技术架构 | Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    前端 Frontend                         │
│              (HTML5 + Canvas + Vanilla JS)               │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP API
┌────────────────────▼────────────────────────────────────┐
│                    后端 Backend                          │
│                    Flask Server                          │
├─────────────────────────────────────────────────────────┤
│  OCR识别  │  主体检测  │  智能分类  │  图像修复          │
│  PaddleOCR│  GrabCut  │  关键词匹配│  MI-GAN / LaMa     │
└─────────────────────────────────────────────────────────┘
```

### 双模型修复策略 | Dual-Model Inpainting

| 模型 | 优势 | 适用场景 |
|------|------|----------|
| **MI-GAN** | 速度快(0.2s)、显存低(4G) | 小面积消除、实时预览 |
| **LaMa** | 质量高、纹理连续 | 大面积修复、替换文字背景 |

---

## 🚀 快速开始 | Quick Start

### 环境要求 | Requirements

- Python 3.8+
- 4GB+ GPU 显存 (推荐) / CPU 也可运行
- Windows / Linux / macOS

### 安装步骤 | Installation

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/product-image-cleaner.git
cd product-image-cleaner

# 2. 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. 安装依赖
cd backend
pip install -r requirements.txt

# 4. 下载模型文件
# 模型文件较大，请从 Release 页面下载
# - migan_pipeline_v2.onnx (28MB)
# - lama_fp32.onnx (200MB)
# 放置到 models/ 目录

# 5. 启动服务
python app.py
```

### Windows 一键启动 | Windows Quick Start

双击运行 `start.bat`，自动完成环境配置并启动服务。

---

## 📖 使用指南 | Usage

### 1. 自动处理模式

```bash
POST /api/auto-process
```

上传图片后自动：
1. OCR 识别所有文字
2. 按关键词分类（含量/配料 → 消除；商标 → 替换）
3. 智能修复背景

### 2. 精确标注模式

```bash
POST /api/process-with-annotations
```

支持手动标注：
- 🔴 **消除区域**：框选要消除的文字
- 🟣 **替换区域**：框选要替换为"xxx"的文字
- ✏️ **画笔涂抹**：自由绘制处理区域

### 3. 主体识别抠图

```bash
POST /api/subject-cutout
```

自动识别商品主体，生成透明背景 PNG。

---

## ⚙️ 配置参数 | Configuration

通过环境变量自定义行为：

```bash
# 修复模型选择: auto | migan | lama
export INPAINT_MODEL=auto

# LaMa 触发阈值（区域占比 > 5% 时用 LaMa）
export LAMA_AREA_THRESHOLD=0.05

# 合并邻近 mask
export MERGE_MASKS_ENABLED=true
export MERGE_MASKS_DISTANCE=30

# 边缘羽化
export FEATHERING_ENABLED=true
export FEATHERING_RADIUS=3
```

---

## 📁 项目结构 | Project Structure

```
product-image-cleaner/
├── backend/              # Flask 后端
│   ├── app.py           # 主服务入口
│   ├── migan_inpainter.py   # MI-GAN 修复器
│   ├── lama_inpainter.py    # LaMa 修复器
│   └── requirements.txt     # Python 依赖
├── frontend/            # 前端页面
│   └── index.html       # 标注工作台
├── models/              # ONNX 模型文件
│   ├── migan_pipeline_v2.onnx
│   └── lama_fp32.onnx
├── start.bat            # Windows 启动脚本
└── README.md
```

---

## 🛠️ 开发计划 | Roadmap

- [x] 双模型智能切换
- [x] Mask 合并优化
- [x] 边缘羽化处理
- [ ] 批量处理队列
- [ ] 云端 API 服务
- [ ] 更多语言支持

---

## 🤝 贡献指南 | Contributing

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证 | License

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢 | Acknowledgments

- [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) - 快速图像修复模型
- [LaMa](https://github.com/advimman/lama) - 大掩码图像修复模型
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 中文 OCR 识别

---

<p align="center">
  Made with ❤️ for cleaner product images
</p>
