# 模型文件下载指南 | Model Download Guide

本项目使用两个 ONNX 格式的深度学习模型进行图像修复。由于模型文件较大，请从以下链接下载后放置到 `models/` 目录。

## 模型列表 | Model List

### 1. MI-GAN 模型

- **文件名**: `migan_pipeline_v2.onnx`
- **大小**: ~28 MB
- **用途**: 快速图像修复，适合小面积消除
- **来源**: [MI-GAN GitHub](https://github.com/Picsart-AI-Research/MI-GAN)

**下载地址**:
- 官方: https://github.com/Picsart-AI-Research/MI-GAN/releases
- 镜像: [Release 页面](https://github.com/yourusername/product-image-cleaner/releases)

### 2. LaMa 模型

- **文件名**: `lama_fp32.onnx`
- **大小**: ~200 MB
- **用途**: 高质量图像修复，适合大面积替换
- **来源**: [LaMa GitHub](https://github.com/advimman/lama)

**下载地址**:
- 官方: https://github.com/advimman/lama/releases
- 镜像: [Release 页面](https://github.com/yourusername/product-image-cleaner/releases)

## 目录结构 | Directory Structure

下载后，请确保模型文件放在正确的位置：

```
product-image-cleaner/
├── models/
│   ├── migan_pipeline_v2.onnx    # MI-GAN 模型
│   └── lama_fp32.onnx            # LaMa 模型
├── backend/
├── frontend/
└── ...
```

## 自动下载脚本 | Auto Download Script

你也可以使用以下 Python 脚本自动下载模型：

```python
# download_models.py
import os
import urllib.request

MODELS_DIR = "models"
MODELS = {
    "migan_pipeline_v2.onnx": {
        "url": "https://github.com/yourusername/product-image-cleaner/releases/download/v1.0.0/migan_pipeline_v2.onnx",
        "size": "28MB"
    },
    "lama_fp32.onnx": {
        "url": "https://github.com/yourusername/product-image-cleaner/releases/download/v1.0.0/lama_fp32.onnx",
        "size": "200MB"
    }
}

def download_model(filename, url):
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        print(f"✓ {filename} 已存在")
        return
    
    print(f"⬇️  正在下载 {filename} ({MODELS[filename]['size']})...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    print(f"✓ {filename} 下载完成")

if __name__ == "__main__":
    for filename, info in MODELS.items():
        download_model(filename, info["url"])
    print("\n所有模型下载完成！")
```

## 模型转换（高级用户）| Model Conversion

如果你需要从原始 PyTorch 模型转换为 ONNX：

### MI-GAN 转换

```python
# 参考 MI-GAN 官方仓库的导出脚本
# https://github.com/Picsart-AI-Research/MI-GAN
```

### LaMa 转换

```python
# 参考 LaMa 官方仓库的导出脚本
# https://github.com/advimman/lama
```

## 许可证说明 | License Notice

这些模型文件遵循其原始项目的许可证：

- **MI-GAN**: Apache 2.0 License
- **LaMa**: Apache 2.0 License

使用本工具时，请同时遵守模型原始项目的许可证条款。

## 常见问题 | FAQ

**Q: 模型文件太大，可以只用一个吗？**  
A: 可以。如果只下载 MI-GAN，请将 `INPAINT_MODEL` 设置为 `migan`。

**Q: 模型下载很慢怎么办？**  
A: 可以尝试使用 GitHub 镜像或代理下载。

**Q: 可以使用自己的模型吗？**  
A: 可以。只要模型输入输出格式兼容，替换 `models/` 目录下的文件即可。
