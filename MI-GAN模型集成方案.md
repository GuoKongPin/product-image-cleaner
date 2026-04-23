# MI-GAN 模型集成方案

## 现状分析

### 当前使用的算法
- **OpenCV Telea**：基于快速行进法的传统图像修复算法
- **优点**：轻量、快速、无需模型文件
- **缺点**：修复效果一般，复杂背景会有模糊/重复纹理

### Inpaint-Web 使用的模型
- **MI-GAN**（Model: https://github.com/Picsart-AI-Research/MI-GAN）
- **格式**：ONNX（约 30-50MB）
- **运行环境**：浏览器端 ONNX Runtime Web（WebGPU/WASM）
- **优点**：深度学习模型，修复效果更自然
- **缺点**：模型文件大，推理耗时较长

---

## 集成方案对比

### 方案A：Python 后端集成 MI-GAN（推荐）

**实现方式：**
```python
import onnxruntime as ort

# 加载 MI-GAN ONNX 模型
session = ort.InferenceSession("migan.onnx", providers=['CPUExecutionProvider'])

# 输入：图片 (1, 3, H, W) + Mask (1, 1, H, W)
# 输出：修复后的图片
```

**优点：**
- 后端统一处理，前端无需改动
- 支持批量处理
- 可利用服务器 CPU/GPU 资源
- 模型文件集中管理

**缺点：**
- 需要下载模型文件（~30-50MB）
- 推理速度较慢（CPU 上可能需要几秒）
- 增加后端内存占用

**技术难点：**
- MI-GAN 模型输入输出格式需要适配
- 需要预处理（图片 → CHW 格式）和后处理
- 可能需要固定输入尺寸（如 512x512）

---

### 方案B：前端集成 MI-GAN

**实现方式：**
- 将 Inpaint-Web 的 ONNX 推理逻辑集成到当前前端
- 使用 `onnxruntime-web` 在浏览器端运行模型
- 后端只负责保存标注和返回结果

**优点：**
- 后端压力小，无需模型文件
- 利用用户本地 GPU（WebGPU）加速
- 符合 Inpaint-Web 原设计

**缺点：**
- 前端代码复杂度高
- 首次加载需要下载模型（~30-50MB）
- 浏览器兼容性问题（WebGPU 支持有限）
- 批量处理体验差

---

### 方案C：混合方案（渐进式）

**实现方式：**
```
┌─────────────────────────────────────────┐
│              前端                        │
│  ┌─────────┐    ┌─────────────────┐    │
│  │ 轻量模式 │ or │  高质量模式      │    │
│  │ OpenCV  │ →  │  MI-GAN (Web)   │    │
│  │ (Telea) │    │  ONNX Runtime   │    │
│  └─────────┘    └─────────────────┘    │
└─────────────────────────────────────────┘
```

- **默认**：使用后端 OpenCV Telea（快速）
- **可选**：切换到前端 MI-GAN（高质量）

**优点：**
- 兼顾速度和效果
- 用户自主选择
- 向后兼容

**缺点：**
- 实现复杂度高
- 需要维护两套代码

---

## 推荐方案：方案A（Python 后端集成）

### 实现步骤

#### 1. 下载 MI-GAN 模型
```bash
# 模型地址
https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx

# 保存到后端目录
product-image-cleaner/backend/models/migan.onnx
```

#### 2. 安装依赖
```bash
pip install onnxruntime  # 或 onnxruntime-gpu
```

#### 3. 实现 MI-GAN 修复函数
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

class MiganInpainter:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.mask_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
    
    def inpaint(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        # 1. 预处理：转为 CHW 格式
        img_np = np.array(img).transpose(2, 0, 1)  # HWC → CHW
        mask_np = np.array(mask)
        
        # 2. 归一化
        img_np = img_np.astype(np.float32) / 255.0
        mask_np = mask_np.astype(np.float32) / 255.0
        
        # 3. 添加 batch 维度
        img_np = np.expand_dims(img_np, 0)  # (1, 3, H, W)
        mask_np = np.expand_dims(np.expand_dims(mask_np, 0), 0)  # (1, 1, H, W)
        
        # 4. 推理
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: img_np, self.mask_name: mask_np}
        )
        
        # 5. 后处理
        result = outputs[0][0].transpose(1, 2, 0)  # CHW → HWC
        result = (result * 255).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
```

#### 4. 替换后端修复函数
```python
# backend/app.py

# 全局模型实例（懒加载）
_migan_inpainter = None

def get_migan_inpainter():
    global _migan_inpainter
    if _migan_inpainter is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'migan.onnx')
        if os.path.exists(model_path):
            _migan_inpainter = MiganInpainter(model_path)
        else:
            print("[Warning] MI-GAN model not found, fallback to OpenCV")
    return _migan_inpainter

def inpaint_image(img: Image.Image, mask: Image.Image) -> Image.Image:
    """智能选择修复算法"""
    migan = get_migan_inpainter()
    if migan:
        try:
            return migan.inpaint(img, mask)
        except Exception as e:
            print(f"[MI-GAN] Error: {e}, fallback to OpenCV")
    
    # 回退到 OpenCV Telea
    return inpaint_opencv(img, mask)
```

---

## 模型输入输出规格

根据 Inpaint-Web 代码分析：

### 输入
| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| image | (1, 3, H, W) | uint8 | RGB 图片，CHW 格式 |
| mask | (1, 1, H, W) | uint8 | 二值掩码，0=保留，255=修复 |

### 输出
| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| output | (1, 3, H, W) | uint8 | 修复后的图片，CHW 格式 |

**注意**：MI-GAN 可能需要固定尺寸输入（如 512x512），需要预处理 resize 和后处理还原。

---

## 性能对比预估

| 算法 | 速度（512x512） | 效果 | 资源占用 |
|------|----------------|------|----------|
| OpenCV Telea | ~50ms | ⭐⭐⭐ | 低 |
| MI-GAN (CPU) | ~2-5s | ⭐⭐⭐⭐⭐ | 高 |
| MI-GAN (GPU) | ~200-500ms | ⭐⭐⭐⭐⭐ | 高 |

---

## 建议

### 短期（快速实现）
保持当前 OpenCV Telea 方案，效果已能满足大部分需求。

### 中期（渐进增强）
1. 下载 MI-GAN 模型到后端
2. 实现 Python ONNX 推理封装
3. 提供配置选项让用户选择算法

### 长期（完整方案）
- 支持模型热切换
- 异步队列处理大图片
- GPU 服务器部署

---

## 模型下载地址

```
主模型：
https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx

备用：
https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx
```

模型大小约 30-50MB，需要科学上网下载。
