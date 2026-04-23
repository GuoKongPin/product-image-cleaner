"""
LaMa ONNX 图像修复模型封装
基于 Carve/LaMa-ONNX 模型 (big-lama)
"""
import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort


class LaMaInpainter:
    """LaMa 深度学习图像修复器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化 LaMa 修复器
        
        Args:
            model_path: ONNX 模型文件路径，默认查找 models/lama_fp32.onnx
        """
        if model_path is None:
            # 默认模型路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'lama_fp32.onnx')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa 模型文件不存在: {model_path}\n"
                                   f"请从 https://huggingface.co/Carve/LaMa-ONNX 下载 lama_fp32.onnx")
        
        self.model_path = model_path
        self.session = None
        self.input_names = None
        self.output_names = None
        
        # 延迟加载模型（首次使用时加载）
        self._model_loaded = False
    
    def _load_model(self):
        """加载 ONNX 模型"""
        if self._model_loaded:
            return
        
        print(f"[LaMa] 加载模型: {self.model_path}")
        
        # 配置 ONNX Runtime 会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 优先使用 GPU，其次 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            print(f"[LaMa] 模型加载成功")
            print(f"[LaMa] 输入: {self.input_names}")
            print(f"[LaMa] 输出: {self.output_names}")
            self._model_loaded = True
        except Exception as e:
            print(f"[LaMa] 模型加载失败: {e}")
            raise
    
    def _ceil_modulo(self, x: int, mod: int) -> int:
        """向上取整到模长的倍数"""
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod
    
    def _pad_to_modulo(self, img: np.ndarray, mod: int = 8) -> tuple:
        """
        将图像填充到模长的倍数（FourierUnit要求）
        
        Returns:
            (padded_img, original_height, original_width)
        """
        if img.ndim == 3:
            channels, height, width = img.shape
        else:
            height, width = img.shape
            channels = 1
            img = img[np.newaxis, ...]
        
        out_height = self._ceil_modulo(height, mod)
        out_width = self._ceil_modulo(width, mod)
        
        if out_height == height and out_width == width:
            return img, height, width
        
        padded = np.pad(
            img, 
            ((0, 0), (0, out_height - height), (0, out_width - width)), 
            mode="symmetric"
        )
        return padded, height, width
    
    def _preprocess_image(self, img: Image.Image, target_size: tuple = None) -> tuple:
        """
        预处理图像
        
        Args:
            img: PIL RGB 图像
            target_size: 目标尺寸 (width, height)，None则保持原尺寸
            
        Returns:
            (numpy array shape: (1, 3, H, W), original_size, padding_info)
        """
        # 调整尺寸
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        
        # 转换为 numpy array (HWC)
        img_array = np.array(img).astype(np.float32)  # (H, W, 3)
        original_h, original_w = img_array.shape[:2]
        
        # 归一化到 [0, 1]
        img_array = img_array / 255.0
        
        # 转换为 CHW 格式
        img_chw = np.transpose(img_array, (2, 0, 1))  # (3, H, W)
        
        # 填充到 8 的倍数 (FourierUnit要求)
        img_padded, padded_h, padded_w = self._pad_to_modulo(img_chw, mod=8)
        
        # 添加 batch 维度
        img_bchw = np.expand_dims(img_padded, axis=0)  # (1, 3, H, W)
        
        return img_bchw, (original_h, original_w), (padded_h, padded_w)
    
    def _preprocess_mask(self, mask: Image.Image, target_size: tuple) -> np.ndarray:
        """
        预处理 mask
        
        LaMa mask 格式：
        - 1 (白色/True) 表示需要修复的区域
        - 0 (黑色/False) 表示保留的区域
        
        Args:
            mask: PIL L 模式图像（灰度）- 框选区域为白色(255)
            target_size: 目标尺寸 (width, height)
            
        Returns:
            numpy array shape: (1, 1, H, W), dtype=float32
        """
        # 调整 mask 尺寸
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)
        
        # 转换为 numpy array
        mask_array = np.array(mask).astype(np.float32)  # (H, W)
        
        # 二值化：白色(255) -> 1 (需要修复), 黑色(0) -> 0 (保留)
        mask_binary = (mask_array > 127).astype(np.float32)
        
        # 添加 channel 维度
        mask_chw = mask_binary[np.newaxis, ...]  # (1, H, W)
        
        # 填充到 8 的倍数
        mask_padded, _, _ = self._pad_to_modulo(mask_chw, mod=8)
        
        # 添加 batch 维度
        mask_bchw = np.expand_dims(mask_padded, axis=0)  # (1, 1, H, W)
        
        return mask_bchw
    
    def _postprocess(self, output: np.ndarray, original_size: tuple, padded_size: tuple) -> Image.Image:
        """
        后处理模型输出
        
        Args:
            output: 模型输出 (NCHW format), 值域 [0, 255]
            original_size: (height, width) 原始尺寸
            padded_size: (height, width) 填充后的尺寸
            
        Returns:
            PIL RGB 图像
        """
        # 移除 batch 维度
        output_chw = output[0]  # (3, H, W)
        
        # 裁剪回原始填充尺寸
        padded_h, padded_w = padded_size
        output_chw = output_chw[:, :padded_h, :padded_w]
        
        # 裁剪回原始图像尺寸
        original_h, original_w = original_size
        output_chw = output_chw[:, :original_h, :original_w]
        
        # 转换为 HWC
        output_hwc = np.transpose(output_chw, (1, 2, 0))  # (H, W, 3)
        
        # 裁剪到有效范围 [0, 255]
        output_hwc = np.clip(output_hwc, 0, 255)
        
        # 转换为 uint8
        output_hwc = output_hwc.astype(np.uint8)
        
        # 创建 PIL Image
        result_img = Image.fromarray(output_hwc, mode='RGB')
        
        return result_img
    
    def inpaint(self, image: Image.Image, mask: Image.Image, target_size: tuple = None) -> Image.Image:
        """
        执行图像修复
        LaMa模型固定输入512x512，需要将图像resize后处理再resize回原尺寸
        
        Args:
            image: PIL RGB 图像
            mask: PIL L 模式 mask，白色区域表示需要修复
            target_size: 可选，强制调整到的尺寸 (width, height)
            
        Returns:
            修复后的 PIL RGB 图像
        """
        self._load_model()
        
        # 保存原始尺寸
        original_size = image.size
        
        # LaMa模型固定输入512x512
        model_input_size = (512, 512)
        
        # Resize图像和mask到512x512
        img_resized = image.resize(model_input_size, Image.LANCZOS)
        mask_resized = mask.resize(model_input_size, Image.NEAREST)
        
        # 预处理
        img_tensor, _, _ = self._preprocess_image(img_resized, model_input_size)
        mask_tensor = self._preprocess_mask(mask_resized, model_input_size)
        
        print(f"[LaMa] 处理尺寸: {original_size} -> {model_input_size} (模型固定输入)")
        
        # 推理
        feed_dict = {
            self.input_names[0]: img_tensor,
            self.input_names[1]: mask_tensor
        }
        
        outputs = self.session.run(self.output_names, feed_dict)
        
        # 后处理 (512x512)
        result_512 = self._postprocess(outputs[0], (512, 512), (512, 512))
        
        # Resize回原尺寸
        result = result_512.resize(original_size, Image.LANCZOS)
        
        return result


# 全局单例实例
_lama_inpainter = None


def get_lama_inpainter(model_path: str = None) -> LaMaInpainter:
    """获取 LaMa 修复器单例"""
    global _lama_inpainter
    if _lama_inpainter is None:
        _lama_inpainter = LaMaInpainter(model_path)
    return _lama_inpainter


def lama_inpaint(image: Image.Image, mask: Image.Image, model_path: str = None, target_size: tuple = None) -> Image.Image:
    """
    使用 LaMa 模型修复图像的便捷函数
    
    Args:
        image: PIL RGB 图像
        mask: PIL L 模式 mask
        model_path: 可选，自定义模型路径
        target_size: 可选，强制调整到的尺寸 (width, height)
        
    Returns:
        修复后的 PIL RGB 图像
    """
    inpainter = get_lama_inpainter(model_path)
    return inpainter.inpaint(image, mask, target_size)
