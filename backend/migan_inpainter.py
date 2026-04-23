"""
MI-GAN ONNX 图像修复模型封装
基于 Inpaint-Web 的 migan_pipeline_v2.onnx 模型
"""
import os
import numpy as np
from PIL import Image
import onnxruntime as ort


class MIGANInpainter:
    """MI-GAN 深度学习图像修复器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化 MI-GAN 修复器
        
        Args:
            model_path: ONNX 模型文件路径，默认查找 models/migan_pipeline_v2.onnx
        """
        if model_path is None:
            # 默认模型路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'migan_pipeline_v2.onnx')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MI-GAN 模型文件不存在: {model_path}")
        
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
        
        print(f"[MI-GAN] 加载模型: {self.model_path}")
        
        # 配置 ONNX Runtime 会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 优先使用 CPU 执行（兼容性好）
        providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            print(f"[MI-GAN] 模型加载成功")
            print(f"[MI-GAN] 输入: {self.input_names}")
            print(f"[MI-GAN] 输出: {self.output_names}")
            self._model_loaded = True
        except Exception as e:
            print(f"[MI-GAN] 模型加载失败: {e}")
            raise
    
    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        预处理图像
        将 PIL Image 转换为模型输入格式 (NCHW, uint8)
        
        Args:
            img: PIL RGB 图像
            
        Returns:
            numpy array shape: (1, 3, H, W), dtype=uint8
        """
        # 转换为 numpy array (HWC)
        img_array = np.array(img)  # (H, W, 3)
        
        # 转换为 CHW 格式
        img_chw = np.transpose(img_array, (2, 0, 1))  # (3, H, W)
        
        # 添加 batch 维度
        img_bchw = np.expand_dims(img_chw, axis=0)  # (1, 3, H, W)
        
        return img_bchw.astype(np.uint8)
    
    def _preprocess_mask(self, mask: Image.Image, target_size: tuple) -> np.ndarray:
        """
        预处理 mask
        将 mask 转换为模型输入格式 (NCHW, uint8)
        
        注意：MI-GAN 模型的 mask 格式与常规相反：
        - 0 (黑色) 表示需要修复的区域
        - 255 (白色) 表示保留的区域
        
        参考 Inpaint-Web 的 markProcess 逻辑：
        (channelData !== 255) * 255
        即：非白色(255)的像素变成255，白色像素变成0
        
        Args:
            mask: PIL L 模式图像（灰度）- 框选区域为白色(255)
            target_size: 目标尺寸 (width, height)
            
        Returns:
            numpy array shape: (1, 1, H, W), dtype=uint8
        """
        # 调整 mask 尺寸
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)
        
        # 转换为 numpy array
        mask_array = np.array(mask)  # (H, W)
        
        # MI-GAN 模型 mask 格式转换：
        # 输入：白色(255)表示框选区域（需要修复）
        # 输出：0 表示修复区域，255 表示保留区域
        # 逻辑：(像素 !== 255) * 255，即非白色变成255，白色变成0
        mask_binary = (mask_array != 255).astype(np.uint8) * 255
        
        # 添加 channel 和 batch 维度
        mask_bchw = np.expand_dims(np.expand_dims(mask_binary, axis=0), axis=0)  # (1, 1, H, W)
        
        return mask_bchw.astype(np.uint8)
    
    def _postprocess(self, output: np.ndarray, width: int, height: int) -> Image.Image:
        """
        后处理模型输出
        将模型输出转换为 PIL Image
        
        Args:
            output: 模型输出 (NCHW format)
            width: 输出图像宽度
            height: 输出图像高度
            
        Returns:
            PIL RGB 图像
        """
        # 移除 batch 维度
        output_chw = output[0]  # (3, H, W)
        
        # 转换为 HWC
        output_hwc = np.transpose(output_chw, (1, 2, 0))  # (H, W, 3)
        
        # 裁剪到有效范围 [0, 255]
        output_hwc = np.clip(output_hwc, 0, 255)
        
        # 转换为 uint8
        output_hwc = output_hwc.astype(np.uint8)
        
        # 创建 PIL Image
        result_img = Image.fromarray(output_hwc, mode='RGB')
        
        return result_img
    
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """执行图像修复"""
        self._load_model()
        
        img_tensor = self._preprocess_image(image)
        mask_tensor = self._preprocess_mask(mask, image.size)
        
        feed_dict = {
            self.input_names[0]: img_tensor,
            self.input_names[1]: mask_tensor
        }
        
        outputs = self.session.run(self.output_names, feed_dict)
        
        return self._postprocess(outputs[0], image.size[0], image.size[1])


# 全局单例实例
_migan_inpainter = None


def get_migan_inpainter(model_path: str = None) -> MIGANInpainter:
    """获取 MI-GAN 修复器单例"""
    global _migan_inpainter
    if _migan_inpainter is None:
        _migan_inpainter = MIGANInpainter(model_path)
    return _migan_inpainter


def migan_inpaint(image: Image.Image, mask: Image.Image, model_path: str = None) -> Image.Image:
    """
    使用 MI-GAN 模型修复图像的便捷函数
    
    Args:
        image: PIL RGB 图像
        mask: PIL L 模式 mask
        model_path: 可选，自定义模型路径
        
    Returns:
        修复后的 PIL RGB 图像
    """
    inpainter = get_migan_inpainter(model_path)
    return inpainter.inpaint(image, mask)
