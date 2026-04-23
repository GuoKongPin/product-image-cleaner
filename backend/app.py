"""
商品图片净化工具 - 后端服务
核心功能：
1. OCR识别图片文字（EasyOCR，支持中英文）
2. 自动主体识别 + 抠图消除背景
3. 按关键词分类：含量/商品类型 → Inpaint消除；商标文字 → 替换为xxx
4. MI-GAN 深度学习模型修复（基于 Inpaint-Web，效果更好）
"""

import os, io, re, json, traceback, base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# 导入 MI-GAN 修复器
from migan_inpainter import get_migan_inpainter, migan_inpaint
# 导入 LaMa 修复器
from lama_inpainter import get_lama_inpainter, lama_inpaint

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 全局模型（懒加载，首次请求时初始化）
# ─────────────────────────────────────────────
_ocr = None
_migan = None
_lama = None

# 修复模型选择: 'migan', 'lama', 或 'auto' (智能选择)
INPAINT_MODEL = os.environ.get('INPAINT_MODEL', 'auto')  # 默认智能选择

# 智能选择阈值
# - 当消除区域总面积占图像比例超过此值时使用 LaMa
# - 否则使用 MI-GAN
LAMA_AREA_THRESHOLD = float(os.environ.get('LAMA_AREA_THRESHOLD', '0.05'))  # 默认5%

# 合并mask相关配置
MERGE_MASKS_ENABLED = os.environ.get('MERGE_MASKS_ENABLED', 'true').lower() == 'true'  # 默认开启
MERGE_MASKS_DISTANCE = int(os.environ.get('MERGE_MASKS_DISTANCE', '30'))  # 合并距离阈值（像素）
MERGE_MASKS_MIN_AREAS = int(os.environ.get('MERGE_MASKS_MIN_AREAS', '3'))  # 最少区域数才触发合并
MERGE_MASKS_MAX_TOTAL_RATIO = float(os.environ.get('MERGE_MASKS_MAX_TOTAL_RATIO', '0.30'))  # 最大合并区域占比（超过则用逐个处理）

# 羽化配置
FEATHERING_ENABLED = os.environ.get('FEATHERING_ENABLED', 'true').lower() == 'true'  # 默认开启
FEATHERING_RADIUS = int(os.environ.get('FEATHERING_RADIUS', '3'))  # 羽化半径（像素）

def get_ocr():
    global _ocr
    if _ocr is None:
        print("[OCR] 正在加载 EasyOCR 模型（首次需下载，约100MB）...")
        import easyocr
        _ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        print("[OCR] 加载完成")
    return _ocr

def get_migan():
    """获取 MI-GAN 修复器实例（懒加载）"""
    global _migan
    if _migan is None:
        print("[MI-GAN] 正在加载 MI-GAN 模型（首次需加载，约28MB）...")
        _migan = get_migan_inpainter()
        print("[MI-GAN] 加载完成")
    return _migan


def get_lama():
    """获取 LaMa 修复器实例（懒加载）"""
    global _lama
    if _lama is None:
        print("[LaMa] 正在加载 LaMa 模型（首次需加载，约200MB）...")
        _lama = get_lama_inpainter()
        print("[LaMa] 加载完成")
    return _lama


def get_inpainter():
    """获取当前配置的修复器实例"""
    if INPAINT_MODEL == 'lama':
        return get_lama()
    return get_migan()


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def detect_subject_auto(img: Image.Image, use_ocr=True, use_edge=True) -> dict:
    """
    改进的自动识别图片主体区域 - 类似PS的选择主体功能
    结合GrabCut算法 + 边缘检测 + 显著性 + 多尺度分析
    返回: {bbox: [x0,y0,x1,y1], confidence: float}
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    
    # 方法1: GrabCut算法 - 前景分割
    mask_gc = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 初始矩形 - 中心区域
    margin = 0.15
    rect = (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))
    
    try:
        cv2.grabCut(arr, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # 0和2是背景，1和3是前景
        mask_fg = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
    except:
        mask_fg = np.ones((h, w), dtype=np.uint8)
    
    # 方法2: 显著性检测 - 基于频域的显著性
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not success:
        saliencyMap = np.ones((h, w), dtype=np.float32) * 0.5
    
    # 方法3: 边缘检测
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # 方法4: 颜色聚类 - 找主要颜色区域
    # 将图片缩小加速处理
    small = cv2.resize(arr, (w//4, h//4))
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 找最大的聚类（通常为主体）
    label_counts = np.bincount(labels.flatten())
    main_label = np.argsort(label_counts)[-2] if len(label_counts) > 1 else 0  # 第二大的通常是主体
    
    # 综合权重图
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # GrabCut前景权重
    weight_map += mask_fg.astype(np.float32) * 0.35
    
    # 显著性权重
    weight_map += saliencyMap * 0.30
    
    # 边缘权重
    weight_map += (edges / 255.0) * 0.15
    
    # 颜色聚类权重（上采样到原图尺寸）
    label_img = labels.reshape(h//4, w//4)
    label_img_up = cv2.resize((label_img == main_label).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    weight_map += label_img_up.astype(np.float32) * 0.20
    
    # 中心偏好（主体通常在中心）
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    center_weight = 1 - (dist_from_center / max_dist) * 0.2
    weight_map *= center_weight
    
    # OCR文字排除
    if use_ocr:
        try:
            ocr = get_ocr()
            result = ocr.readtext(arr)
            for (box, text, conf) in result:
                pts = np.array(box, dtype=np.int32)
                x0, y0 = pts[:,0].min(), pts[:,1].min()
                x1, y1 = pts[:,0].max(), pts[:,1].max()
                # 降低文字区域的权重
                weight_map[max(0,y0-5):min(h,y1+5), max(0,x0-5):min(w,x1+5)] *= 0.3
        except Exception as e:
            print(f"[SubjectDetect] OCR排除失败: {e}")
    
    # 二值化找主体区域
    weight_norm = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
    binary = (weight_norm > 0.4).astype(np.uint8) * 255
    
    # 形态学操作清理
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 找最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:
        # 排除背景（label 0），找最大的前景区域
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        bw = stats[largest_idx, cv2.CC_STAT_WIDTH]
        bh = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        best_bbox = [x, y, x + bw, y + bh]
        confidence = min(1.0, areas[largest_idx-1] / (w * h * 0.8))
    else:
        # fallback: 使用滑动窗口找最佳区域
        best_score = 0
        best_bbox = None
        
        for scale in [0.4, 0.6, 0.8]:
            bw, bh = int(w * scale), int(h * scale)
            if bw < 100 or bh < 100:
                continue
            step_x, step_y = max(30, bw // 8), max(30, bh // 8)
            for y in range(0, h - bh, step_y):
                for x in range(0, w - bw, step_x):
                    score = weight_map[y:y+bh, x:x+bw].mean()
                    if score > best_score:
                        best_score = score
                        best_bbox = [x, y, x + bw, y + bh]
        
        if best_bbox is None:
            margin = 0.1
            best_bbox = [int(w*margin), int(h*margin), int(w*(1-margin)), int(h*(1-margin))]
        confidence = 0.5
    
    # 扩展bbox确保包含完整主体
    margin = 15
    best_bbox[0] = max(0, best_bbox[0] - margin)
    best_bbox[1] = max(0, best_bbox[1] - margin)
    best_bbox[2] = min(w, best_bbox[2] + margin)
    best_bbox[3] = min(h, best_bbox[3] + margin)
    
    print(f"[SubjectDetect] 检测到主体: {best_bbox}, 置信度: {confidence:.2f}")
    
    return {'bbox': best_bbox, 'confidence': confidence}

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 全局模型（懒加载，首次请求时初始化）
# ─────────────────────────────────────────────
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        print("[OCR] 正在加载 EasyOCR 模型（首次需下载，约100MB）...")
        import easyocr
        _ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        print("[OCR] 加载完成")
    return _ocr


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def run_ocr(img: Image.Image):
    """
    返回列表，每项：
    {
      'text': str,
      'box': [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
      'confidence': float,
      'bbox': (x_min, y_min, x_max, y_max)   # 轴对齐包围盒，带 padding
    }
    """
    ocr = get_ocr()
    arr = np.array(img.convert("RGB"))
    # EasyOCR 返回: list of (box, text, confidence)
    result = ocr.readtext(arr)
    items = []
    for (box, text, conf) in result:
        # box 是 4 个点 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        pad = 4
        bbox = (
            max(0, int(min(xs)) - pad),
            max(0, int(min(ys)) - pad),
            min(img.width,  int(max(xs)) + pad),
            min(img.height, int(max(ys)) + pad),
        )
        items.append({'text': text, 'box': box, 'confidence': conf, 'bbox': bbox})
    return items


def classify_text(text: str, keywords: dict) -> str:
    """
    返回分类：'dosage' / 'category' / 'brand' / 'keep' / 'unknown'
    优先级：keep > dosage > category > brand > unknown
    """
    t = text.strip()

    # keep 优先
    for kw in keywords.get('keep', []):
        if re.search(kw, t, re.IGNORECASE):
            return 'keep'

    # dosage
    for kw in keywords.get('dosage', []):
        if re.search(kw, t, re.IGNORECASE):
            return 'dosage'

    # category
    for kw in keywords.get('category', []):
        if kw.lower() in t.lower():
            return 'category'

    # brand
    for kw in keywords.get('brand', []):
        if kw.lower() in t.lower():
            return 'brand'

    return 'unknown'


def make_mask(size: tuple, bboxes: list, expand_margin: int = 3) -> Image.Image:
    """生成白色掩码图（要修复的区域为白色）"""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        # 稍微扩大一圈，覆盖更干净
        draw.rectangle([x0 - expand_margin, y0 - expand_margin, x1 + expand_margin, y1 + expand_margin], fill=255)
    return mask


def merge_nearby_bboxes(bboxes: list, image_size: tuple, merge_distance: int = 30) -> list:
    """
    将距离小于 merge_distance 的 bbox 合并为更大的矩形框
    
    策略：
    1. 将每个 bbox 转为二值 mask
    2. 用形态学膨胀连接邻近区域
    3. 提取连通域作为新的合并区域
    
    Args:
        bboxes: 原始 bbox 列表 [[x0,y0,x1,y1], ...]
        image_size: (width, height) 图像尺寸
        merge_distance: 合并距离阈值（像素）
    
    Returns:
        合并后的 bbox 列表
    """
    if len(bboxes) <= 1:
        return bboxes
    
    w, h = image_size
    
    # 创建二值 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    
    # 形态学膨胀连接邻近区域
    kernel_size = merge_distance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # 闭运算填充小空洞
    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
    
    # 提取连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    merged_bboxes = []
    for i in range(1, num_labels):  # 跳过背景（label 0）
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 稍微收缩回原始范围（去除膨胀的影响）
        shrink = merge_distance // 2
        x0 = max(0, x + shrink)
        y0 = max(0, y + shrink)
        x1 = min(w, x + bw - shrink)
        y1 = min(h, y + bh - shrink)
        
        if x1 > x0 and y1 > y0:
            merged_bboxes.append([x0, y0, x1, y1])
    
    print(f"[MergeMasks] 原始区域: {len(bboxes)} 个，合并后: {len(merged_bboxes)} 个")
    return merged_bboxes


def should_merge_masks(bboxes: list, image_size: tuple) -> bool:
    """
    判断是否满足合并条件
    
    条件：
    1. 区域数量 >= MERGE_MASKS_MIN_AREAS (默认3个)
    2. 总区域面积占比 <= MERGE_MASKS_MAX_TOTAL_RATIO (默认30%)
    
    Returns:
        True - 应该合并
        False - 不应该合并，逐个处理
    """
    if len(bboxes) < MERGE_MASKS_MIN_AREAS:
        return False
    
    w, h = image_size
    total_area = w * h
    
    # 计算总消除面积（不重复计算重叠区域）
    mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    
    erase_area = np.sum(mask > 0)
    area_ratio = erase_area / total_area
    
    if area_ratio > MERGE_MASKS_MAX_TOTAL_RATIO:
        print(f"[MergeMasks] 总消除面积占比 {area_ratio:.1%} 超过阈值 {MERGE_MASKS_MAX_TOTAL_RATIO:.1%}，不合并")
        return False
    
    print(f"[MergeMasks] 满足合并条件: {len(bboxes)} 个区域，总占比 {area_ratio:.1%}")
    return True


def analyze_background_complexity(img: Image.Image, bbox: tuple) -> dict:
    """
    分析框选区域的背景复杂度
    
    Returns:
        {
            'is_uniform': bool,      # 是否是纯色背景
            'dominant_color': tuple,  # 主色调 (R, G, B)
            'color_variance': float,  # 颜色方差（越小越纯）
            'suggested_action': str   # 'fill' 或 'inpaint'
        }
    """
    x0, y0, x1, y1 = bbox
    # 确保坐标有效
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.width, int(x1)), min(img.height, int(y1))
    
    if x1 <= x0 or y1 <= y0:
        return {'is_uniform': False, 'dominant_color': (255, 255, 255), 'color_variance': 999, 'suggested_action': 'inpaint'}
    
    # 裁剪区域
    region = img.crop((x0, y0, x1, y1)).convert('RGB')
    arr = np.array(region)
    
    # 计算颜色分布
    # 方法：采样边缘像素作为背景参考（文字通常在中间）
    h, w = arr.shape[:2]
    edge_pixels = []
    
    # 采样边缘（上下左右各一圈）
    border = max(1, min(h, w) // 6)  # 边缘宽度
    edge_pixels.extend(arr[:border, :].reshape(-1, 3))  # 上
    edge_pixels.extend(arr[-border:, :].reshape(-1, 3))  # 下
    edge_pixels.extend(arr[:, :border].reshape(-1, 3))  # 左
    edge_pixels.extend(arr[:, -border:].reshape(-1, 3))  # 右
    
    edge_pixels = np.array(edge_pixels)
    
    # 计算主色调（中位数，更鲁棒）
    dominant_color = tuple(np.median(edge_pixels, axis=0).astype(int))
    
    # 计算颜色方差
    color_variance = np.mean(np.var(edge_pixels, axis=0))
    
    # 判断是否是纯色背景（方差小且颜色集中）
    # 同时检查最大最小值差异，避免渐变背景误判
    color_range = np.max(edge_pixels, axis=0) - np.min(edge_pixels, axis=0)
    max_range = np.max(color_range)
    
    is_uniform = (color_variance < 100) and (max_range < 20)  # 更严格条件，让更多区域走MI-GAN
    
    return {
        'is_uniform': is_uniform,
        'dominant_color': dominant_color,
        'color_variance': color_variance,
        'suggested_action': 'fill' if is_uniform else 'inpaint'
    }


def _calculate_mask_area_ratio(mask: Image.Image) -> float:
    """计算mask白色区域占图像总面积的比例"""
    mask_array = np.array(mask.convert('L'))
    white_pixels = np.sum(mask_array > 127)
    total_pixels = mask_array.size
    return white_pixels / total_pixels if total_pixels > 0 else 0


def _select_inpaint_model(img: Image.Image, mask: Image.Image) -> str:
    """
    智能选择修复模型
    
    策略：
    - 当消除区域面积 > 5% 图像面积时使用 LaMa（效果更好）
    - 否则使用 MI-GAN（速度快，保持原分辨率）
    
    Returns:
        'lama' 或 'migan'
    """
    area_ratio = _calculate_mask_area_ratio(mask)
    
    print(f"[Inpaint] 消除区域占比: {area_ratio:.2%} (阈值: {LAMA_AREA_THRESHOLD:.2%})")
    
    if area_ratio > LAMA_AREA_THRESHOLD:
        print(f"[Inpaint] 消除区域较大，选择 LaMa 模型")
        return 'lama'
    else:
        print(f"[Inpaint] 消除区域较小，选择 MI-GAN 模型")
        return 'migan'


def inpaint_image(img: Image.Image, mask: Image.Image, use_migan: bool = True) -> Image.Image:
    """修复被掩码标记的区域"""
    # 根据配置选择模型
    print(f"[Inpaint] 当前配置: {INPAINT_MODEL}")
    
    # 确定使用哪个模型
    if INPAINT_MODEL == 'auto':
        selected_model = _select_inpaint_model(img, mask)
    else:
        selected_model = INPAINT_MODEL
    
    # 执行修复
    if selected_model == 'lama':
        try:
            print("[Inpaint] 使用 LaMa 模型进行修复...")
            return inpaint_lama(img, mask)
        except Exception as e:
            print(f"[Inpaint] LaMa 修复失败，回退到 MI-GAN: {e}")
            try:
                return inpaint_migan(img, mask)
            except Exception as e2:
                print(f"[Inpaint] MI-GAN 也失败，回退到 OpenCV: {e2}")
                return inpaint_opencv(img, mask)
    elif selected_model == 'migan' or use_migan:
        try:
            print("[Inpaint] 使用 MI-GAN 模型进行修复...")
            return inpaint_migan(img, mask)
        except Exception as e:
            print(f"[Inpaint] MI-GAN 修复失败，回退到 OpenCV: {e}")
            return inpaint_opencv(img, mask)
    else:
        print("[Inpaint] 使用 OpenCV 进行修复...")
        return inpaint_opencv(img, mask)


def merge_inpaint_result(base_img: Image.Image, inpaint_result: Image.Image, mask: Image.Image, 
                         use_feathering: bool = True, feather_radius: int = 3) -> Image.Image:
    """
    合并修复结果到基础图像
    只取mask区域内的修复结果，其他区域保持基础图像不变
    
    支持羽化边缘，消除拼接痕迹
    
    Args:
        base_img: 基础图像（当前累积结果）
        inpaint_result: 修复结果（基于原图的某个区域修复结果）
        mask: 修复区域mask
        use_feathering: 是否使用羽化边缘
        feather_radius: 羽化半径（像素）
        
    Returns:
        合并后的图像
    """
    mask_array = np.array(mask.convert('L'))
    base_array = np.array(base_img)
    result_array = np.array(inpaint_result)
    
    # 是否使用羽化
    if use_feathering and FEATHERING_ENABLED and feather_radius > 0:
        # 高斯模糊实现羽化
        mask_blur = cv2.GaussianBlur(mask_array, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        mask_norm = mask_blur.astype(np.float32) / 255.0
        print(f"[Merge] 使用羽化合并，半径: {feather_radius}px")
    else:
        # 硬边缘
        mask_norm = mask_array.astype(np.float32) / 255.0
    
    # mask=255的区域使用修复结果，mask=0的区域保持基础图像
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    combined = (result_array * mask_3ch + base_array * (1 - mask_3ch)).astype(np.uint8)
    
    return Image.fromarray(combined, mode='RGB')


def inpaint_migan(img: Image.Image, mask: Image.Image, preserve_unmasked: bool = True, 
                  use_feathering: bool = True) -> Image.Image:
    """
    使用 MI-GAN 深度学习模型修复图像
    
    Args:
        img: 输入图像
        mask: 修复区域mask（白色表示需要修复）
        preserve_unmasked: 是否只更新mask区域，保留其他区域完全不变
        use_feathering: 是否使用羽化边缘合并
    """
    migan = get_migan()
    result = migan.inpaint(img, mask)
    
    if preserve_unmasked:
        # 只更新mask区域内的像素，其他区域保持原图不变
        return merge_inpaint_result(img, result, mask, use_feathering=use_feathering, 
                                   feather_radius=FEATHERING_RADIUS)
    
    return result


def inpaint_lama(img: Image.Image, mask: Image.Image, preserve_unmasked: bool = True,
                 use_feathering: bool = True) -> Image.Image:
    """
    使用 LaMa 深度学习模型修复图像
    
    Args:
        img: 输入图像
        mask: 修复区域mask（白色表示需要修复）
        preserve_unmasked: 是否只更新mask区域，保留其他区域完全不变（默认True）
        use_feathering: 是否使用羽化边缘合并
    """
    lama = get_lama()
    print(f"[LaMa] 开始修复，图像尺寸: {img.size}")
    result = lama.inpaint(img, mask)
    print(f"[LaMa] 修复完成")
    
    if preserve_unmasked:
        # 只更新mask区域内的像素，其他区域保持原图不变
        return merge_inpaint_result(img, result, mask, use_feathering=use_feathering,
                                   feather_radius=FEATHERING_RADIUS)
    
    return result


def inpaint_opencv(img: Image.Image, mask: Image.Image) -> Image.Image:
    """OpenCV Telea 算法备用修复"""
    src = pil_to_cv(img)
    msk = np.array(mask.convert("L"))
    result = cv2.inpaint(src, msk, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return cv_to_pil(result)


def sample_background_color(img: Image.Image, bbox: tuple) -> tuple:
    """
    采样文字区域周边背景色（取包围盒外圈均值）
    """
    x0, y0, x1, y1 = bbox
    w, h = img.width, img.height

    samples = []
    margin = 8
    regions = [
        (max(0, x0 - margin), max(0, y0 - margin), x0,              min(h, y1 + margin)),  # 左
        (min(w, x1),          max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)),  # 右
        (x0, max(0, y0 - margin), x1,              y0),              # 上
        (x0, y1,               x1,                 min(h, y1 + margin)),  # 下
    ]
    for r in regions:
        rx0, ry0, rx1, ry1 = [int(v) for v in r]
        if rx1 > rx0 and ry1 > ry0:
            patch = img.crop((rx0, ry0, rx1, ry1))
            arr = np.array(patch.convert("RGB")).reshape(-1, 3)
            if len(arr):
                samples.append(arr.mean(axis=0))

    if samples:
        mean = np.array(samples).mean(axis=0)
        return tuple(int(v) for v in mean)
    return (255, 255, 255)


def estimate_text_color(img: Image.Image, bbox: tuple) -> tuple:
    """估算文字本身的颜色（取包围盒内最常见非背景色）"""
    x0, y0, x1, y1 = bbox
    patch = img.crop((x0, y0, x1, y1)).convert("RGB")
    arr = np.array(patch).reshape(-1, 3)
    if len(arr) == 0:
        return (0, 0, 0)

    bg = np.array(sample_background_color(img, bbox))
    # 找与背景色差异最大的色
    dists = np.linalg.norm(arr.astype(float) - bg.astype(float), axis=1)
    idx = np.argmax(dists)
    color = arr[idx]
    return tuple(int(v) for v in color)


def detect_font_from_image(img: Image.Image, bbox: list) -> str:
    """
    通过图像分析自动识别字体类型
    基于文字的几何特征、笔画粗细、衬线等特征进行判断
    返回: 字体家族名称 (simhei, simsun, microsoft-yahei, kaiti, fangsong, arial, times, courier)
    """
    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        # 确保坐标有效
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, x1), min(img.height, y1)
        if x1 <= x0 or y1 <= y0:
            return 'microsoft-yahei'  # 默认返回微软雅黑
        
        # 裁剪文字区域
        text_region = img.crop((x0, y0, x1, y1))
        arr = np.array(text_region.convert('L'))  # 转灰度
        
        # 二值化
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 计算文字几何特征
        h, w = binary.shape
        
        # 1. 计算笔画粗细 (通过腐蚀/膨胀分析)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        stroke_thickness = np.sum(binary) / (np.sum(eroded) + 1)
        
        # 2. 检测衬线 (通过边缘分析)
        edges = cv2.Canny(binary, 50, 150)
        # 统计水平边缘和垂直边缘的比例
        h_edges = np.sum(edges[:, :w//4]) + np.sum(edges[:, -w//4:])
        v_edges = np.sum(edges[:h//4, :]) + np.sum(edges[-h//4:, :])
        serif_score = h_edges / (v_edges + 1)  # 衬线字体通常有更多水平边缘
        
        # 3. 计算宽高比
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找最大的轮廓
            largest = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest)
            aspect_ratio = rw / (rh + 1)
        else:
            aspect_ratio = w / h
        
        # 4. 分析笔画特征 (黑体 vs 宋体)
        # 黑体笔画均匀，宋体有粗细变化
        grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
        gradient_var = np.var(np.abs(grad_x)) + np.var(np.abs(grad_y))
        
        # 特征评分
        scores = {
            'simhei': 0,      # 黑体 - 无衬线，笔画均匀
            'simsun': 0,      # 宋体 - 有衬线，笔画有粗细
            'microsoft-yahei': 0,  # 微软雅黑 - 无衬线，现代
            'kaiti': 0,       # 楷体 - 手写风格
            'fangsong': 0,    # 仿宋 - 细长
            'arial': 0,       # Arial - 无衬线
            'times': 0,       # Times - 有衬线
            'courier': 0      # Courier - 等宽
        }
        
        # 根据特征评分
        # 衬线检测
        if serif_score > 0.5:
            scores['simsun'] += 2
            scores['times'] += 2
            scores['fangsong'] += 1
        else:
            scores['simhei'] += 2
            scores['microsoft-yahei'] += 2
            scores['arial'] += 2
        
        # 笔画粗细均匀度
        if gradient_var < 1000:  # 笔画均匀
            scores['simhei'] += 2
            scores['microsoft-yahei'] += 1
        else:  # 笔画有粗细变化
            scores['simsun'] += 1
            scores['kaiti'] += 1
        
        # 宽高比 (细长字体)
        if aspect_ratio > 1.5:
            scores['fangsong'] += 2
        
        # 等宽字体检测
        if 0.9 < aspect_ratio < 1.1 and stroke_thickness > 2:
            scores['courier'] += 2
        
        # 选择得分最高的字体
        best_font = max(scores, key=scores.get)
        print(f"[FontDetect] 检测到字体: {best_font} (衬线分:{serif_score:.2f}, 笔画均匀度:{gradient_var:.0f})")
        return best_font
        
    except Exception as e:
        print(f"[FontDetect] 字体识别失败: {e}, 使用默认字体")
        return 'microsoft-yahei'  # 默认返回微软雅黑


def get_font(size: int = 20, font_family: str = None) -> ImageFont.FreeTypeFont:
    """
    尝试加载系统字体，支持指定字体家族
    font_family: simhei, simsun, microsoft-yahei, kaiti, fangsong, arial, times, courier, segoe-ui-symbol
    
    特殊符号支持：®™©等符号需要字体支持，优先使用Segoe UI Symbol
    """
    # 字体映射 - 更新包含新字体和符号字体
    font_map = {
        'simhei': ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simhei.ttf'],
        'simsun': ['C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simsunb.ttf'],
        'microsoft-yahei': ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc'],
        'kaiti': ['C:/Windows/Fonts/simkai.ttf', 'C:/Windows/Fonts/simkai.ttf'],
        'fangsong': ['C:/Windows/Fonts/simfang.ttf', 'C:/Windows/Fonts/simfang.ttf'],
        'arial': ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'],
        'times': ['C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/timesbd.ttf'],
        'courier': ['C:/Windows/Fonts/cour.ttf', 'C:/Windows/Fonts/courbd.ttf'],
        'segoe-ui-symbol': ['C:/Windows/Fonts/seguisym.ttf', 'C:/Windows/Fonts/seguisym.ttf']
    }
    
    # 如果指定了字体家族，优先尝试
    if font_family and font_family in font_map:
        for fp in font_map[font_family]:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    pass
    
    # 默认字体候选 - 包含支持特殊符号的字体
    font_candidates = [
        "C:/Windows/Fonts/seguisym.ttf",  # Segoe UI Symbol - 支持®™©等特殊符号
        "C:/Windows/Fonts/arial.ttf",     # Arial - 对特殊符号支持较好
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def contains_special_symbols(text: str) -> bool:
    """检测文本是否包含特殊符号（如®™©等）"""
    special_symbols = ['®', '™', '©', '℗', '℠', '§', '¶', '†', '‡']
    return any(symbol in text for symbol in special_symbols)


def get_font_for_text(text: str, size: int, font_family: str = None) -> ImageFont.FreeTypeFont:
    """
    根据文本内容选择合适的字体
    默认使用 Segoe UI Symbol（支持所有商标符号 ©℗®™℠ 和中文）
    """
    # 默认优先使用 Segoe UI Symbol（支持所有商标符号 ©℗®™℠）
    try:
        if os.path.exists('C:/Windows/Fonts/seguisym.ttf'):
            return ImageFont.truetype('C:/Windows/Fonts/seguisym.ttf', size)
    except Exception:
        pass
    
    # 如果文本包含特殊符号，尝试其他兼容字体
    if contains_special_symbols(text):
        # fallback 到 Arial（支持®™©℠）
        try:
            if os.path.exists('C:/Windows/Fonts/arial.ttf'):
                return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
        except Exception:
            pass
        # 再尝试微软雅黑（支持®）
        try:
            if os.path.exists('C:/Windows/Fonts/msyh.ttc'):
                return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)
        except Exception:
            pass
        print(f"[Font] 警告：未找到支持特殊符号的字体，尝试使用默认字体")
    
    # 使用常规字体选择逻辑
    return get_font(size, font_family)


def get_symbol_font(size: int) -> ImageFont.FreeTypeFont:
    """获取支持商标符号的字体（Segoe UI Symbol）"""
    try:
        if os.path.exists('C:/Windows/Fonts/seguisym.ttf'):
            return ImageFont.truetype('C:/Windows/Fonts/seguisym.ttf', size)
    except Exception:
        pass
    # fallback
    return get_font(size)


def render_text_with_trademark(img: Image.Image, draw: ImageDraw.Draw, text: str, 
                                position: tuple, font_size: int, text_color: tuple,
                                font_family: str = None) -> None:
    """
    渲染带商标符号的文字，使用复合字体：
    - 中文部分使用微软雅黑/黑体
    - 商标符号使用 Segoe UI Symbol（缩小60%）
    例如：迈瑞葆®，其中®字号比文字小
    """
    x, y = position
    
    # 分离文字和商标符号
    trademark_symbols = ['®', '™', '©', '℠', '℗']
    
    # 找到商标符号位置
    trademark_idx = -1
    trademark_char = ''
    for i, char in enumerate(text):
        if char in trademark_symbols:
            trademark_idx = i
            trademark_char = char
            break
    
    # 如果没有商标符号，直接渲染
    if trademark_idx == -1:
        font = get_font(size, font_family)
        draw.text((x, y), text, fill=text_color, font=font)
        return
    
    # 分离主体文字和商标
    main_text = text[:trademark_idx]
    
    # 加载中文字体（微软雅黑优先）
    main_font = get_font(font_size, font_family or 'microsoft-yahei')
    
    # 商标符号使用 Segoe UI Symbol，字号缩小为60%
    trademark_font_size = max(8, int(font_size * 0.6))
    trademark_font = get_symbol_font(trademark_font_size)
    
    # 计算主体文字尺寸
    try:
        left, top, right, bottom = draw.textbbox((0, 0), main_text, font=main_font)
        main_width = right - left
        main_height = bottom - top
    except AttributeError:
        main_width, main_height = draw.textsize(main_text, font=main_font)
    
    # 计算商标符号尺寸
    try:
        left, top, right, bottom = draw.textbbox((0, 0), trademark_char, font=trademark_font)
        tm_height = bottom - top
    except AttributeError:
        _, tm_height = draw.textsize(trademark_char, font=trademark_font)
    
    # 渲染主体文字（中文）
    if main_text:
        draw.text((x, y), main_text, fill=text_color, font=main_font)
    
    # 渲染商标符号（右上角，稍微上移）
    tm_x = x + main_width
    tm_y = y - int(tm_height * 0.2)  # 稍微上移
    draw.text((tm_x, tm_y), trademark_char, fill=text_color, font=trademark_font)


def replace_text_in_place(img: Image.Image, items_to_replace: list, replace_text: str, font_family: str = None, font_color: str = None) -> Image.Image:
    """
    对每个 bbox：
    1. 分析背景复杂度，决定消除策略（纯色填充 vs MI-GAN修复）
    2. 消除原文字
    3. 在原位置用用户指定颜色/字号渲染 replace_text
    支持指定字体家族，支持自动识别字体 (font_family='auto')
    支持指定字体颜色 (font_color='#RRGGBB' 或 'auto')
    默认字体：宋体(simsun)，默认颜色：黑色
    自动检测特殊符号（®™©等）并选择合适的字体
    """
    if not items_to_replace:
        return img

    # 调试：打印接收到的替换文本
    print(f"[Replace] 接收到的替换文本: {repr(replace_text)}")
    print(f"[Replace] 文本长度: {len(replace_text) if replace_text else 0}")
    print(f"[Replace] 是否含特殊符号: {contains_special_symbols(replace_text) if replace_text else False}")

    draw = ImageDraw.Draw(img)

    for item in items_to_replace:
        bbox = item['bbox']
        x0, y0, x1, y1 = bbox
        bw = x1 - x0
        bh = y1 - y0

        # 注意：外层已经用MI-GAN消除了背景，这里直接写文字即可

        # 估算字体大小（让替换文字占满包围盒高度的 ~80%）
        font_size = max(8, int(bh * 0.8))

        # 确定文字颜色 - 默认黑色，优先使用用户指定的颜色
        text_color = (0, 0, 0)  # 默认黑色
        if font_color and font_color != 'auto':
            # 使用用户指定的颜色
            try:
                # 解析 #RRGGBB 格式
                if font_color.startswith('#'):
                    text_color = tuple(int(font_color[i:i+2], 16) for i in (1, 3, 5))
            except:
                text_color = (0, 0, 0)  # 解析失败默认黑色
        # 注意：不再使用 estimate_text_color，避免颜色变成背景色

        # 确定使用的字体 - 根据文本内容自动选择
        actual_font_family = 'simsun'  # 默认宋体
        if font_family == 'auto':
            # 自动识别字体
            actual_font_family = detect_font_from_image(img, bbox)
            print(f"[Replace] 自动识别字体: {actual_font_family}")
        elif font_family and font_family != 'auto':
            # 使用用户指定的字体
            actual_font_family = font_family
        
        # 计算居中位置
        # 如果包含商标符号，使用特殊渲染（商标缩小）
        if contains_special_symbols(replace_text):
            print(f"[Replace] 检测到特殊符号，使用商标缩小渲染")
            # 先计算完整文字尺寸（用于居中定位）
            temp_font = get_font_for_text(replace_text, font_size, actual_font_family)
            try:
                left, top, right, bottom = draw.textbbox((0, 0), replace_text, font=temp_font)
                tw = right - left
                th = bottom - top
            except AttributeError:
                tw, th = draw.textsize(replace_text, font=temp_font)
            
            tx = x0 + (bw - tw) // 2
            ty = y0 + (bh - th) // 2
            
            # 使用商标缩小渲染
            render_text_with_trademark(img, draw, replace_text, (tx, ty), font_size, text_color, actual_font_family)
        else:
            # 普通文字渲染
            font = get_font_for_text(replace_text, font_size, actual_font_family)
            try:
                left, top, right, bottom = draw.textbbox((0, 0), replace_text, font=font)
                tw = right - left
                th = bottom - top
            except AttributeError:
                tw, th = draw.textsize(replace_text, font=font)
            
            tx = x0 + (bw - tw) // 2
            ty = y0 + (bh - th) // 2
            
            # 步骤2: 覆盖新文字
            draw.text((tx, ty), replace_text, fill=text_color, font=font)
        
        font_name = actual_font_family if not contains_special_symbols(replace_text) else 'microsoft-yahei-with-tm'
        print(f"[Replace] 步骤2: 在 ({x0},{y0})-({x1},{y1}) 覆盖文字 '{replace_text}'，字体: {font_name}, 颜色: {text_color}")

    return img


# ─────────────────────────────────────────────
# 核心处理入口
# ─────────────────────────────────────────────

def process_product_image(img: Image.Image, options: dict) -> Image.Image:
    """
    options 字段：
      remove_dosage   bool
      remove_category bool
      replace_brand   bool
      replace_text    str   （默认 'xxx'）
      inpaint_strength int  （1-5，暂用于控制mask扩展大小）
      keywords        dict  {'dosage':[], 'category':[], 'brand':[], 'keep':[]}
    """
    remove_dosage   = options.get('remove_dosage', True)
    remove_category = options.get('remove_category', True)
    replace_brand   = options.get('replace_brand', True)
    replace_text    = options.get('replace_text', 'xxx')
    keywords        = options.get('keywords', {})

    # ── 1. OCR ──
    print("[Process] 开始 OCR 识别...")
    ocr_items = run_ocr(img)
    print(f"[Process] 识别到 {len(ocr_items)} 个文字块")

    # ── 2. 分类 ──
    to_erase   = []   # bbox 列表，直接 inpaint
    to_replace = []   # item 列表，inpaint + 渲染替换

    for item in ocr_items:
        cls = classify_text(item['text'], keywords)
        print(f"  [{cls:10s}] {item['text']!r}")
        if cls == 'keep' or cls == 'unknown':
            continue
        if cls in ('dosage', 'category'):
            if (cls == 'dosage' and remove_dosage) or (cls == 'category' and remove_category):
                to_erase.append(item['bbox'])
        elif cls == 'brand' and replace_brand:
            to_replace.append(item)

    # ── 3. Inpaint 消除（含量/类型） ──
    if to_erase:
        print(f"[Process] Inpaint 消除 {len(to_erase)} 个区域...")
        mask = make_mask(img.size, to_erase)
        img = inpaint_image(img, mask)
        print("[Process] Inpaint 完成")

    # ── 4. 商标替换为 xxx ──
    if to_replace:
        print(f"[Process] 商标替换 {len(to_replace)} 个文字...")
        img = replace_text_in_place(img, to_replace, replace_text)
        print("[Process] 商标替换完成")

    return img


# ─────────────────────────────────────────────
# Flask 路由
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': '商品图片净化服务运行中'})


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': '未收到图片文件'}), 400

    file = request.files['file']
    options_raw = request.form.get('options', '{}')
    try:
        options = json.loads(options_raw)
    except Exception:
        options = {}

    try:
        img = Image.open(file.stream).convert("RGBA")
        # 保留透明度信息，处理时转RGB，最后合并回来
        has_alpha = img.mode == 'RGBA'
        alpha = img.split()[3] if has_alpha else None
        rgb_img = img.convert("RGB")

        result = process_product_image(rgb_img, options)

        # 还原透明度
        if has_alpha and alpha:
            result = result.convert("RGBA")
            result.putalpha(alpha)

        # 输出
        buf = io.BytesIO()
        fmt = 'PNG' if has_alpha else 'JPEG'
        result.save(buf, format=fmt, quality=95)
        buf.seek(0)

        return send_file(
            buf,
            mimetype='image/png' if has_alpha else 'image/jpeg',
            as_attachment=False
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/ocr_preview', methods=['POST'])
def ocr_preview():
    """
    仅返回 OCR + 分类结果（用于预览，不修改图片）
    """
    if 'file' not in request.files:
        return jsonify({'error': '未收到图片文件'}), 400

    file = request.files['file']
    options_raw = request.form.get('options', '{}')
    try:
        options = json.loads(options_raw)
    except Exception:
        options = {}

    try:
        img = Image.open(file.stream).convert("RGB")
        ocr_items = run_ocr(img)
        keywords = options.get('keywords', {})

        results = []
        for item in ocr_items:
            cls = classify_text(item['text'], keywords)
            results.append({
                'text': item['text'],
                'bbox': item['bbox'],
                'confidence': round(item['confidence'], 3),
                'category': cls
            })

        return jsonify({'items': results, 'total': len(results)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def decode_mask_data(mask_data: str, size: tuple) -> Image.Image:
    """解码前端传来的 base64 mask 图片数据"""
    if not mask_data:
        return None
    try:
        # data:image/png;base64,xxx
        if ',' in mask_data:
            mask_data = mask_data.split(',')[1]
        mask_bytes = base64.b64decode(mask_data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
        # 确保尺寸匹配
        if mask_img.size != size:
            mask_img = mask_img.resize(size, Image.NEAREST)
        return mask_img
    except Exception as e:
        print(f"[MaskDecode] 解码失败: {e}")
        return None


def process_cutout(img: Image.Image, cutout_data: dict, replace_text: str) -> Image.Image:
    """
    处理抠图：
    1. 提取主体区域
    2. 对原图主体区域进行背景填充（Inpaint）
    3. 将主体重新贴回（可选：保持原位或根据用户调整）
    返回处理后的图片
    """
    if not cutout_data:
        return img
    
    bbox = cutout_data.get('bbox', [0, 0, 100, 100])
    x0, y0, x1, y1 = [int(v) for v in bbox]
    
    print(f"[Cutout] 处理抠图区域: ({x0},{y0})-({x1},{y1})")
    
    # 确保坐标在图片范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    
    if x1 <= x0 or y1 <= y0:
        print("[Cutout] 无效区域，跳过")
        return img
    
    # 1. 提取主体
    subject = img.crop((x0, y0, x1, y1))
    
    # 2. 创建mask（主体区域为白色，需要修复）
    mask = Image.new('L', img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    # 稍微扩大一点mask边缘，让修复更自然
    margin = 3
    mask_draw.rectangle([x0-margin, y0-margin, x1+margin, y1+margin], fill=255)
    
    # 3. 对原图进行背景填充（消除主体）
    result = inpaint_image(img, mask)
    print("[Cutout] 背景填充完成")
    
    # 4. 将主体贴回（目前贴回原位，后续可支持位置调整）
    result.paste(subject, (x0, y0))
    print("[Cutout] 主体贴回完成")
    
    return result


@app.route('/process_with_annotations', methods=['POST'])
def process_with_annotations():
    """
    按前端传来的标注框坐标精确处理图片。
    支持画笔涂抹mask、抠图模式、保存训练数据、字体选择。

    annotations 格式：
      [{ "label": "erase"|"brand"|"cutout", "bbox": [x0,y0,x1,y1], "text": "..." }, ...]
    cutout_data: 抠图数据 {bbox, subject: base64, operation}
    mask_data: base64编码的PNG mask图（可选，画笔涂抹区域）
    font_family: 替换字体 (simhei, simsun, microsoft-yahei, arial, times, courier)
    zoom_scale: 前端缩放比例（用于精确计算）
    """
    if 'file' not in request.files:
        return jsonify({'error': '未收到图片文件'}), 400

    file = request.files['file']
    options_raw = request.form.get('options', '{}')
    try:
        options = json.loads(options_raw)
    except Exception:
        options = {}

    annotations    = options.get('annotations', [])
    replace_text   = options.get('replace_text', 'xxx')
    font_family    = options.get('font_family', None)  # 新增：字体选择
    font_color     = options.get('font_color', None)   # 新增：字体颜色
    save_training  = options.get('save_training', False)
    filename       = options.get('filename', 'image.jpg')
    mask_data      = options.get('mask_data', None)
    cutout_data    = options.get('cutout_data', None)
    zoom_scale     = options.get('zoom_scale', 1.0)    # 新增：缩放比例

    try:
        pil_img = Image.open(file.stream)
        has_alpha = pil_img.mode == 'RGBA'
        alpha = pil_img.split()[3] if has_alpha else None
        img = pil_img.convert("RGB")

        # 读取原图字节（用于训练数据保存）
        file.stream.seek(0)
        orig_bytes = file.stream.read()

        print(f"[AnnotProcess] 图片: {filename}，标注 {len(annotations)} 个，缩放: {zoom_scale:.2f}x")

        # ── 分类标注 ──
        erase_bboxes  = []   # 直接 inpaint 消除（背景填充）
        replace_items = []   # inpaint + 渲染 replace_text
        replace_bboxes = []  # 替换区域的bbox（用于背景消除）
        cutout_bboxes = []   # 抠图区域

        for anno in annotations:
            label = anno.get('label', 'erase')
            bbox  = anno.get('bbox', [])
            text  = anno.get('text', '')
            if len(bbox) < 4:
                continue
            bbox = [int(v) for v in bbox]

            if label == 'cutout':
                cutout_bboxes.append(bbox)
                print(f"  [抠图] bbox={bbox}")
            elif label in ('erase', 'manual_erase'):
                erase_bboxes.append(bbox)
                print(f"  [消除-背景填充] {text!r} bbox={bbox}")
            elif label == 'brand':
                replace_items.append({'bbox': bbox, 'text': text})
                replace_bboxes.append(bbox)  # 替换区域也需要背景消除
                print(f"  [替换→{replace_text}] {text!r} bbox={bbox}")

        # ── 处理抠图（优先处理）──
        if cutout_data and cutout_data.get('bbox'):
            img = process_cutout(img, cutout_data, replace_text)
        elif cutout_bboxes:
            # 如果没有cutout_data但有cutout标注，使用第一个
            img = process_cutout(img, {'bbox': cutout_bboxes[0]}, replace_text)

        # ── 步骤1: 分别处理消除区域和替换区域（使用不同模型）──
        # 🔴 消除区域使用 MI-GAN（小模型，速度快）- 基于原图分别处理，然后合并结果
        if erase_bboxes:
            print(f"[AnnotProcess] 🔴 消除区域共 {len(erase_bboxes)} 处")
            
            # 判断是否合并处理
            should_merge = MERGE_MASKS_ENABLED and should_merge_masks(erase_bboxes, img.size)
            
            if should_merge and len(erase_bboxes) >= MERGE_MASKS_MIN_AREAS:
                # 策略一：合并邻近mask，用LaMa一次性修复
                print(f"[AnnotProcess] 🔴 合并 {len(erase_bboxes)} 个邻近区域，使用 LaMa 统一修复")
                merged_bboxes = merge_nearby_bboxes(erase_bboxes, img.size, MERGE_MASKS_DISTANCE)
                
                if len(merged_bboxes) < len(erase_bboxes):
                    # 合并成功，使用LaMa处理
                    for i, merged_bbox in enumerate(merged_bboxes):
                        merged_mask = make_mask(img.size, [merged_bbox], expand_margin=5)
                        print(f"[AnnotProcess] 🔴 处理合并区域 {i+1}/{len(merged_bboxes)}: {merged_bbox}")
                        img = inpaint_lama(img, merged_mask, preserve_unmasked=True, use_feathering=True)
                    print(f"[AnnotProcess] 🔴 合并区域消除完成")
                else:
                    # 合并未减少区域数，回退到逐个处理
                    should_merge = False
            
            if not should_merge or len(erase_bboxes) < MERGE_MASKS_MIN_AREAS:
                # 策略二：逐个处理（小区域或分散区域）
                print(f"[AnnotProcess] 🔴 逐个处理消除区域（基于原图）")
                original_img = img.copy()  # 保存原图
                for i, bbox in enumerate(erase_bboxes):
                    single_mask = make_mask(img.size, [bbox])
                    print(f"[AnnotProcess] 🔴 处理消除区域 {i+1}/{len(erase_bboxes)}: {bbox}")
                    # 始终基于原图处理，避免累积误差
                    result = inpaint_migan(original_img, single_mask, preserve_unmasked=True, use_feathering=True)
                    # 只取mask区域的修复结果，合并到当前结果
                    img = merge_inpaint_result(img, result, single_mask, use_feathering=True, 
                                              feather_radius=FEATHERING_RADIUS)
                print(f"[AnnotProcess] 🔴 消除完成")
        
        # 合并画笔涂抹的mask（如果有）
        if mask_data:
            brush_mask = decode_mask_data(mask_data, img.size)
            if brush_mask:
                print(f"[AnnotProcess] 处理画笔涂抹区域")
                # 根据mask面积智能选择模型
                brush_area_ratio = _calculate_mask_area_ratio(brush_mask)
                if brush_area_ratio > LAMA_AREA_THRESHOLD:
                    print(f"[AnnotProcess] 画笔区域占比 {brush_area_ratio:.1%}，使用 LaMa")
                    img = inpaint_lama(img, brush_mask, preserve_unmasked=True, use_feathering=True)
                else:
                    print(f"[AnnotProcess] 画笔区域占比 {brush_area_ratio:.1%}，使用 MI-GAN")
                    img = inpaint_migan(img, brush_mask, preserve_unmasked=True, use_feathering=True)
        
        # 🟣 替换区域使用 LaMa（大模型，效果更好）
        if replace_bboxes:
            print(f"[AnnotProcess] 🟣 替换区域共 {len(replace_bboxes)} 处")
            
            # 策略四：批量合并替换区域，一次性消除所有背景
            if len(replace_bboxes) > 1:
                print(f"[AnnotProcess] 🟣 合并所有替换区域，一次性消除背景")
                combined_mask = make_mask(img.size, replace_bboxes, expand_margin=5)
                img = inpaint_lama(img, combined_mask, preserve_unmasked=True, use_feathering=True)
                print(f"[AnnotProcess] 🟣 替换区域背景消除完成（批量处理）")
            else:
                # 单个区域直接处理
                single_mask = make_mask(img.size, replace_bboxes, expand_margin=5)
                print(f"[AnnotProcess] 🟣 处理替换区域: {replace_bboxes[0]}")
                img = inpaint_lama(img, single_mask, preserve_unmasked=True, use_feathering=True)
                print(f"[AnnotProcess] 🟣 替换区域背景消除完成")

        # ── 步骤2: 覆盖替换文字 ──
        if replace_items:
            # 使用指定的字体和颜色（默认宋体+黑色）
            img = replace_text_in_place(img, replace_items, replace_text, font_family, font_color)
            print(f"[AnnotProcess] 步骤2完成: 覆盖替换文字，共 {len(replace_items)} 处")

        # ── 还原透明度 ──
        if has_alpha and alpha:
            img = img.convert("RGBA")
            img.putalpha(alpha)

        # ── 保存训练数据 ──
        if save_training and (annotations or mask_data or cutout_data):
            save_training_data(orig_bytes, img, annotations, filename, replace_text, mask_data, cutout_data)

        # ── 返回处理结果 ──
        buf = io.BytesIO()
        fmt = 'PNG' if has_alpha else 'JPEG'
        img.save(buf, format=fmt, quality=95)
        buf.seek(0)

        return send_file(buf, mimetype='image/png' if has_alpha else 'image/jpeg', as_attachment=False)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def process_subject_cutout(img: Image.Image, subject_bbox: list, bg_type: str = 'transparent', 
                           other_annotations: list = None, replace_text: str = 'xxx') -> Image.Image:
    """
    主体抠图处理：
    1. 提取主体
    2. 根据bg_type处理背景：
       - transparent: 透明背景
       - white: 白色背景
       - blur: 模糊背景
       - inpaint: 智能填充背景
    3. 合并主体和新背景
    4. 处理其他标注（消除/替换）
    """
    x0, y0, x1, y1 = [int(v) for v in subject_bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    
    if x1 <= x0 or y1 <= y0:
        raise ValueError("无效的主体区域")
    
    print(f"[Cutout] 主体区域: ({x0},{y0})-({x1},{y1}), 背景类型: {bg_type}")
    
    # 提取主体
    subject = img.crop((x0, y0, x1, y1))
    
    # 创建新背景
    if bg_type == 'transparent':
        # 透明背景
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(subject, (x0, y0))
        
    elif bg_type == 'white':
        # 白色背景
        result = Image.new('RGB', img.size, (255, 255, 255))
        result.paste(subject, (x0, y0))
        
    elif bg_type == 'blur':
        # 模糊背景
        bg = img.filter(ImageFilter.GaussianBlur(radius=20))
        # 在主体位置贴回原图（不模糊主体）
        bg.paste(subject, (x0, y0))
        result = bg
        
    elif bg_type == 'inpaint':
        # 智能填充背景
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x0, y0, x1, y1], fill=255)
        # 反转mask（背景是要修复的区域）
        mask = Image.fromarray(255 - np.array(mask))
        result = inpaint_image(img, mask)
        # 贴回主体（确保清晰）
        result.paste(subject, (x0, y0))
        
    else:
        result = img.copy()
    
    # 处理其他标注（在主体之外的区域）
    if other_annotations:
        erase_bboxes = []
        replace_items = []
        
        for anno in other_annotations:
            label = anno.get('label', '')
            bbox = anno.get('bbox', [])
            text = anno.get('text', '')
            if len(bbox) < 4:
                continue
            bbox = [int(v) for v in bbox]
            
            if label in ('erase', 'manual_erase'):
                erase_bboxes.append(bbox)
            elif label == 'brand':
                replace_items.append({'bbox': bbox, 'text': text})
        
        if erase_bboxes:
            mask = make_mask(result.size, erase_bboxes)
            result = inpaint_image(result, mask)
        
        if replace_items:
            result = replace_text_in_place(result, replace_items, replace_text)
    
    return result


def save_training_data(orig_bytes: bytes, result_img: Image.Image, annotations: list, filename: str, 
                       replace_text: str, mask_data: str = None, cutout_data: dict = None, 
                       subject_bbox: list = None):
    """
    保存训练数据到 training_data/ 目录
    """
    import datetime
    from pathlib import Path

    base_dir = Path(__file__).parent.parent / 'training_data'
    base_dir.mkdir(exist_ok=True)

    stem = Path(filename).stem
    ext  = Path(filename).suffix or '.jpg'
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{stem}_{ts}"

    # 原图
    orig_path = base_dir / f"{prefix}_orig{ext}"
    with open(orig_path, 'wb') as f:
        f.write(orig_bytes)

    # 处理后图片
    cleaned_path = base_dir / f"{prefix}_cleaned{ext}"
    has_alpha = result_img.mode == 'RGBA'
    fmt = 'PNG' if has_alpha else 'JPEG'
    result_img.save(str(cleaned_path), format=fmt, quality=95)

    # 保存mask（如有）
    mask_filename = None
    if mask_data:
        try:
            if ',' in mask_data:
                mask_bytes = base64.b64decode(mask_data.split(',')[1])
            else:
                mask_bytes = base64.b64decode(mask_data)
            mask_path = base_dir / f"{prefix}_mask.png"
            with open(mask_path, 'wb') as f:
                f.write(mask_bytes)
            mask_filename = f"{prefix}_mask.png"
        except Exception as e:
            print(f"[TrainData] mask保存失败: {e}")

    # 标注 JSON
    anno_path = base_dir / f"{prefix}_anno.json"
    anno_data = {
        "filename": filename,
        "timestamp": ts,
        "replace_text": replace_text,
        "annotations": annotations,
        "subject_bbox": subject_bbox,
        "orig_image": f"{prefix}_orig{ext}",
        "cleaned_image": f"{prefix}_cleaned{ext}",
        "mask_image": mask_filename,
    }
    with open(anno_path, 'w', encoding='utf-8') as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    print(f"[TrainData] 已保存: {prefix}_*.json/.jpg")


@app.route('/detect_subject', methods=['POST'])
def detect_subject():
    """
    自动识别图片主体区域
    返回: {bbox: [x0,y0,x1,y1], confidence: float}
    """
    if 'file' not in request.files:
        return jsonify({'error': '未收到图片文件'}), 400

    file = request.files['file']
    options_raw = request.form.get('options', '{}')
    try:
        options = json.loads(options_raw)
    except Exception:
        options = {}

    use_ocr = options.get('use_ocr', True)
    use_edge = options.get('use_edge', True)

    try:
        img = Image.open(file.stream).convert("RGB")
        result = detect_subject_auto(img, use_ocr=use_ocr, use_edge=use_edge)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/process_subject_cutout', methods=['POST'])
def process_subject_cutout():
    """
    主体抠图处理API
    保留主体，根据bg_type处理背景
    
    options:
      subject_bbox: [x0,y0,x1,y1] 主体区域
      bg_type: 'transparent'|'white'|'blur'|'inpaint' 背景处理方式
      annotations: 其他标注（消除/替换）
      replace_text: 替换文字
      save_training: 是否保存训练数据
      filename: 原始文件名
    """
    if 'file' not in request.files:
        return jsonify({'error': '未收到图片文件'}), 400

    file = request.files['file']
    options_raw = request.form.get('options', '{}')
    try:
        options = json.loads(options_raw)
    except Exception:
        options = {}

    subject_bbox = options.get('subject_bbox')
    bg_type = options.get('bg_type', 'transparent')
    other_annotations = options.get('annotations', [])
    replace_text = options.get('replace_text', 'xxx')
    save_training = options.get('save_training', False)
    filename = options.get('filename', 'image.jpg')

    if not subject_bbox or len(subject_bbox) < 4:
        return jsonify({'error': '未提供主体区域'}), 400

    try:
        pil_img = Image.open(file.stream)
        has_alpha = pil_img.mode == 'RGBA'
        img = pil_img.convert("RGB")

        # 读取原图字节
        file.stream.seek(0)
        orig_bytes = file.stream.read()

        print(f"[SubjectCutout] 处理: {filename}, 主体: {subject_bbox}, 背景: {bg_type}")

        # 执行抠图处理
        result = process_subject_cutout(img, subject_bbox, bg_type, other_annotations, replace_text)

        # 保存训练数据
        if save_training:
            save_training_data(orig_bytes, result, other_annotations, filename, 
                             replace_text, subject_bbox=subject_bbox)

        # 返回结果
        buf = io.BytesIO()
        fmt = 'PNG' if bg_type == 'transparent' or has_alpha else 'JPEG'
        result.save(buf, format=fmt, quality=95)
        buf.seek(0)

        return send_file(buf, mimetype='image/png' if fmt == 'PNG' else 'image/jpeg', as_attachment=False)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """直接通过Flask访问前端页面"""
    frontend = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    frontend = os.path.abspath(frontend)
    if os.path.exists(frontend):
        with open(frontend, 'r', encoding='utf-8') as f:
            content = f.read()
        from flask import Response
        return Response(content, mimetype='text/html')
    return "前端文件未找到", 404


if __name__ == '__main__':
    print("=" * 50)
    print("  商品图片净化工具 - 后端服务")
    print("  访问前端页面: http://127.0.0.1:5000")
    print("  API健康检查: http://127.0.0.1:5000/health")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
