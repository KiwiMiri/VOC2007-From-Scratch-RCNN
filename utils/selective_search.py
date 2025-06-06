import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def selective_search(image, mode='fast'):
    """
    Selective Search를 사용하여 객체 proposal을 생성합니다.
    
    Args:
        image: PIL Image 또는 numpy array
        mode: 'fast' 또는 'quality'
    
    Returns:
        regions: 바운딩 박스 리스트 (x, y, w, h)
    """
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # OpenCV 형식으로 변환 (RGB -> BGR)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # Selective Search 실행
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        
        if mode == 'fast':
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSelectiveSearchQuality()
        
        # Region proposals 생성
        rects = ss.process()
        
        # 결과를 (x, y, w, h) 형식으로 반환
        return rects
    except Exception as e:
        print(f"Selective Search error: {e}")
        print("Using simple sliding window as fallback...")
        return simple_sliding_window(image)

def simple_sliding_window(image, min_size=50, step_size=30):
    """
    간단한 슬라이딩 윈도우로 proposal 생성 (fallback 방법)
    """
    h, w = image.shape[:2]
    proposals = []
    
    # 다양한 크기의 윈도우
    sizes = [min_size, min_size*2, min_size*3, min_size*4]
    
    for size in sizes:
        for y in range(0, h - size, step_size):
            for x in range(0, w - size, step_size):
                proposals.append((x, y, size, size))
    
    return np.array(proposals)

def compute_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        iou: IoU 값
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 넓이
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 영역
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Non-Maximum Suppression을 수행합니다.
    
    Args:
        boxes: 바운딩 박스 리스트 [[x1, y1, x2, y2], ...]
        scores: 각 박스의 점수
        threshold: NMS 임계값
    
    Returns:
        kept_indices: 유지할 박스의 인덱스
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 점수에 따라 내림차순 정렬
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # 가장 높은 점수의 박스 선택
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 나머지 박스들과 IoU 계산
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = []
        for box in remaining_boxes:
            iou = compute_iou(current_box, box)
            ious.append(iou)
        
        ious = np.array(ious)
        
        # IoU가 임계값보다 낮은 박스들만 유지
        indices = indices[1:][ious < threshold]
    
    return keep

def visualize_proposals(image, proposals, max_proposals=50):
    """
    이미지에 region proposals를 시각화합니다.
    
    Args:
        image: PIL Image
        proposals: region proposals [(x, y, w, h), ...]
        max_proposals: 표시할 최대 proposal 수
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 최대 개수만큼만 표시
    proposals_to_show = proposals[:max_proposals]
    
    for i, (x, y, w, h) in enumerate(proposals_to_show):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor='red',
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)
    
    ax.set_title(f'Region Proposals (showing {len(proposals_to_show)} of {len(proposals)})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def convert_proposals_to_xyxy(proposals):
    """
    (x, y, w, h) 형식을 (x1, y1, x2, y2) 형식으로 변환합니다.
    """
    converted = []
    for x, y, w, h in proposals:
        converted.append([x, y, x + w, y + h])
    return converted

def filter_proposals(proposals, min_size=20, max_size=None, image_shape=None):
    """
    크기 기준으로 proposals를 필터링합니다.
    
    Args:
        proposals: [(x, y, w, h), ...] 형식의 proposals
        min_size: 최소 크기
        max_size: 최대 크기 (None이면 이미지 크기의 80%)
        image_shape: (height, width) 형식의 이미지 크기
    
    Returns:
        filtered_proposals: 필터링된 proposals
    """
    if max_size is None and image_shape is not None:
        max_size = min(image_shape[0], image_shape[1]) * 0.8
    
    filtered = []
    for x, y, w, h in proposals:
        # 크기 조건 확인
        if w < min_size or h < min_size:
            continue
        if max_size is not None and (w > max_size or h > max_size):
            continue
        
        # 이미지 경계 내에 있는지 확인
        if image_shape is not None:
            height, width = image_shape[:2]
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue
        
        filtered.append((x, y, w, h))
    
    return filtered