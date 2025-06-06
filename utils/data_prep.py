import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.selective_search import selective_search, compute_iou, convert_proposals_to_xyxy, filter_proposals
import warnings
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class RCNNDataPreprocessor:
    """R-CNN 훈련을 위한 데이터 전처리기"""
    
    def __init__(self, input_size=224, positive_threshold=0.5, negative_threshold=0.1):
        self.input_size = input_size
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def prepare_training_data(self, image, gt_boxes, gt_labels, max_proposals=2000, 
                            positive_ratio=0.25):
        """
        훈련용 데이터를 준비합니다.
        
        Args:
            image: PIL Image
            gt_boxes: Ground truth boxes [[x1, y1, x2, y2], ...]
            gt_labels: Ground truth labels [label1, label2, ...]
            max_proposals: 최대 proposal 수
            positive_ratio: positive sample 비율
        
        Returns:
            regions: 추출된 region images
            labels: 각 region의 라벨
            bbox_targets: 바운딩 박스 회귀 타겟
            bbox_masks: 회귀 손실 계산용 마스크
        """
        try:
            # 입력 검증
            if image is None:
                raise ValueError("Image is None")

            # 이미지 크기 검증    
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError(f"Image too small: {image.size}")
                
            # Ground truth 데이터 검증 및 정규화
            if not isinstance(gt_boxes, np.ndarray):
                gt_boxes = np.array(gt_boxes if gt_boxes is not None else [[0,0,1,1]])
            if not isinstance(gt_labels, np.ndarray):
                gt_labels = np.array(gt_labels if gt_labels is not None else [0])
                
            # 이미지 배열 변환
            try:
                image_array = np.array(image)
            except Exception as e:
                raise ValueError(f"Failed to convert image to array: {e}")
                
            # Selective Search로 region proposals 생성
            try:
                proposals = selective_search(image, mode='fast')
                if len(proposals) == 0:
                    raise ValueError("No proposals generated")
            except Exception as e:
                raise ValueError(f"Selective search failed: {e}")
            
            # Proposals 필터링
            try:
                proposals = filter_proposals(
                    proposals, 
                    min_size=20, 
                    image_shape=image_array.shape
                )
                if len(proposals) == 0:
                    raise ValueError("No proposals after filtering")
            except Exception as e:
                raise ValueError(f"Proposal filtering failed: {e}")
                
            # (x, y, w, h) -> (x1, y1, x2, y2) 변환
            try:
                proposals_xyxy = convert_proposals_to_xyxy(proposals)
            except Exception as e:
                raise ValueError(f"Failed to convert proposals to xyxy format: {e}")
            
            # IoU 계산으로 positive/negative 샘플 생성
            try:
                labels, bbox_targets, bbox_masks = self._assign_labels(
                    proposals_xyxy, gt_boxes, gt_labels
                )
            except Exception as e:
                raise ValueError(f"Label assignment failed: {e}")
            
            # 샘플링 (positive : negative = 1:3 비율)
            try:
                positive_indices = np.where(labels > 0)[0]
                negative_indices = np.where(labels == 0)[0]
                
                # Positive 샘플 수 결정 (최소 1개는 보장)
                num_positive = max(1, min(len(positive_indices), int(max_proposals * positive_ratio)))
                num_negative = min(len(negative_indices), max_proposals - num_positive)
                
                # 랜덤 샘플링
                if len(positive_indices) > 0:
                    positive_indices = np.random.choice(
                        positive_indices, num_positive, replace=True
                    )
                else:
                    # positive 샘플이 없으면 dummy positive 생성
                    warnings.warn("No positive samples, using dummy positive")
                    positive_indices = np.array([0], dtype=np.int32)
                    labels[0] = 1
                    bbox_targets[0] = np.array([0, 0, 0, 0])
                    bbox_masks[0] = True
                
                if num_negative > 0:
                    negative_indices = np.random.choice(
                        negative_indices, num_negative, replace=True
                    )
                else:
                    negative_indices = np.array([], dtype=np.int32)
                
            except Exception as e:
                raise ValueError(f"Sampling failed: {e}")
            
            # 선택된 인덱스들
            selected_indices = np.concatenate([positive_indices, negative_indices])
            
            # Region 이미지 추출
            regions = []
            valid_indices = []
            
            for i, idx in enumerate(selected_indices):
                try:
                    x1, y1, x2, y2 = proposals_xyxy[idx]
                    # 좌표 유효성 검증
                    if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and \
                       x2 <= image.size[0] and y2 <= image.size[1]:
                        region = image.crop((x1, y1, x2, y2))
                        region_tensor = self.transform(region)
                        regions.append(region_tensor)
                        valid_indices.append(i)
                except Exception as e:
                    warnings.warn(f"Failed to process region {idx}: {e}")
                    continue
                    
            if not regions:
                raise ValueError("No valid regions extracted")
            
            # 유효한 샘플들만 선택
            regions = torch.stack(regions)
            valid_indices = np.array(valid_indices)
            selected_labels = labels[selected_indices[valid_indices]]
            selected_bbox_targets = bbox_targets[selected_indices[valid_indices]]
            selected_bbox_masks = bbox_masks[selected_indices[valid_indices]]
            
            return regions, selected_labels, selected_bbox_targets, selected_bbox_masks
            
        except Exception as e:
            logging.error(f"Failed to prepare training data: {e}")
            # 더미 데이터 반환
            dummy_region = torch.zeros(1, 3, self.input_size, self.input_size)
            dummy_label = np.array([0], dtype=np.int64)
            dummy_bbox_target = np.zeros((1, 4), dtype=np.float32)
            dummy_bbox_mask = np.zeros(1, dtype=bool)
            return dummy_region, dummy_label, dummy_bbox_target, dummy_bbox_mask
            
    def _assign_labels(self, proposals, gt_boxes, gt_labels):
        """
        Proposals에 라벨을 할당합니다.
        
        Args:
            proposals: Region proposals [[x1, y1, x2, y2], ...]
            gt_boxes: Ground truth boxes [[x1, y1, x2, y2], ...]
            gt_labels: Ground truth labels [label1, label2, ...]
        
        Returns:
            labels: 각 proposal의 라벨 (0은 background)
            bbox_targets: 바운딩 박스 회귀 타겟
            bbox_masks: 회귀 손실 계산용 마스크
        """
        num_proposals = len(proposals)
        labels = np.zeros(num_proposals, dtype=np.int64)
        bbox_targets = np.zeros((num_proposals, 4), dtype=np.float32)
        bbox_masks = np.zeros(num_proposals, dtype=bool)
        
        if len(gt_boxes) == 0:
            return labels, bbox_targets, bbox_masks
        
        # 각 proposal에 대해 최대 IoU를 가지는 GT box 찾기
        try:
            for i, proposal in enumerate(proposals):
                try:
                    max_iou = 0
                    max_gt_idx = -1
                    
                    for j, gt_box in enumerate(gt_boxes):
                        try:
                            iou = compute_iou(proposal, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                max_gt_idx = j
                        except Exception as e:
                            warnings.warn(f"Failed to compute IoU for proposal {i} and gt_box {j}: {e}")
                            continue
                    
                    # 라벨 할당
                    if max_iou >= self.positive_threshold:
                        # Positive sample
                        labels[i] = gt_labels[max_gt_idx] + 1  # +1 because 0 is background
                        bbox_masks[i] = True
                        
                        # 바운딩 박스 회귀 타겟 계산
                        gt_box = gt_boxes[max_gt_idx]
                        bbox_targets[i] = self._compute_bbox_targets(proposal, gt_box)
                        
                    elif max_iou < self.negative_threshold:
                        # Negative sample (background)
                        labels[i] = 0
                        bbox_masks[i] = False
                except Exception as e:
                    warnings.warn(f"Failed to process proposal {i}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Failed to assign labels: {e}")
            # 모든 proposal을 background로 설정
            labels.fill(0)
            bbox_masks.fill(False)
            
        return labels, bbox_targets, bbox_masks
    
    def _compute_bbox_targets(self, proposal, gt_box):
        """
        바운딩 박스 회귀 타겟을 계산합니다.
        
        Args:
            proposal: [x1, y1, x2, y2]
            gt_box: [x1, y1, x2, y2]
        
        Returns:
            targets: [dx, dy, dw, dh] 정규화된 오프셋
        """
        # Center coordinates and dimensions
        px = (proposal[0] + proposal[2]) / 2.0
        py = (proposal[1] + proposal[3]) / 2.0
        pw = proposal[2] - proposal[0]
        ph = proposal[3] - proposal[1]
        
        gx = (gt_box[0] + gt_box[2]) / 2.0
        gy = (gt_box[1] + gt_box[3]) / 2.0
        gw = gt_box[2] - gt_box[0]
        gh = gt_box[3] - gt_box[1]
        
        # Compute targets
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = np.log(gw / pw)
        dh = np.log(gh / ph)
        
        return np.array([dx, dy, dw, dh], dtype=np.float32)
    
    def prepare_inference_data(self, image, proposals=None):
        """
        추론용 데이터를 준비합니다.
        
        Args:
            image: PIL Image
            proposals: 미리 계산된 proposals (None이면 새로 계산)
        
        Returns:
            regions: 추출된 region images
            proposals_xyxy: proposal 좌표
        """
        if proposals is None:
            proposals = selective_search(image, mode='quality')
            image_array = np.array(image)
            proposals = filter_proposals(
                proposals, 
                min_size=20, 
                image_shape=image_array.shape
            )
            proposals = convert_proposals_to_xyxy(proposals)
        
        # Region 이미지 추출
        regions = []
        for proposal in proposals:
            x1, y1, x2, y2 = proposal
            region = image.crop((x1, y1, x2, y2))
            region_tensor = self.transform(region)
            regions.append(region_tensor)
        
        if len(regions) > 0:
            regions = torch.stack(regions)
        else:
            regions = torch.empty(0, 3, self.input_size, self.input_size)
        
        return regions, proposals

def collate_fn(batch):
    """
    DataLoader용 collate 함수
    """
    images = []
    targets = []
    proposals_list = []
    
    for item in batch:
        images.append(item['image'])
        
        # 타겟 딕셔너리 생성
        target = {
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'boxes': torch.tensor(item['boxes'], dtype=torch.float32)
        }
        targets.append(target)
        
        # 간단한 더미 proposals (실제로는 selective search 결과를 사용해야 함)
        # 여기서는 단순화를 위해 전체 이미지를 proposal로 사용
        proposals = torch.tensor([[0, 0, 224, 224]], dtype=torch.float32)
        proposals_list.append(proposals)
    
    # 이미지들을 배치로 스택
    images = torch.stack(images, 0)
    
    return images, targets, proposals_list

def prepare_training_data(batch):
    """
    훈련 데이터 준비를 위한 함수
    """
    return collate_fn(batch)