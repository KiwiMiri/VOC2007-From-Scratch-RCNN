#!/usr/bin/env python3
# filepath: /Users/milkylee/Documents/Projects/VOC/voc2007/rcnn_implementation/data/voc_dataset_fixed.py
"""
개선된 VOC Dataset 클래스
- XML 손상 파일 에러 처리 강화
- 이미지 손상 파일 에러 처리 강화
- 안정적인 데이터 로딩을 위한 예외 처리
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import warnings

class VOC2007Dataset(Dataset):
    """VOC2007 데이터셋 로더"""
    
    def __init__(self, root_dir, image_set='train', transform=None):
        """
        Args:
            root_dir (string): VOC2007 루트 디렉터리 경로
            image_set (string): 'train', 'val', 'test' 중 하나
            transform (callable, optional): 이미지에 적용할 변환
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        
        # VOC2007 클래스 정의 (20개 클래스)
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # 이미지 ID 목록 로드
        self.image_ids = self._load_image_ids()
        
        # 손상된 파일 리스트 (알려진 문제 파일들)
        self.known_corrupt_annotations = self._load_known_corrupt_files('annotations')
        self.known_corrupt_images = self._load_known_corrupt_files('images')
        
        logging.info(f"Dataset initialized with {len(self.image_ids)} images")
        logging.info(f"Known corrupt annotations: {len(self.known_corrupt_annotations)}")
        logging.info(f"Known corrupt images: {len(self.known_corrupt_images)}")
        
    def _load_known_corrupt_files(self, file_type):
        """알려진 손상 파일 목록 반환"""
        corrupt_files = set()
        
        # 알려진 손상 주석 파일
        if file_type == 'annotations':
            corrupt_files.update([
                '003550', '003551',  # 비어있는 XML 파일
                # 추가 손상된 어노테이션 파일이 있다면 여기에 추가
            ])
        
        # 알려진 손상 이미지 파일
        elif file_type == 'images':
            corrupt_files.update([
                '006495', '001597', '001604'  # 손상된 이미지 파일
                # 추가 손상된 이미지 파일이 있다면 여기에 추가
            ])
        
        return corrupt_files
        
    def _load_image_ids(self):
        """이미지 ID 목록을 로드합니다."""
        image_set_file = os.path.join(
            self.root_dir, 'ImageSets', 'Main', f'{self.image_set}.txt'
        )
        
        try:
            with open(image_set_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            return image_ids
        except Exception as e:
            logging.error(f"Failed to load image IDs from {image_set_file}: {e}")
            return []
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """데이터셋에서 하나의 아이템을 가져옵니다."""
        max_retries = 10  # 최대 재시도 횟수
        
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.image_ids)
                image_id = self.image_ids[current_idx]
                
                # 알려진 손상 파일인지 확인
                if image_id in self.known_corrupt_annotations or image_id in self.known_corrupt_images:
                    logging.warning(f"Skipping known corrupt file with ID: {image_id}")
                    continue
                
                # 이미지 로드
                image_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
                
                # 이미지 파일 검증
                if not os.path.exists(image_path):
                    logging.warning(f"Image file not found: {image_path}")
                    continue
                    
                # 이미지 크기 검증 (완화: 작은 이미지도 허용)
                if os.path.getsize(image_path) < 100:  # 100바이트 미만이면 손상된 것으로 간주
                    logging.warning(f"Image file too small (possibly corrupted): {image_path}")
                    # 알려진 손상 목록에 추가
                    self.known_corrupt_images.add(image_id)
                    continue
                
                try:
                    # 이미지 로드 시도
                    image = Image.open(image_path).convert('RGB')
                    
                    # 추가 이미지 유효성 검사
                    if image.width < 10 or image.height < 10:
                        logging.warning(f"Image dimensions too small: {image_path}, {image.width}x{image.height}")
                        self.known_corrupt_images.add(image_id)
                        continue
                except Exception as e:
                    logging.warning(f"Cannot load image {image_path}: {e}")
                    self.known_corrupt_images.add(image_id)
                    continue
                
                # 어노테이션 로드 (XML 파일)
                annotation_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')
                
                try:
                    # XML 파일 파싱
                    boxes, labels = self._parse_annotation(annotation_path)
                    
                    if len(boxes) == 0 or len(labels) == 0:
                        logging.warning(f"No valid annotations found for image: {image_id}")
                        # 어노테이션이 없는 경우 더미 값 대신 백그라운드 클래스로 설정 (클래스 0)
                        boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
                        labels = np.array([0], dtype=np.int64)
                except Exception as e:
                    logging.warning(f"Cannot parse annotation {annotation_path}: {e}")
                    # 알려진 손상 목록에 추가
                    self.known_corrupt_annotations.add(image_id)
                    
                    # 어노테이션 오류 시 더미 값 대신 백그라운드 클래스로 설정
                    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
                    labels = np.array([0], dtype=np.int64)
                
                # 변환 적용
                if self.transform:
                    try:
                        image = self.transform(image)
                    except Exception as e:
                        logging.warning(f"Transform failed for image {image_id}: {e}")
                        # 기본 변환 적용
                        image = transforms.ToTensor()(image)
                else:
                    # transform이 없으면 기본 ToTensor 적용
                    image = transforms.ToTensor()(image)
                
                return {
                    'image': image,
                    'boxes': boxes.astype(np.float32),
                    'labels': labels.astype(np.int64),
                    'image_id': image_id
                }
                
            except Exception as e:
                logging.warning(f"Unexpected error processing image {image_id}: {e}")
                continue
        
        # 모든 재시도가 실패한 경우 기본 더미 데이터 반환
        logging.error(f"Failed to load any valid image after {max_retries} retries starting from index {idx}")
        
        # 더미 이미지와 어노테이션 생성
        dummy_image = torch.zeros(3, 224, 224)
        dummy_boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)  # 정규화된 좌표
        dummy_labels = np.zeros(1, dtype=np.int64)
        
        return {
            'image': dummy_image,
            'boxes': dummy_boxes,
            'labels': dummy_labels,
            'image_id': 'dummy'
        }
    
    def _parse_annotation(self, annotation_path):
        """XML 어노테이션 파일을 파싱합니다."""
        boxes = []
        labels = []
        
        try:
            # 파일 존재 확인
            if not os.path.exists(annotation_path):
                logging.error(f"Annotation file not found: {annotation_path}")
                return np.array(boxes), np.array(labels)
                
            # 파일 크기 확인 (비어있는 파일 감지)
            if os.path.getsize(annotation_path) < 10:  # 10바이트 미만이면 실질적으로 비어있다고 간주
                logging.error(f"Annotation file is empty or too small: {annotation_path}")
                return np.array(boxes), np.array(labels)
            
            # XML 파일 파싱 시도
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
            except ET.ParseError as e:
                logging.error(f"XML parse error in {annotation_path}: {e}")
                # 파일 내용 직접 확인
                try:
                    with open(annotation_path, 'r') as f:
                        content = f.read(200)  # 처음 200자만 로그로 출력
                    logging.error(f"Content preview: {content}...")
                except:
                    pass
                return np.array(boxes), np.array(labels)
            
            # 이미지 크기 정보 가져오기 (bbox 정규화를 위해)
            size_elem = root.find('size')
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                if width_elem is not None and height_elem is not None:
                    try:
                        img_width = float(width_elem.text)
                        img_height = float(height_elem.text)
                    except (ValueError, AttributeError):
                        # 크기 정보를 가져올 수 없으면 기본값 사용
                        img_width, img_height = 300.0, 300.0
                else:
                    img_width, img_height = 300.0, 300.0
            else:
                img_width, img_height = 300.0, 300.0
            
            # 각 객체 처리
            for obj in root.findall('object'):
                try:
                    # 클래스 라벨
                    name_element = obj.find('name')
                    if name_element is None or name_element.text is None:
                        logging.warning(f"Missing class name in {annotation_path}")
                        continue
                    
                    class_name = name_element.text.strip()
                    if class_name in self.class_to_idx:
                        # 바운딩 박스 좌표
                        bbox = obj.find('bndbox')
                        if bbox is None:
                            logging.warning(f"Missing bndbox in {annotation_path}")
                            continue
                            
                        # 안전하게 바운딩 박스 좌표 추출
                        try:
                            xmin_elem = bbox.find('xmin')
                            ymin_elem = bbox.find('ymin')
                            xmax_elem = bbox.find('xmax')
                            ymax_elem = bbox.find('ymax')
                            
                            # 모든 좌표가 존재하는지 확인
                            if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                                logging.warning(f"Incomplete bounding box in {annotation_path}")
                                continue
                                
                            # 모든 좌표에 텍스트가 있는지 확인
                            if None in [xmin_elem.text, ymin_elem.text, xmax_elem.text, ymax_elem.text]:
                                logging.warning(f"Missing coordinate values in {annotation_path}")
                                continue
                            
                            # 좌표를 정수로 변환
                            xmin = int(float(xmin_elem.text))
                            ymin = int(float(ymin_elem.text))
                            xmax = int(float(xmax_elem.text))
                            ymax = int(float(ymax_elem.text))
                            
                            # 좌표 유효성 검증 (픽셀 좌표 기준)
                            if xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin:
                                # bbox를 0-1 범위로 정규화
                                xmin_norm = xmin / img_width
                                ymin_norm = ymin / img_height
                                xmax_norm = xmax / img_width
                                ymax_norm = ymax / img_height
                                
                                boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                                labels.append(self.class_to_idx[class_name])
                            else:
                                # 더 자세한 검증 조건 로깅
                                conditions = []
                                if xmin < 0:
                                    conditions.append(f"xmin<0 ({xmin})")
                                if ymin < 0:
                                    conditions.append(f"ymin<0 ({ymin})")
                                if xmax <= xmin:
                                    conditions.append(f"xmax<=xmin ({xmax}<={xmin})")
                                if ymax <= ymin:
                                    conditions.append(f"ymax<=ymin ({ymax}<={ymin})")
                                
                                logging.warning(f"Invalid box coordinates in {annotation_path}: [{xmin}, {ymin}, {xmax}, {ymax}] - Failed: {', '.join(conditions)}")
                        except (ValueError, AttributeError) as e:
                            logging.warning(f"Invalid bounding box in {annotation_path}: {e}")
                except Exception as e:
                    logging.warning(f"Error processing object in {annotation_path}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Failed to process annotation {annotation_path}: {e}")
        
        return np.array(boxes), np.array(labels)
    
    def get_image_info(self, idx):
        """이미지 정보를 반환합니다. 안전하게 처리합니다."""
        try:
            image_id = self.image_ids[idx]
            annotation_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')
            
            # 알려진 손상 파일이면 기본값 반환
            if image_id in self.known_corrupt_annotations:
                return {'width': 300, 'height': 300, 'image_id': image_id}
            
            # 파일 존재 확인
            if not os.path.exists(annotation_path):
                return {'width': 300, 'height': 300, 'image_id': image_id}
                
            # XML 파일 파싱
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                size = root.find('size')
                if size is None:
                    return {'width': 300, 'height': 300, 'image_id': image_id}
                
                width_elem = size.find('width')
                height_elem = size.find('height')
                
                if width_elem is None or height_elem is None or width_elem.text is None or height_elem.text is None:
                    return {'width': 300, 'height': 300, 'image_id': image_id}
                
                width = int(width_elem.text)
                height = int(height_elem.text)
                
                return {'width': width, 'height': height, 'image_id': image_id}
            except Exception as e:
                logging.warning(f"Error getting image info from {annotation_path}: {e}")
                return {'width': 300, 'height': 300, 'image_id': image_id}
        except Exception as e:
            logging.error(f"Error in get_image_info for index {idx}: {e}")
            return {'width': 300, 'height': 300, 'image_id': 'unknown'}

def get_transform(is_train=True):
    """이미지 변환 함수"""
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    
    if is_train:
        transforms_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
    
    transforms_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transforms_list)
