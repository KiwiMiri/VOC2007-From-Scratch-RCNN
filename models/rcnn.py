import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class RCNN(nn.Module):
    """
    R-CNN 모델 구현
    - CNN feature extractor (ResNet-50)
    - Classifier (FC layers)
    - Bounding box regressor
    """
    
    def __init__(self, num_classes=20, pretrained=True):
        super(RCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extractor: ResNet-50 백본
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # ResNet의 마지막 FC layer 제거
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes + 1)  # +1 for background class
        )
        
        # Bounding box regressor
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes * 4)  # 4 coordinates per class
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 - 수치 안정성 개선"""
        for module in [self.classifier, self.bbox_regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Xavier/Glorot 초기화 사용
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x, proposals=None):
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        
        # Feature extraction with normalization
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Feature normalization for stability
        features = F.normalize(features, p=2, dim=1) * (features.size(1) ** 0.5)
        
        # Classification with stability check
        cls_scores = self.classifier(features)
        if torch.isnan(cls_scores).any():
            print("Warning: NaN in classification scores")
            cls_scores = torch.zeros_like(cls_scores)
        
        # Bounding box regression with clipping
        bbox_pred = self.bbox_regressor(features)
        bbox_pred = torch.clamp(bbox_pred, min=-10.0, max=10.0)  # Prevent extreme values
        
        if torch.isnan(bbox_pred).any():
            print("Warning: NaN in bbox predictions")
            bbox_pred = torch.zeros_like(bbox_pred)
        
        return cls_scores, bbox_pred
    
    def extract_features(self, x):
        """특징만 추출하는 함수"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        return features

class RCNNLoss(nn.Module):
    """R-CNN 손실 함수 - 수치 안정성 개선"""
    
    def __init__(self, lambda_reg=1.0, cls_weight=1.0):
        super(RCNNLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.cls_weight = cls_weight
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.eps = 1e-8  # 수치 안정성을 위한 작은 값
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: tuple of (cls_scores, bbox_pred)
                cls_scores: [N, num_classes + 1]
                bbox_pred: [N, num_classes * 4]
            targets: list of dicts with 'labels' and 'boxes'
        """
        cls_scores, bbox_pred = predictions
        
        # 배치에서 타겟 정보 추출
        cls_targets = []
        bbox_targets = []
        valid_mask = []
        
        for target in targets:
            if 'labels' in target and len(target['labels']) > 0:
                # 첫 번째 객체만 사용 (단순화)
                cls_targets.append(target['labels'][0].long())
                if 'boxes' in target and len(target['boxes']) > 0:
                    bbox_targets.append(target['boxes'][0])
                    valid_mask.append(True)
                else:
                    bbox_targets.append(torch.zeros(4, device=cls_scores.device))
                    valid_mask.append(False)
            else:
                cls_targets.append(torch.tensor(0, device=cls_scores.device))  # background
                bbox_targets.append(torch.zeros(4, device=cls_scores.device))
                valid_mask.append(False)
        
        cls_targets = torch.stack(cls_targets)
        bbox_targets = torch.stack(bbox_targets)
        valid_mask = torch.tensor(valid_mask, device=cls_scores.device)
        
        # Classification loss with stability check
        if torch.isnan(cls_scores).any() or torch.isinf(cls_scores).any():
            print("Warning: Invalid cls_scores detected")
            cls_loss = torch.tensor(0.0, device=cls_scores.device, requires_grad=True)
        else:
            # 점수를 클리핑하여 안정성 확보
            cls_scores_stable = torch.clamp(cls_scores, min=-10.0, max=10.0)
            cls_loss = self.cls_loss(cls_scores_stable, cls_targets)
        
        # Bounding box regression loss
        reg_loss = torch.tensor(0.0, device=cls_scores.device, requires_grad=True)
        
        if valid_mask.sum() > 0 and not torch.isnan(bbox_pred).any():
            # 포지티브 샘플에 대해서만 회귀 손실 계산
            positive_indices = (cls_targets > 0) & valid_mask
            
            if positive_indices.sum() > 0:
                # 단순화: 모든 클래스에 대해 같은 bbox 사용
                bbox_pred_simple = bbox_pred[:, :4]  # 첫 번째 클래스의 bbox만 사용
                
                # 값 클리핑
                bbox_pred_clipped = torch.clamp(bbox_pred_simple, min=-10.0, max=10.0)
                bbox_targets_clipped = torch.clamp(bbox_targets, min=-10.0, max=10.0)
                
                try:
                    reg_loss = self.reg_loss(
                        bbox_pred_clipped[positive_indices], 
                        bbox_targets_clipped[positive_indices]
                    )
                except:
                    reg_loss = torch.tensor(0.0, device=cls_scores.device, requires_grad=True)
        
        # 총 손실 계산
        total_loss = self.cls_weight * cls_loss + self.lambda_reg * reg_loss
        
        # 최종 안정성 체크
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf in total loss, using fallback")
            total_loss = torch.tensor(1.0, device=cls_scores.device, requires_grad=True)
        
        return total_loss