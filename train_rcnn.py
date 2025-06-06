import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# 프로젝트 루트 디렉터리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.voc_dataset import VOC2007Dataset
from models.rcnn import RCNN, RCNNLoss
from utils.data_prep import RCNNDataPreprocessor

class RCNNTrainer:
    """R-CNN 모델 훈련기"""
    
    def __init__(self, config):
        self.config = config
        # GPU 자동 감지 (CUDA/MPS 지원)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # 데이터셋 로드
        self.train_dataset = VOC2007Dataset(
            root_dir=config['data_path'],
            image_set='train'
        )
        
        self.val_dataset = VOC2007Dataset(
            root_dir=config['data_path'],
            image_set='val'
        )
        
        # 데이터 전처리기
        self.preprocessor = RCNNDataPreprocessor(
            input_size=config['input_size'],
            positive_threshold=config['positive_threshold'],
            negative_threshold=config['negative_threshold']
        )
        
        # 모델 초기화
        self.model = RCNN(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # 손실 함수
        self.criterion = RCNNLoss(lambda_reg=config['lambda_reg'])
        
        # 옵티마이저
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """한 에포크 훈련"""
        self.model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        
        # 훈련 데이터셋에서 이미지를 하나씩 처리
        for idx in tqdm(range(len(self.train_dataset)), desc="Training"):
            try:
                # 데이터 로드
                sample = self.train_dataset[idx]
                image = sample['image']
                boxes = sample['boxes']
                labels = sample['labels']
                
                # PIL Image로 변환 (전처리를 위해)
                if isinstance(image, torch.Tensor):
                    # tensor를 PIL Image로 변환
                    image_pil = transforms.ToPILImage()(image)
                else:
                    image_pil = image
                
                # 훈련 데이터 준비
                regions, region_labels, bbox_targets, bbox_masks = self.preprocessor.prepare_training_data(
                    image_pil, boxes, labels,
                    max_proposals=self.config['max_proposals_train'],
                    positive_ratio=self.config['positive_ratio']
                )
                
                if len(regions) == 0:
                    continue
                
                # GPU로 이동
                regions = regions.to(self.device)
                region_labels = torch.from_numpy(region_labels).to(self.device)
                bbox_targets = torch.from_numpy(bbox_targets).to(self.device)
                bbox_masks = torch.from_numpy(bbox_masks).to(self.device)
                
                # Forward pass
                cls_scores, bbox_pred = self.model(regions)
                
                # 손실 계산
                total_loss, cls_loss, reg_loss = self.criterion(
                    cls_scores, bbox_pred, region_labels, bbox_targets, bbox_masks
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 손실 기록
                epoch_loss += total_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # 평균 손실 계산
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_cls_loss = epoch_cls_loss / num_batches
            avg_reg_loss = epoch_reg_loss / num_batches
        else:
            avg_loss = avg_cls_loss = avg_reg_loss = 0.0
        
        return avg_loss, avg_cls_loss, avg_reg_loss
    
    def validate(self):
        """검증"""
        self.model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_reg_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for idx in tqdm(range(min(len(self.val_dataset), 100)), desc="Validation"):
                try:
                    # 데이터 로드
                    sample = self.val_dataset[idx]
                    image = sample['image']
                    boxes = sample['boxes']
                    labels = sample['labels']
                    
                    # PIL Image로 변환
                    if isinstance(image, torch.Tensor):
                        image_pil = transforms.ToPILImage()(image)
                    else:
                        image_pil = image
                    
                    # 검증 데이터 준비
                    regions, region_labels, bbox_targets, bbox_masks = self.preprocessor.prepare_training_data(
                        image_pil, boxes, labels,
                        max_proposals=self.config['max_proposals_val'],
                        positive_ratio=self.config['positive_ratio']
                    )
                    
                    if len(regions) == 0:
                        continue
                    
                    # GPU로 이동
                    regions = regions.to(self.device)
                    region_labels = torch.from_numpy(region_labels).to(self.device)
                    bbox_targets = torch.from_numpy(bbox_targets).to(self.device)
                    bbox_masks = torch.from_numpy(bbox_masks).to(self.device)
                    
                    # Forward pass
                    cls_scores, bbox_pred = self.model(regions)
                    
                    # 손실 계산
                    total_loss, cls_loss, reg_loss = self.criterion(
                        cls_scores, bbox_pred, region_labels, bbox_targets, bbox_masks
                    )
                    
                    val_loss += total_loss.item()
                    val_cls_loss += cls_loss.item()
                    val_reg_loss += reg_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error processing validation sample {idx}: {str(e)}")
                    continue
        
        if num_batches > 0:
            avg_val_loss = val_loss / num_batches
            avg_val_cls_loss = val_cls_loss / num_batches
            avg_val_reg_loss = val_reg_loss / num_batches
        else:
            avg_val_loss = avg_val_cls_loss = avg_val_reg_loss = 0.0
        
        return avg_val_loss, avg_val_cls_loss, avg_val_reg_loss
    
    def train(self):
        """전체 훈련 과정"""
        print("Starting R-CNN training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 훈련
            train_loss, train_cls_loss, train_reg_loss = self.train_epoch()
            
            # 검증
            val_loss, val_cls_loss, val_reg_loss = self.validate()
            
            # 학습률 스케줄링
            self.scheduler.step()
            
            # 손실 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 결과 출력
            print(f"Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, reg: {train_reg_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} (cls: {val_cls_loss:.4f}, reg: {val_reg_loss:.4f})")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")
                print(f"Best model saved with val_loss: {best_val_loss:.4f}")
            
            # 주기적으로 모델 저장
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")
        
        # 훈련 완료 후 손실 그래프 그리기
        self.plot_losses()
        
        print("Training completed!")
    
    def save_model(self, filename):
        """모델 저장"""
        if not os.path.exists(self.config['save_dir']):
            os.makedirs(self.config['save_dir'])
        
        filepath = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def plot_losses(self):
        """손실 그래프 그리기"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('R-CNN Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        if not os.path.exists(self.config['save_dir']):
            os.makedirs(self.config['save_dir'])
        plt.savefig(os.path.join(self.config['save_dir'], 'training_loss.png'))
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='R-CNN Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to VOC2007 dataset')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # 훈련 설정
    config = {
        'data_path': args.data_path,
        'save_dir': args.save_dir,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_step_size': 5,
        'lr_gamma': 0.1,
        'num_classes': 20,  # VOC2007 클래스 수
        'input_size': 224,
        'positive_threshold': 0.5,
        'negative_threshold': 0.1,
        'lambda_reg': 1.0,
        'max_proposals_train': 128,
        'max_proposals_val': 64,
        'positive_ratio': 0.25,
        'pretrained': True,
        'save_interval': 2
    }
    
    # 훈련 시작
    trainer = RCNNTrainer(config)
    trainer.train()

if __name__ == "__main__":
    # 필요한 import 추가
    import torchvision.transforms as transforms
    
    # 예제 실행을 위한 기본 설정
    config = {
        'data_path': '/Users/milkylee/Documents/Projects/VOC/voc2007/VOCdevkit/VOC2007',
        'save_dir': './models',
        'num_epochs': 5,  # 빠른 테스트를 위해 줄임
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_step_size': 3,
        'lr_gamma': 0.1,
        'num_classes': 20,
        'input_size': 224,
        'positive_threshold': 0.5,
        'negative_threshold': 0.1,
        'lambda_reg': 1.0,
        'max_proposals_train': 64,  # 빠른 훈련을 위해 줄임
        'max_proposals_val': 32,
        'positive_ratio': 0.25,
        'pretrained': True,
        'save_interval': 2
    }
    
    print("R-CNN Training Script Ready!")
    print("To start training, run:")
    print(f"python train_rcnn.py --data_path {config['data_path']}")