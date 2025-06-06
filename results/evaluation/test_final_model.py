#!/usr/bin/env python3
"""
최종 편향 제거 모델 성능 평가
정확한 성능 측정
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.rcnn import RCNN
from data.voc_dataset import VOC2007Dataset
import torchvision.transforms as transforms

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def evaluate_model_properly(model, dataset, device, sample_size=200):
    """정확한 모델 평가"""
    print(f"\n🎯 Evaluating model on {sample_size} samples...")
    
    model.eval()
    
    # 평가 결과 저장
    predictions = []
    ground_truths = []
    confidences = []
    
    # 다양한 평가 지표
    exact_match = 0  # 정확히 일치
    partial_match = 0  # 부분 일치 (멀티라벨에서 하나라도 맞음)
    total_samples = 0
    
    # 클래스별 성능
    class_tp = [0] * len(VOC_CLASSES)  # True Positive
    class_fp = [0] * len(VOC_CLASSES)  # False Positive  
    class_fn = [0] * len(VOC_CLASSES)  # False Negative
    
    with torch.no_grad():
        for i in tqdm(range(min(sample_size, len(dataset))), desc="Evaluating"):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            labels = sample['labels']
            
            if len(labels) == 0:
                continue
                
            cls_scores, _ = model(image)
            probs = torch.softmax(cls_scores, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
            
            # Background class 처리
            if predicted_class >= len(VOC_CLASSES):
                continue  # Skip background predictions
                
            predictions.append(predicted_class)
            confidences.append(confidence)
            
            # Ground truth 처리 (첫 번째 객체만 고려)
            gt_labels = [l.item() for l in labels]
            primary_gt = gt_labels[0] if gt_labels else -1
            
            ground_truths.append(primary_gt)
            
            # 정확도 계산
            if predicted_class == primary_gt:
                exact_match += 1
            
            if predicted_class in gt_labels:
                partial_match += 1
            
            # 클래스별 성능 계산
            if predicted_class < len(VOC_CLASSES):
                if predicted_class in gt_labels:
                    class_tp[predicted_class] += 1
                else:
                    class_fp[predicted_class] += 1
            
            for gt_label in gt_labels:
                if gt_label != predicted_class and gt_label < len(VOC_CLASSES):
                    class_fn[gt_label] += 1
            
            total_samples += 1
    
    if total_samples == 0:
        print("❌ No valid samples found!")
        return
    
    # 전체 성능 계산
    exact_accuracy = (exact_match / total_samples) * 100
    partial_accuracy = (partial_match / total_samples) * 100
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total samples: {total_samples}")
    print(f"  Exact match accuracy: {exact_accuracy:.2f}% ({exact_match}/{total_samples})")
    print(f"  Partial match accuracy: {partial_accuracy:.2f}% ({partial_match}/{total_samples})")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    print(f"  Confidence std: {np.std(confidences):.4f}")
    
    # 예측 분포 분석
    pred_counter = Counter(predictions)
    gt_counter = Counter(ground_truths)
    
    print(f"\n📈 Prediction Distribution:")
    for class_idx, count in pred_counter.most_common():
        if class_idx < len(VOC_CLASSES):
            name = VOC_CLASSES[class_idx]
            percentage = (count / len(predictions)) * 100
            print(f"  {name:12s}: {count:3d} ({percentage:5.1f}%)")
    
    # 편향 분석
    max_pred_count = max(pred_counter.values()) if pred_counter else 0
    bias_percentage = (max_pred_count / len(predictions)) * 100 if predictions else 0
    diversity = len(pred_counter)
    
    print(f"\n🎯 Bias Analysis:")
    print(f"  Max bias: {bias_percentage:.1f}%")
    print(f"  Classes predicted: {diversity}/{len(VOC_CLASSES)}")
    
    # 클래스별 성능 (precision, recall, f1)
    print(f"\n📋 Class-wise Performance (Top 10):")
    print("  Class        | Precision | Recall   | F1-Score")
    print("  " + "-" * 48)
    
    class_scores = []
    
    for i, class_name in enumerate(VOC_CLASSES):
        tp = class_tp[i]
        fp = class_fp[i]
        fn = class_fn[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_scores.append((class_name, precision, recall, f1))
    
    # F1 score로 정렬하여 상위 10개만 표시
    class_scores.sort(key=lambda x: x[3], reverse=True)
    
    for class_name, precision, recall, f1 in class_scores[:10]:
        print(f"  {class_name:12s} | {precision:9.3f} | {recall:8.3f} | {f1:8.3f}")
    
    return {
        'exact_accuracy': exact_accuracy,
        'partial_accuracy': partial_accuracy,
        'bias_percentage': bias_percentage,
        'diversity': diversity,
        'avg_confidence': np.mean(confidences),
        'total_samples': total_samples
    }

def test_all_models():
    """모든 모델 성능 비교"""
    print("🔄 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = VOC2007Dataset(
            root_dir='/Users/milkylee/Documents/Projects/VOC/voc2007/VOCdevkit/VOC2007',
            image_set='test',
            transform=transform
        )
        print(f"✅ Test dataset loaded: {len(dataset)} images")
    except:
        dataset = VOC2007Dataset(
            root_dir='/Users/milkylee/Documents/Projects/VOC/voc2007/VOCdevkit/VOC2007',
            image_set='val',
            transform=transform
        )
        print(f"✅ Validation dataset loaded: {len(dataset)} images")
    
    # 테스트할 모델들
    models_to_test = [
        ('models/test_model_from_scratch.pth', 'Original From Scratch'),
        ('models/fixed_existing_model.pth', 'Xavier Fixed'),
        ('models/completely_debiased_model.pth', 'Completely Debiased'),
    ]
    
    results = {}
    
    for model_path, model_name in models_to_test:
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue
            
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # 모델 로드
        model = RCNN(num_classes=20, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        
        # 성능 평가
        result = evaluate_model_properly(model, dataset, device, sample_size=200)
        if result:
            results[model_name] = result
    
    # 최종 비교
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("🏆 FINAL COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"{'Model':<20} | {'Exact Acc':<9} | {'Partial Acc':<11} | {'Bias':<6} | {'Diversity':<9}")
        print("-" * 70)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} | {result['exact_accuracy']:8.2f}% | {result['partial_accuracy']:10.2f}% | {result['bias_percentage']:5.1f}% | {result['diversity']:2d}/20")
        
        # 최고 성능 모델 찾기
        best_model = max(results.items(), key=lambda x: x[1]['exact_accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Exact Accuracy: {best_model[1]['exact_accuracy']:.2f}%")
        print(f"   Bias Level: {best_model[1]['bias_percentage']:.1f}%")

if __name__ == "__main__":
    test_all_models() 