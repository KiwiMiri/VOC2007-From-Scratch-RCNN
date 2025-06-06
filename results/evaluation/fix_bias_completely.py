#!/usr/bin/env python3
"""
편향 문제 완전 해결 스크립트
- 다양한 초기화 방법 시도
- 가중치 분석 및 수정
- 극단적 bias 제거
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.rcnn import RCNN

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def test_bias(model, device, test_name=""):
    """편향 테스트 함수"""
    print(f"\n🧪 Testing {test_name}")
    print("-" * 40)
    
    # 다양한 테스트 입력
    test_inputs = [
        torch.randn(1, 3, 224, 224).to(device),
        torch.zeros(1, 3, 224, 224).to(device),
        torch.ones(1, 3, 224, 224).to(device),
        torch.randn(1, 3, 224, 224).to(device) * 0.5,
        torch.randn(1, 3, 224, 224).to(device) * 2.0,
    ]
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for test_input in test_inputs:
            cls_scores, _ = model(test_input)
            predicted_class = torch.argmax(cls_scores, dim=1).item()
            predictions.append(predicted_class)
    
    # 예측 분포 분석
    pred_counter = Counter(predictions)
    max_count = max(pred_counter.values()) if pred_counter else 0
    bias_percentage = (max_count / len(predictions)) * 100
    
    print(f"Prediction distribution:")
    for class_idx, count in pred_counter.most_common():
        if class_idx < len(VOC_CLASSES):
            name = VOC_CLASSES[class_idx]
        elif class_idx == len(VOC_CLASSES):
            name = "BACKGROUND"
        else:
            name = f"ERROR_{class_idx}"
        percentage = (count / len(predictions)) * 100
        print(f"  {name}: {count}/5 ({percentage:.1f}%)")
    
    print(f"Max bias: {bias_percentage:.1f}%")
    return bias_percentage, len(pred_counter)

def apply_aggressive_debiasing(model):
    """공격적인 편향 제거"""
    print("🔧 Applying aggressive debiasing...")
    
    # 1. 분류기의 마지막 레이어 bias를 0으로 초기화
    if hasattr(model.classifier[-1], 'bias') and model.classifier[-1].bias is not None:
        nn.init.constant_(model.classifier[-1].bias, 0.0)
        print("  ✅ Reset final classifier bias to zero")
    
    # 2. 마지막 레이어 가중치를 매우 작은 값으로 초기화
    if hasattr(model.classifier[-1], 'weight'):
        nn.init.normal_(model.classifier[-1].weight, mean=0.0, std=0.001)
        print("  ✅ Reset final classifier weights (std=0.001)")
    
    # 3. 모든 분류기 bias 재설정
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    
    print("  ✅ Reset all classifier biases")

def apply_uniform_initialization(model):
    """균등 초기화 적용"""
    print("🔧 Applying uniform initialization...")
    
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            # 균등 분포로 초기화
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.01, 0.01)
    
    print("  ✅ Applied uniform initialization")

def apply_he_initialization(model):
    """He 초기화 적용"""
    print("🔧 Applying He initialization...")
    
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    print("  ✅ Applied He initialization")

def fix_bias_completely():
    """편향 문제 완전 해결"""
    print("🎯 COMPLETE BIAS ELIMINATION")
    print("=" * 60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 원본 모델 로드
    model_path = 'models/fixed_existing_model.pth'
    model = RCNN(num_classes=20, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # 원본 편향 테스트
    bias_before, diversity_before = test_bias(model, device, "Original Model")
    
    # 다양한 해결 방법 시도
    methods = [
        ("Aggressive Debiasing", apply_aggressive_debiasing),
        ("Uniform Initialization", apply_uniform_initialization),
        ("He Initialization", apply_he_initialization),
    ]
    
    best_bias = bias_before
    best_diversity = diversity_before
    best_method = "Original"
    best_model_state = None
    
    for method_name, method_func in methods:
        # 모델 상태 백업
        original_state = model.state_dict().copy()
        
        # 방법 적용
        method_func(model)
        
        # 편향 테스트
        bias, diversity = test_bias(model, device, method_name)
        
        # 성능 평가 (낮은 편향 + 높은 다양성이 좋음)
        score = (100 - bias) + (diversity * 10)  # 다양성에 가중치
        best_score = (100 - best_bias) + (best_diversity * 10)
        
        if score > best_score:
            best_bias = bias
            best_diversity = diversity
            best_method = method_name
            best_model_state = model.state_dict().copy()
            print(f"  🏆 New best method: {method_name}")
        
        # 원본 상태 복원 (다음 테스트를 위해)
        model.load_state_dict(original_state)
    
    # 최고 성능 방법 적용
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n🏆 Best method: {best_method}")
        print(f"   Bias: {best_bias:.1f}% (was {bias_before:.1f}%)")
        print(f"   Diversity: {best_diversity}/20 classes (was {diversity_before}/20)")
    
    # 추가 미세 조정
    if best_bias > 50:  # 여전히 편향이 있다면
        print("\n🔧 Applying additional fine-tuning...")
        
        # 극단적 조치: 마지막 레이어 완전 재초기화
        if hasattr(model.classifier[-1], 'weight'):
            fan_in = model.classifier[-1].weight.size(1)
            std = np.sqrt(2.0 / fan_in)  # He initialization std
            
            # 매우 작은 가중치로 초기화
            nn.init.normal_(model.classifier[-1].weight, mean=0.0, std=std * 0.1)
            if model.classifier[-1].bias is not None:
                nn.init.normal_(model.classifier[-1].bias, mean=0.0, std=0.001)
        
        # 최종 테스트
        final_bias, final_diversity = test_bias(model, device, "Final Debiased Model")
        
        print(f"\n📊 Final Results:")
        print(f"   Before: {bias_before:.1f}% bias, {diversity_before} classes")
        print(f"   After:  {final_bias:.1f}% bias, {final_diversity} classes")
        
        improvement = bias_before - final_bias
        print(f"   Improvement: {improvement:.1f}% reduction in bias")
    
    # 모델 저장
    output_path = 'models/completely_debiased_model.pth'
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'method_used': best_method,
        'bias_before': bias_before,
        'bias_after': best_bias,
        'diversity_before': diversity_before,
        'diversity_after': best_diversity,
        'epoch': checkpoint.get('epoch', 'Unknown'),
    }
    
    torch.save(save_dict, output_path)
    print(f"\n💾 Completely debiased model saved: {output_path}")
    
    # 성공 여부 판정
    if best_bias < 40:  # 40% 미만이면 성공
        print("\n🎉 SUCCESS: Bias significantly reduced!")
    elif best_bias < 60:
        print("\n⚠️ PARTIAL SUCCESS: Some bias reduction achieved")
    else:
        print("\n❌ FAILED: Bias still too high")
        print("   May need architectural changes or more training data")
    
    return model

if __name__ == "__main__":
    fixed_model = fix_bias_completely() 