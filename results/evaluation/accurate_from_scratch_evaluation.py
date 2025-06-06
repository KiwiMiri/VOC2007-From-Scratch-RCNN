import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# 로컬 모듈 임포트
from data.voc_dataset import VOC2007Dataset
from models.rcnn import RCNN

# VOC 클래스 정의
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_from_scratch_model(model_path):
    """From Scratch 모델 로드"""
    print(f"🔄 Loading From Scratch model: {model_path}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 모델 생성 (pretrained=False)
    model = RCNN(num_classes=20, pretrained=False).to(device)
    
    # 체크포인트 로드
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model state dict from checkpoint")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Loaded state dict from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights directly")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model, device

def evaluate_model_comprehensive(model, device, sample_size=None):
    """종합적인 모델 평가"""
    print("\n📊 Starting comprehensive evaluation...")
    
    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 테스트 데이터셋 로드
    try:
        test_dataset = VOC2007Dataset(
            root_dir='/Users/milkylee/Documents/Projects/VOC/voc2007/VOCdevkit/VOC2007',
            image_set='test',
            transform=transform
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    except:
        print("⚠️ Test set not available, using validation set...")
        test_dataset = VOC2007Dataset(
            root_dir='/Users/milkylee/Documents/Projects/VOC/voc2007/VOCdevkit/VOC2007',
            image_set='val',
            transform=transform
        )
        print(f"✅ Validation dataset loaded: {len(test_dataset)} samples")
    
    # 샘플 크기 조정
    if sample_size:
        evaluation_size = min(sample_size, len(test_dataset))
    else:
        evaluation_size = len(test_dataset)
    
    print(f"📏 Evaluating on {evaluation_size} samples")
    
    # 평가 메트릭 초기화
    total_samples = 0
    correct_predictions = 0
    class_predictions = Counter()
    class_ground_truth = Counter()
    class_correct = defaultdict(int)
    
    # 클래스별 TP, FP, FN 카운터
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    all_predictions = []
    all_ground_truths = []
    prediction_confidences = []
    
    print("\n🔄 Running evaluation...")
    
    with torch.no_grad():
        for i in tqdm(range(evaluation_size), desc="Evaluating"):
            try:
                sample = test_dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                labels = sample['labels']  # 실제 레이블들
                
                if len(labels) == 0:
                    continue
                
                # 모델 예측
                cls_scores, _ = model(image)
                probs = torch.softmax(cls_scores, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
                
                # 통계 수집
                total_samples += 1
                prediction_confidences.append(confidence)
                all_predictions.append(predicted_class)
                all_ground_truths.append(labels)
                
                # 클래스별 카운팅
                class_predictions[predicted_class] += 1
                for label in labels:
                    class_ground_truth[label] += 1
                
                # 정확도 계산 (예측한 클래스가 실제 레이블 중 하나와 일치하는지)
                if predicted_class in labels:
                    correct_predictions += 1
                    class_correct[predicted_class] += 1
                    true_positives[predicted_class] += 1
                else:
                    false_positives[predicted_class] += 1
                
                # False Negatives 계산
                for label in labels:
                    if label != predicted_class:
                        false_negatives[label] += 1
                        
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # 전체 정확도 계산
    overall_accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
    
    # 클래스별 성능 계산
    class_metrics = {}
    for class_idx in range(20):
        tp = true_positives[class_idx]
        fp = false_positives[class_idx]
        fn = false_negatives[class_idx]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_idx] = {
            'class_name': VOC_CLASSES[class_idx],
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'predictions': class_predictions[class_idx],
            'ground_truth': class_ground_truth[class_idx]
        }
    
    # 평균 메트릭 계산
    avg_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
    avg_f1_score = np.mean([metrics['f1_score'] for metrics in class_metrics.values()])
    
    # 예측 다양성 분석
    unique_predictions = len(class_predictions)
    max_entropy = np.log(20)  # 20개 클래스에 대한 최대 엔트로피
    pred_counts = np.array(list(class_predictions.values()))
    pred_probs = pred_counts / np.sum(pred_counts)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-8))
    
    # 편향 분석
    aeroplane_idx = VOC_CLASSES.index('aeroplane')
    aeroplane_predictions = class_predictions.get(aeroplane_idx, 0)
    aeroplane_bias = (aeroplane_predictions / total_samples * 100) if total_samples > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1_score,
        'class_metrics': class_metrics,
        'prediction_diversity': {
            'unique_classes': unique_predictions,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'diversity_ratio': entropy / max_entropy
        },
        'bias_analysis': {
            'aeroplane_bias': aeroplane_bias,
            'aeroplane_predictions': aeroplane_predictions,
            'most_predicted_class': max(class_predictions, key=class_predictions.get) if class_predictions else None,
            'prediction_distribution': dict(class_predictions)
        },
        'confidence_stats': {
            'mean_confidence': np.mean(prediction_confidences),
            'std_confidence': np.std(prediction_confidences),
            'min_confidence': np.min(prediction_confidences),
            'max_confidence': np.max(prediction_confidences)
        }
    }

def print_detailed_results(results):
    """상세 결과 출력"""
    print("\n" + "="*80)
    print("📊 FROM SCRATCH R-CNN 정확한 성능 측정 결과")
    print("="*80)
    
    # 전체 성능
    print(f"\n🎯 전체 성능:")
    print(f"  📊 정확도: {results['overall_accuracy']:.2f}% ({results['correct_predictions']}/{results['total_samples']})")
    print(f"  📈 평균 정밀도: {results['avg_precision']:.4f}")
    print(f"  📉 평균 재현율: {results['avg_recall']:.4f}")
    print(f"  🎯 평균 F1-score: {results['avg_f1_score']:.4f}")
    
    # 예측 다양성
    diversity = results['prediction_diversity']
    print(f"\n🌈 예측 다양성:")
    print(f"  📊 예측된 클래스 수: {diversity['unique_classes']}/20")
    print(f"  📈 엔트로피: {diversity['entropy']:.3f}/{diversity['max_entropy']:.3f}")
    print(f"  📊 다양성 비율: {diversity['diversity_ratio']:.3f}")
    
    # 편향 분석
    bias = results['bias_analysis']
    print(f"\n⚖️ 편향 분석:")
    print(f"  ✈️ Aeroplane 편향: {bias['aeroplane_bias']:.1f}% ({bias['aeroplane_predictions']}회)")
    if bias['aeroplane_bias'] == 100.0:
        print("  ❌ 심각한 편향: 100% aeroplane 예측")
    elif bias['aeroplane_bias'] > 50.0:
        print("  ⚠️ 높은 편향: >50% aeroplane 예측")
    elif bias['aeroplane_bias'] > 20.0:
        print("  ⚠️ 중간 편향: >20% aeroplane 예측")
    else:
        print("  ✅ 편향 해결: 정상적 예측 분포")
    
    # 신뢰도 통계
    conf = results['confidence_stats']
    print(f"\n📊 예측 신뢰도:")
    print(f"  📈 평균: {conf['mean_confidence']:.4f}")
    print(f"  📊 표준편차: {conf['std_confidence']:.4f}")
    print(f"  📉 최소: {conf['min_confidence']:.4f}")
    print(f"  📈 최대: {conf['max_confidence']:.4f}")
    
    # 상위 5개 클래스 성능
    print(f"\n🏆 클래스별 성능 (상위 5개):")
    sorted_classes = sorted(results['class_metrics'].items(), 
                          key=lambda x: x[1]['f1_score'], reverse=True)
    
    print(f"{'순위':<4} {'클래스':<12} {'정밀도':<8} {'재현율':<8} {'F1-Score':<8} {'예측수':<6}")
    print("-" * 60)
    
    for i, (class_idx, metrics) in enumerate(sorted_classes[:5]):
        print(f"{i+1:<4} {metrics['class_name']:<12} "
              f"{metrics['precision']:<8.3f} {metrics['recall']:<8.3f} "
              f"{metrics['f1_score']:<8.3f} {metrics['predictions']:<6}")

def save_results_to_report(results, model_name):
    """결과를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        'model_name': model_name,
        'evaluation_time': timestamp,
        'performance': {
            'accuracy': results['overall_accuracy'],
            'avg_precision': results['avg_precision'],
            'avg_recall': results['avg_recall'],
            'avg_f1_score': results['avg_f1_score']
        },
        'detailed_results': results
    }
    
    # JSON 파일로 저장
    with open('from_scratch_accurate_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 텍스트 보고서 저장
    with open('from_scratch_accurate_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"FROM SCRATCH R-CNN 정확한 성능 측정 보고서\n")
        f.write(f"="*60 + "\n")
        f.write(f"모델: {model_name}\n")
        f.write(f"평가 시간: {timestamp}\n")
        f.write(f"평가 샘플 수: {results['total_samples']}\n\n")
        
        f.write(f"전체 성능:\n")
        f.write(f"  정확도: {results['overall_accuracy']:.2f}%\n")
        f.write(f"  평균 정밀도: {results['avg_precision']:.4f}\n")
        f.write(f"  평균 재현율: {results['avg_recall']:.4f}\n")
        f.write(f"  평균 F1-score: {results['avg_f1_score']:.4f}\n\n")
        
        f.write(f"편향 분석:\n")
        f.write(f"  Aeroplane 편향: {results['bias_analysis']['aeroplane_bias']:.1f}%\n")
        f.write(f"  예측 다양성: {results['prediction_diversity']['unique_classes']}/20 클래스\n\n")
        
        f.write(f"클래스별 성능:\n")
        for class_idx, metrics in results['class_metrics'].items():
            f.write(f"  {metrics['class_name']}: P={metrics['precision']:.3f}, "
                   f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}\n")
    
    print(f"✅ 결과 저장 완료:")
    print(f"  📄 JSON: from_scratch_accurate_results.json")
    print(f"  📝 텍스트: from_scratch_accurate_report.txt")

def main():
    """메인 평가 함수"""
    print("🎯 FROM SCRATCH R-CNN 정확한 성능 측정 시작")
    print("="*60)
    
    # 테스트할 모델들
    models_to_test = [
        ('models/test_model_from_scratch.pth', 'From Scratch Model'),
        ('models/completely_debiased_model.pth', 'Debiased From Scratch'),
        ('models/improved_best_model.pth', 'Improved From Scratch')
    ]
    
    all_results = {}
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            print(f"\n🔄 평가 중: {model_name}")
            print("-" * 40)
            
            try:
                # 모델 로드
                model, device = load_from_scratch_model(model_path)
                
                # 종합 평가 실행
                results = evaluate_model_comprehensive(model, device, sample_size=1000)
                
                # 결과 출력
                print_detailed_results(results)
                
                # 결과 저장
                save_results_to_report(results, model_name)
                
                all_results[model_name] = results
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
        else:
            print(f"⚠️ Model not found: {model_path}")
    
    # 모델 비교 (여러 모델이 있는 경우)
    if len(all_results) > 1:
        print(f"\n🔄 모델 비교 요약:")
        print("="*80)
        
        for model_name, results in all_results.items():
            print(f"\n{model_name}:")
            print(f"  정확도: {results['overall_accuracy']:.2f}%")
            print(f"  예측 다양성: {results['prediction_diversity']['unique_classes']}/20")
            print(f"  Aeroplane 편향: {results['bias_analysis']['aeroplane_bias']:.1f}%")

if __name__ == "__main__":
    main() 