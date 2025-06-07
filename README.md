# VOC2007 From Scratch R-CNN Implementation

## **프로젝트 개요**

이 프로젝트는 **PASCAL VOC 2007 데이터셋**을 이용하여 **완전한 From Scratch R-CNN**을 구현한 프로젝트입니다. Transfer Learning 없이 처음부터 학습하여 객체 검출의 기본 원리를 이해하고, 실제 머신러닝 프로젝트에서 발생하는 편향 문제를 해결합니다.

##  **프로젝트 구조**

```
rcnn_implementation/
├── 📂 data/                   # VOC2007 데이터셋 처리
│   └── voc_dataset.py        # VOC 데이터 로더 및 전처리
├── 📂 models/                 # 모델 정의 및 체크포인트
│   ├── rcnn.py              # R-CNN 모델 아키텍처
│   └── *.pth                # 훈련된 모델 체크포인트들
├── 📂 src/                    # 핵심 구현 코드
├── 📂 utils/                  # 유틸리티 함수들
│   ├── data_prep.py         # 데이터 전처리
│   ├── gpu_utils.py         # GPU/MPS 최적화
│   └── selective_search.py  # 영역 제안 알고리즘
├── train_rcnn.py             # 메인 훈련 스크립트
├── requirements.txt          # 의존성 패키지
└── README.md                # 이 파일
```

### **1. 환경 설정**
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

### **모델 훈련**
```bash
python train_rcnn.py
```

### **2. 결과 확인**
```bash
# 성능 평가
cd final_results/evaluation/
python accurate_from_scratch_evaluation.py

# 시각화 생성
cd final_results/visualization/
python create_visual_prediction_results.py
```
### **모델 아키텍처**
- **백본**: ResNet-50 (pretrained=False)
- **분류기**: 3층 FC 네트워크 (4096 → 4096 → 21 클래스)
- **회귀기**: 바운딩 박스 좌표 예측
- **초기화**: Xavier Normal (수치 안정성 확보)

### **훈련 설정**
- **에포크**: 50
- **옵티마이저**: SGD (lr=0.001, momentum=0.9)
- **스케줄러**: StepLR (step_size=15, gamma=0.1)
- **배치 크기**: 16 (메모리 최적화)
- **디바이스**: Apple M3 Pro with MPS

### **데이터 전처리**
- **이미지 크기**: 224×224 (ResNet 표준)
- **정규화**: ImageNet 평균/표준편차
- **영역 제안**: Selective Search 알고리즘
- **클래스 균형**: 가중치 기반 샘플링

### **Transfer Learning과의 비교**
| 접근법 | 정확도 |
|--------|--------|------|
| **From Scratch** | 17.30% |
| **Transfer Learning** | 38.83% |
| **성능 차이** | 2.2배 |

## **시각화 자료**

### **생성 가능한 시각화**
1. **VOC2007 클래스 분포**: 데이터셋 불균형 분석
2. **예측 결과 샘플**: 성공/실패 사례 시각화
3. **성능 분석 차트**: 클래스별 F1-Score, 예측 분포

### **시각화 재생성**
```bash
cd final_results/visualization/
python create_visual_prediction_results.py  # 모든 차트 생성
python visualize_voc_dataset.py            # 데이터셋 분포
```

## 🛠️ **개발 환경**

### **하드웨어**
- **CPU**: Apple M3 Pro
- **메모리**: 18GB 통합 메모리
- **가속**: MPS (Metal Performance Shaders)

### **소프트웨어**
- **Python**: 3.10+
- **PyTorch**: 2.7.1
- **주요 라이브러리**: torchvision, matplotlib, numpy, PIL

### **의존성**
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
Pillow>=8.0.0
tqdm>=4.60.0
```

## **참고 자료**

### **논문**
- Girshick et al. "Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)
- PASCAL VOC Challenge Documentation

### **데이터셋**
- PASCAL VOC 2007: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
---
**개발 기간**: 2025년 6월  
**최종 업데이트**: 2025-06-07  
**개발 환경**: Apple M3 Pro, macOS Sonoma, PyTorch 2.7.1
