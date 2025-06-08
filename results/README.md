# VOC2007 From Scratch R-CNN - 최종 결과 자료

## 📂 **폴더 구조**

###  `visualization/` - 시각화 관련
- **`create_visual_prediction_results.py`** - 종합 시각화 생성 코드
- **`visualize_voc_dataset.py`** - VOC2007 데이터셋 분포 시각화
- **`performance_analysis_charts.png`** - 성능 분석 종합 차트
- **`prediction_samples_grid.png`** - 실제 예측 결과 샘플
- **`voc2007_class_distribution.png`** - VOC2007 클래스 분포 차트

###  `evaluation/` - 성능 평가 관련
- **`accurate_from_scratch_evaluation.py`** - 정확한 From Scratch 성능 측정
- **`evaluate_test_set.py`** - 테스트셋 종합 평가
- **`fix_bias_completely.py`** - 편향 문제 해결 코드
- **`test_final_model.py`** - 최종 모델 테스트

###  `reports/` - 문서 및 보고서
- **`README.md`** - 프로젝트 전체 설명서


##  **사용 방법**

### **시각화 재생성**
```bash
cd visualization/
python create_visual_prediction_results.py
```

### **성능 재평가**
```bash
cd evaluation/
python accurate_from_scratch_evaluation.py
```

### **데이터셋 분포 확인**
```bash
cd visualization/
python visualize_voc_dataset.py
```
---

**생성일**: 2025-06-07  
**환경**: Apple M3 Pro, PyTorch 2.7.1, MPS 가속 
