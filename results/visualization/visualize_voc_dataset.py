import matplotlib.pyplot as plt
import numpy as np
# seaborn 제거 - matplotlib만 사용
from matplotlib import font_manager
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# VOC2007 클래스 분포 데이터 (보고서 기준)
voc_classes = [
    'person', 'car', 'chair', 'diningtable', 'bottle', 
    'dog', 'bicycle', 'sofa', 'cat', 'bird',
    'tvmonitor', 'boat', 'pottedplant', 'horse', 'motorbike',
    'train', 'aeroplane', 'cow', 'sheep', 'bus'
]

# 각 클래스별 객체 수 (보고서 데이터 기준)
object_counts = [
    10674, 3185, 2806, 1629, 1518,
    1508, 1169, 1119, 1100, 1006,
    859, 850, 792, 652, 648,
    643, 642, 613, 553, 534
]

# 비율 계산
total_objects = sum(object_counts)
percentages = [count / total_objects * 100 for count in object_counts]

def create_class_distribution_visualization():
    """VOC2007 클래스 분포 시각화 생성"""
    
    # 컬러 팔레트 설정
    colors = plt.cm.Set3(np.linspace(0, 1, len(voc_classes)))
    
    # Figure 설정 (큰 사이즈)
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 막대 그래프 (상위 10개 클래스)
    plt.subplot(2, 2, 1)
    top_10_classes = voc_classes[:10]
    top_10_counts = object_counts[:10]
    
    bars = plt.bar(range(len(top_10_classes)), top_10_counts, 
                   color=colors[:10], alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('VOC2007 Top 10 Classes Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Objects', fontsize=12)
    plt.xticks(range(len(top_10_classes)), top_10_classes, rotation=45, ha='right')
    
    # 막대 위에 수치 표시
    for i, (bar, count) in enumerate(zip(bars, top_10_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count}\n({percentages[i]:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 2. 전체 클래스 가로 막대 그래프
    plt.subplot(2, 2, 2)
    y_pos = np.arange(len(voc_classes))
    
    bars = plt.barh(y_pos, object_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('All VOC2007 Classes Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Objects', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.yticks(y_pos, voc_classes)
    
    # 막대 끝에 수치 표시
    for i, (bar, count) in enumerate(zip(bars, object_counts)):
        plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{count} ({percentages[i]:.1f}%)', 
                ha='left', va='center', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    
    # 3. 파이 차트 (상위 10개 + 기타)
    plt.subplot(2, 2, 3)
    
    # 상위 10개와 나머지를 "Others"로 묶기
    pie_counts = object_counts[:10] + [sum(object_counts[10:])]
    pie_labels = voc_classes[:10] + ['Others (10 classes)']
    pie_colors = list(colors[:10]) + ['lightgray']
    
    # 파이 차트 그리기
    wedges, texts, autotexts = plt.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%',
                                      colors=pie_colors, startangle=90, 
                                      textprops={'fontsize': 10})
    
    plt.title('VOC2007 Class Distribution (Pie Chart)', fontsize=16, fontweight='bold', pad=20)
    
    # 4. 클래스 불균형 분석 차트
    plt.subplot(2, 2, 4)
    
    # 불균형 비율 계산 (person 기준)
    person_count = object_counts[0]
    imbalance_ratios = [person_count / count for count in object_counts]
    
    bars = plt.bar(range(len(voc_classes)), imbalance_ratios, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Class Imbalance Analysis (Ratio to Person)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Imbalance Ratio (vs Person)', fontsize=12)
    plt.xticks(range(len(voc_classes)), voc_classes, rotation=45, ha='right')
    
    # 기준선 추가
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Person baseline')
    plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5x imbalance')
    plt.axhline(y=10, color='purple', linestyle='--', alpha=0.7, label='10x imbalance')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.yscale('log')  # 로그 스케일로 표시
    
    plt.tight_layout()
    
    # 저장
    output_path = 'voc2007_class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 클래스 분포 시각화 저장: {output_path}")
    
    # 화면에 표시
    plt.show()
    
    return output_path

def create_detailed_statistics_table():
    """상세 통계 테이블 생성"""
    
    print("\n📊 VOC2007 데이터셋 클래스 분포 상세 통계")
    print("=" * 80)
    print(f"{'순위':<4} {'클래스':<12} {'객체 수':<8} {'비율':<8} {'불균형 비율':<12} {'시각화'}")
    print("-" * 80)
    
    person_count = object_counts[0]
    
    for i, (cls, count) in enumerate(zip(voc_classes, object_counts)):
        rank = i + 1
        percentage = count / total_objects * 100
        imbalance_ratio = person_count / count
        
        # 시각 막대 생성
        bar_length = int(percentage / 2)  # 50%를 25글자로 맞춤
        visual_bar = "█" * bar_length + "░" * max(0, 25 - bar_length)
        
        print(f"{rank:<4} {cls:<12} {count:<8} {percentage:<7.1f}% {imbalance_ratio:<11.1f}x {visual_bar}")
    
    print("-" * 80)
    print(f"총 객체 수: {total_objects:,}개")
    print(f"총 클래스 수: {len(voc_classes)}개")
    print(f"최대 불균형 비율: {person_count / min(object_counts):.1f}:1 (person vs bus)")
    print(f"상위 5개 클래스 집중도: {sum(percentages[:5]):.1f}%")

def analyze_class_imbalance():
    """클래스 불균형 분석"""
    
    print("\n🔍 클래스 불균형 분석")
    print("=" * 60)
    
    person_count = object_counts[0]
    bus_count = object_counts[-1]
    
    print(f"📈 최빈 클래스: person ({person_count:,}개, {percentages[0]:.1f}%)")
    print(f"📉 최소 클래스: bus ({bus_count:,}개, {percentages[-1]:.1f}%)")
    print(f"⚖️  불균형 비율: {person_count / bus_count:.1f}:1")
    print(f"📊 상위 5개 클래스가 전체의 {sum(percentages[:5]):.1f}% 차지")
    
    # 불균형 수준별 분류
    severe_imbalance = []  # 10배 이상
    moderate_imbalance = []  # 5-10배
    mild_imbalance = []  # 2-5배
    balanced = []  # 2배 미만
    
    for i, (cls, count) in enumerate(zip(voc_classes, object_counts)):
        ratio = person_count / count
        if ratio >= 10:
            severe_imbalance.append(cls)
        elif ratio >= 5:
            moderate_imbalance.append(cls)
        elif ratio >= 2:
            mild_imbalance.append(cls)
        else:
            balanced.append(cls)
    
    print(f"\n🔴 심각한 불균형 (10배 이상): {len(severe_imbalance)}개 클래스")
    print(f"   {', '.join(severe_imbalance)}")
    print(f"🟡 보통 불균형 (5-10배): {len(moderate_imbalance)}개 클래스")
    print(f"   {', '.join(moderate_imbalance)}")
    print(f"🟢 경미한 불균형 (2-5배): {len(mild_imbalance)}개 클래스")
    print(f"   {', '.join(mild_imbalance)}")
    print(f"✅ 균형적 (2배 미만): {len(balanced)}개 클래스")
    print(f"   {', '.join(balanced)}")

if __name__ == "__main__":
    print("🎨 VOC2007 데이터셋 클래스 분포 시각화 시작...")
    
    # 1. 시각화 생성
    output_file = create_class_distribution_visualization()
    
    # 2. 상세 통계 출력
    create_detailed_statistics_table()
    
    # 3. 불균형 분석
    analyze_class_imbalance()
    
    print(f"\n✅ 시각화 완료! 파일 저장: {output_file}")
    print("📊 모든 분석이 완료되었습니다.") 