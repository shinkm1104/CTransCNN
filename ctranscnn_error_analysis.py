import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from mmcv import Config
from mmcv.runner import load_checkpoint
from model.models import build_classifier  # 프로젝트의 빌드 함수 사용
from PIL import Image
from sklearn.metrics import roc_auc_score
import cv2

# 저장 디렉토리 설정
BASE_RESULT_DIR = "/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/result/error_analysis"
os.makedirs(BASE_RESULT_DIR, exist_ok=True)
plots_dir = os.path.join(BASE_RESULT_DIR, "plots")
metrics_dir = os.path.join(BASE_RESULT_DIR, "metrics")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# 질병 클래스 정의 (NIH 데이터셋에 맞게 수정)
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# GPU 설정
device = 'cuda:2'

def load_model_and_predict():
    """커스텀 모델 방식으로 로드 및 테스트 이미지 예측"""
    print("Loading model and making predictions...")
    
    # 설정 파일 로드
    config_file = "configs/NIH_ChestX-ray14_CTransCNN.py"
    checkpoint_file = "/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/epoch_30.pth"
    
    # 구성 로드
    cfg = Config.fromfile(config_file)
    print("Model config:", cfg.model.type)
    
    # 모델 빌드 (원본 코드와 동일한 방식)
    model = build_classifier(cfg.model)
    print(f"Model built successfully: {type(model)}")
    
    # 가중치 로드
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    print("Checkpoint loaded successfully")
    
    # GPU 설정
    model.to(device)
    model.eval()
    
    # 테스트 이미지 목록 로드
    test_list_path = "/userHome/userhome4/kyoungmin/code/Xray/dataset/test_list.txt"
    with open(test_list_path, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(test_files)} test files")
    
    # NIH 데이터셋 경로
    dataset_path = "/userHome/userhome4/kyoungmin/code/Xray/dataset"
    
    # 이미지 디렉토리 탐색
    image_dirs = []
    for root, dirs, files in os.walk(dataset_path):
        if 'images' in dirs:
            image_dirs.append(os.path.join(root, 'images'))
    
    print(f"Found {len(image_dirs)} image directories")
    
    # 예측 결과 저장 리스트
    all_probs = []
    found_images = []
    
    def preprocess_image(img_path):
        """이미지 전처리 함수"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # config에 맞게 수정
        img = img.astype(np.float32) / 255.0  # 정규화
        
        # ImageNet 평균 및 표준편차로 정규화
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 형태 변환 (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # 배치 차원 추가
        img = np.expand_dims(img, axis=0)
        
        # 중요: 명시적으로 float32 데이터 타입 지정
        return torch.tensor(img, dtype=torch.float32)
    
    # 이미지 로드 및 예측
    for i, filename in enumerate(tqdm(test_files, desc="Predicting")):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(test_files)}")
        
        # 이미지 파일 찾기
        img_path = None
        for img_dir in image_dirs:
            candidate_path = os.path.join(img_dir, filename)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break
        
        if img_path is None:
            print(f"Warning: Image {filename} not found")
            continue
        
        # 이미지 로드 및 예측
        try:
            # 이미지 전처리
            img_tensor = preprocess_image(img_path)
            img_tensor = img_tensor.to(device)
            
            # 모델 추론
            with torch.no_grad():
                output = model(img_tensor, return_loss=False)
            
            # 출력 형식 확인 및 처리
            if isinstance(output, torch.Tensor):
                probs = torch.sigmoid(output).cpu().numpy()[0]
            elif isinstance(output, tuple):
                probs = torch.sigmoid(output[0]).cpu().numpy()[0]
            else:
                probs = torch.sigmoid(torch.tensor(output, device=device)).cpu().numpy()[0]
            
            all_probs.append(probs)
            found_images.append(filename)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # 예측 결과를 numpy 배열로 변환
    all_probs = np.array(all_probs)
    print(f"Made predictions for {len(all_probs)} images")
    
    # 레이블 로드
    all_targets = load_test_labels(found_images)
    
    # 결과 저장
    result_file = os.path.join(BASE_RESULT_DIR, "predictions.pkl")
    with open(result_file, 'wb') as f:
        pickle.dump({
            'probs': all_probs,
            'targets': all_targets,
            'filenames': found_images,
            'disease_labels': disease_labels
        }, f)
    
    print(f"Saved predictions to {result_file}")
    
    return all_probs, all_targets


def load_test_labels(test_files):
    """테스트 이미지에 대한 정답 레이블 로드"""
    print("Loading ground truth labels...")
    
    # NIH 데이터셋 정보 파일 로드
    data_path = "/userHome/userhome4/kyoungmin/code/Xray/dataset/Data_Entry_2017.csv"
    try:
        labels_df = pd.read_csv(data_path)
        print(f"Loaded labels dataframe with shape {labels_df.shape}")
        
        # 컬럼명 출력
        print("Columns in CSV:", labels_df.columns.tolist())
        
        # 이미지 ID 컬럼 확인
        img_col = None
        for col in labels_df.columns:
            if 'Image' in col:
                img_col = col
                break
                
        if img_col is None:
            print("Error: Could not find image column")
            return np.zeros((len(test_files), len(disease_labels)))
            
        print(f"Using '{img_col}' as image identifier column")
        
        # 질병 레이블 컬럼 확인
        finding_col = None
        for col in labels_df.columns:
            if 'Finding' in col:
                finding_col = col
                break
                
        if finding_col is None:
            print("Error: Could not find findings column")
            return np.zeros((len(test_files), len(disease_labels)))
            
        print(f"Using '{finding_col}' as findings column")
        
        # 질병 레이블 원-핫 인코딩
        for disease in disease_labels: 
            labels_df[disease] = labels_df[finding_col].apply(
                lambda x: 1 if disease in str(x) else 0
            )
        
        # 테스트 파일과 동일한 순서로 레이블 배열 생성
        test_labels = np.zeros((len(test_files), len(disease_labels)))
        matched_count = 0
        
        for i, img_name in enumerate(test_files):
            # 이미지 이름에서 확장자와 경로 제거
            img_base = os.path.basename(img_name)
            
            # 완전 일치 시도
            match = labels_df[labels_df[img_col] == img_base]
            
            # 일치하지 않으면 부분 일치 시도
            if len(match) == 0:
                match = labels_df[labels_df[img_col].str.contains(img_base, na=False)]
            
            if len(match) > 0:
                matched_count += 1
                row = match.iloc[0]
                for j, disease in enumerate(disease_labels):
                    test_labels[i, j] = row[disease]
        
        print(f"Matched {matched_count}/{len(test_files)} test files with labels")
        
        if matched_count == 0:
            print("Error: No test files matched with labels!")
            return np.zeros((len(test_files), len(disease_labels)))
        
        return test_labels
        
    except Exception as e:
        print(f"Error loading labels: {e}")
        return np.zeros((len(test_files), len(disease_labels)))

# 오류 계산 함수
def calculate_sample_errors(probs, targets):
    """
    교수님 지시에 따라 각 샘플의 오류 크기 계산:
    - 정답이 0인데 예측이 양수 -> 오류 = 예측값
    - 정답이 1인데 예측이 음수 -> 오류 = 1-예측값
    """
    n_samples = probs.shape[0]
    n_classes = probs.shape[1]
    
    # logit값으로 변환 (시그모이드 역변환)
    epsilon = 1e-7  # 수치 안정성을 위한 작은 값
    logits = np.log((probs + epsilon) / (1 - probs + epsilon))
    
    sample_errors = []
    
    for i in range(n_samples):
        errors = []
        
        for c in range(n_classes):
            true_label = targets[i, c]
            net_value = logits[i, c]
            
            if true_label == 0:
                # 정답이 0일 때, 양수 net value가 오류
                error = max(0, net_value)
            else:
                # 정답이 1일 때, 음수 net value가 오류 (절대값 적용)
                error = max(0, -net_value)
            
            errors.append(error)
        
        # 샘플의 전체 오류 = 각 클래스 오류의 평균
        sample_errors.append(np.mean(errors))
    
    return np.array(sample_errors)

# 상위 오류 샘플 분석
def analyze_top_errors(test_probs, test_targets):
    print("Analyzing top error samples...")
    
    # 오류 계산
    sample_errors = calculate_sample_errors(test_probs, test_targets)
    
    # 오류가 높은 순서대로 인덱스 정렬
    sorted_indices = np.argsort(sample_errors)[::-1]  # 내림차순 정렬
    
    # 상위 20% 추출
    top_20_percent = int(len(sorted_indices) * 0.2)
    high_error_indices = sorted_indices[:top_20_percent]
    
    print(f"Extracted top {top_20_percent} samples with highest errors ({top_20_percent/len(sorted_indices)*100:.1f}%)")
    
    # 이진 예측값 생성
    binary_preds = (test_probs >= 0.5).astype(int)
    
    # 클래스별 오류 패턴 분석
    class_error_counts = {cls: 0 for cls in disease_labels}
    false_positives = {cls: 0 for cls in disease_labels}
    false_negatives = {cls: 0 for cls in disease_labels}
    
    # 상위 20% 오류 샘플에서 클래스별 오류 패턴 분석
    for idx in high_error_indices:
        for c, cls_name in enumerate(disease_labels):
            true_label = test_targets[idx, c]
            pred_label = binary_preds[idx, c]
            
            if pred_label != true_label:
                class_error_counts[cls_name] += 1
                
                if true_label == 0 and pred_label == 1:
                    false_positives[cls_name] += 1
                elif true_label == 1 and pred_label == 0:
                    false_negatives[cls_name] += 1
    
    # 오류 패턴 시각화 및 저장
    plt.figure(figsize=(15, 10))
    
    # 총 오류 수 기준 클래스 정렬
    sorted_classes = sorted(class_error_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [item[0] for item in sorted_classes]
    
    # 거짓 양성과 거짓 음성 데이터 준비
    fp_data = [false_positives[cls] for cls in sorted_class_names]
    fn_data = [false_negatives[cls] for cls in sorted_class_names]
    
    # 누적 막대 그래프 그리기
    bar_width = 0.8
    plt.bar(sorted_class_names, fp_data, bar_width, label='False Positive')
    plt.bar(sorted_class_names, fn_data, bar_width, bottom=fp_data, label='False Negative')
    
    plt.xlabel('Disease Class')
    plt.ylabel('Number of Errors')
    plt.title('Error Patterns by Disease Class (Top 20% Error Samples)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 그래프 저장
    error_plot_file = os.path.join(plots_dir, "ctranscnn_error_patterns.png")
    plt.savefig(error_plot_file)
    plt.close()
    
    # 상위 20% 오류 샘플 정보 저장
    high_error_df = pd.DataFrame({
        'Index': high_error_indices,
        'Error_Score': sample_errors[high_error_indices]
    })
    
    for i, cls in enumerate(disease_labels):
        high_error_df[f'True_{cls}'] = [test_targets[idx, i] for idx in high_error_indices]
        high_error_df[f'Pred_{cls}'] = [test_probs[idx, i] for idx in high_error_indices]
    
    high_error_file = os.path.join(metrics_dir, "ctranscnn_high_error_samples.csv")
    high_error_df.to_csv(high_error_file)
    
    print(f"✅ Error analysis saved to '{error_plot_file}' and '{high_error_file}'")
    
    return class_error_counts, false_positives, false_negatives

# 다중 레이블 분류 지표 계산
def calculate_multilabel_metrics(test_probs, test_targets):
    print("Calculating multilabel classification metrics...")
    
    # 예측 결과 이진화 (임계값 0.5)
    binary_preds = (test_probs >= 0.5).astype(int)
    
    # 다중 레이블 분류 지표 계산
    def hamming_loss(y_true, y_pred):
        return np.mean(np.not_equal(y_true, y_pred))

    def ranking_loss(y_true, y_pred_scores):
        n_samples = y_true.shape[0]
        total_pairs = 0
        total_wrong_pairs = 0
        
        for i in range(n_samples):
            relevant = np.where(y_true[i] == 1)[0]
            irrelevant = np.where(y_true[i] == 0)[0]
            
            if len(relevant) == 0 or len(irrelevant) == 0:
                continue
            
            pairs = len(relevant) * len(irrelevant)
            wrong_pairs = 0
            
            for r in relevant:
                for ir in irrelevant:
                    if y_pred_scores[i, r] <= y_pred_scores[i, ir]:
                        wrong_pairs += 1
            
            total_pairs += pairs
            total_wrong_pairs += wrong_pairs
        
        return total_wrong_pairs / total_pairs if total_pairs > 0 else 0

    def multilabel_accuracy(y_true, y_pred):
        n_samples = y_true.shape[0]
        accuracies = []
        
        for i in range(n_samples):
            intersection = np.sum(np.logical_and(y_true[i], y_pred[i]))
            union = np.sum(np.logical_or(y_true[i], y_pred[i]))
            
            if union == 0:
                accuracies.append(1.0)  # 모든 레이블이 0인 경우
            else:
                accuracies.append(intersection / union)
        
        return np.mean(accuracies)

    def one_error(y_true, y_pred_scores):
        n_samples = y_true.shape[0]
        errors = 0
        
        for i in range(n_samples):
            top_label = np.argmax(y_pred_scores[i])
            if y_true[i, top_label] == 0:
                errors += 1
        
        return errors / n_samples
    
    def multilabel_coverage(y_true, y_pred_scores, k=None):
        n_samples, n_classes = y_true.shape
        if k is None:
            k = n_classes
        coverage = 0
        
        for i in range(n_samples):
            # 관련 레이블(실제 값이 1인 레이블) 찾기
            relevant = np.where(y_true[i] == 1)[0]
            if len(relevant) == 0:
                continue
            
            # 예측 점수에 따라 레이블 순위 매기기
            ranks = np.argsort(y_pred_scores[i])[::-1]
            
            # 관련 레이블의 최대 랭크 찾기
            max_rank = 0
            for label in relevant:
                rank = np.where(ranks == label)[0][0] + 1  # 0-indexed를 1-indexed로 변환
                max_rank = max(max_rank, rank)
            
            coverage += max_rank
        
        return coverage / n_samples

    def subset_accuracy(y_true, y_pred):
        """예측 레이블 세트가 실제 레이블 세트와 정확히 일치하는 비율"""
        return np.mean(np.all(y_true == y_pred, axis=1))

    def micro_f1_score(y_true, y_pred):
        """모든 샘플과 클래스에 대한 전체 TP, FP, FN으로 단일 F1 계산"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if micro_precision + micro_recall == 0:
            return 0
        
        return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    
    # 다중 레이블 분류 지표 계산
    metrics = {
        'hamming_loss': hamming_loss(test_targets, binary_preds),
        'ranking_loss': ranking_loss(test_targets, test_probs),
        'multilabel_accuracy': multilabel_accuracy(test_targets, binary_preds),
        'one_error': one_error(test_targets, test_probs),
        'multilabel_coverage': multilabel_coverage(test_targets, test_probs),
        'subset_accuracy': subset_accuracy(test_targets, binary_preds),
        'micro_f1_score': micro_f1_score(test_targets, binary_preds)
    }
    
    # 클래스별 AUC-ROC 계산
    class_aucs = {}
    for i, cls in enumerate(disease_labels):
        try:
            auc_score = roc_auc_score(test_targets[:, i], test_probs[:, i])
            class_aucs[cls] = auc_score
        except:
            class_aucs[cls] = float('nan')  # 한 클래스만 있는 경우
    
    # 클래스별 정밀도, 재현율, F1 점수 계산
    precision_values = []
    recall_values = []
    f1_values = []
    
    for i, cls in enumerate(disease_labels):
        true_pos = np.sum((test_targets[:, i] == 1) & (binary_preds[:, i] == 1))
        false_pos = np.sum((test_targets[:, i] == 0) & (binary_preds[:, i] == 1))
        false_neg = np.sum((test_targets[:, i] == 1) & (binary_preds[:, i] == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    
    # Macro F1 Score 계산 및 추가
    macro_f1 = np.mean(f1_values)
    metrics['macro_f1_score'] = macro_f1
    
    # 성능 테이블 생성
    performance_table = pd.DataFrame(index=disease_labels)
    performance_table['AUC'] = [class_aucs[cls] for cls in disease_labels]
    performance_table['Precision'] = precision_values
    performance_table['Recall'] = recall_values
    performance_table['F1'] = f1_values
    
    # Macro 평균값 계산
    performance_table.loc['Macro_Average'] = performance_table.mean()
    
    # 결과 저장
    metrics_file = os.path.join(metrics_dir, "ctranscnn_multilabel_metrics.json")
    with open(metrics_file, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    performance_file = os.path.join(metrics_dir, "ctranscnn_performance_table.csv")
    performance_table.to_csv(performance_file)
    
    # ROC-AUC 시각화
    plt.figure(figsize=(15, 8))
    sorted_aucs = sorted(class_aucs.items(), key=lambda x: x[1])
    cls_names = [item[0] for item in sorted_aucs]
    auc_values = [item[1] for item in sorted_aucs]
    
    plt.bar(cls_names, auc_values)
    macro_auc = np.nanmean(list(class_aucs.values()))
    plt.axhline(y=macro_auc, color='r', linestyle='-', label=f'Macro AUC: {macro_auc:.4f}')
    plt.xlabel('Disease Class')
    plt.ylabel('AUC-ROC')
    plt.title('AUC-ROC by Disease Class')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    
    auc_plot_file = os.path.join(plots_dir, "ctranscnn_auc_by_class.png")
    plt.savefig(auc_plot_file)
    plt.close()
    
    print(f"✅ Metrics saved to '{metrics_file}' and '{performance_file}'")
    print(f"✅ AUC plot saved to '{auc_plot_file}'")
    
    return metrics, performance_table, class_aucs

# 저장된 예측 로드 또는 새 예측 실행
def load_or_run_predictions():
    prediction_file = os.path.join(BASE_RESULT_DIR, "predictions.pkl")
    
    if os.path.exists(prediction_file):
        print(f"Loading existing predictions from {prediction_file}")
        with open(prediction_file, 'rb') as f:
            results = pickle.load(f)
        return results['probs'], results['targets']
    else:
        print("No saved predictions found. Running model predictions...")
        return load_model_and_predict()

# 메인 함수
def main():
    print("\n===== CTransCNN 모델 표준 오차 분석 시작 =====\n")
    
    # 예측 결과 로드 또는 새로 실행
    test_probs, test_targets = load_or_run_predictions()
    
    # 상위 오류 샘플 분석
    class_errors, false_pos, false_neg = analyze_top_errors(test_probs, test_targets)
    
    # 다중 레이블 분류 지표 계산
    metrics, performance_table, class_aucs = calculate_multilabel_metrics(test_probs, test_targets)
    
    print("\n===== 분석 완료 =====")
    print(f"결과 저장 위치: {BASE_RESULT_DIR}")
    
    # 주요 지표 출력
    print("\n주요 다중 레이블 분류 지표:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
