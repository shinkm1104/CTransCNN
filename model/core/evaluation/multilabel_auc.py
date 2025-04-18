# import torch
# import numpy as np
# from sklearn.metrics import roc_auc_score
# import warnings


# def add_multi_label_auc(pred, target, thr=None, k=None):
#     if thr is None and k is None:
#         thr = 0.5
#         warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
#                       'default.')
#     elif thr is not None and k is not None:
#         warnings.warn('Both thr and k are given, use threshold in favor of '
#                       'top-k.')

#     if thr is not None:
#         # a label is predicted positive if the confidence is no lower than thr
#         pos_inds = pred >= thr

#     else:
#         # top-k labels will be predicted positive for any example
#         sort_inds = np.argsort(-pred, axis=1)
#         sort_inds_ = sort_inds[:, :k]
#         inds = np.indices(sort_inds_.shape)
#         pos_inds = np.zeros_like(pred)
#         pos_inds[inds[0], sort_inds_] = 1

#     if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
#         pred = pred.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
#     elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
#         raise TypeError('pred and target should both be torch.Tensor or'
#                         'np.ndarray')

#     total_auc = 0.

#     for i in range(target.shape[1]):
#         try:
#             auc = roc_auc_score(target[:, i], pred[:, i])
#         except ValueError:
#             auc = 0.5
#         total_auc += auc
#     multi_auc = total_auc / target.shape[1]
#     return multi_auc

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings
import matplotlib.pyplot as plt
import os


def add_multi_label_auc(pred, target, thr=None, k=None, save_dir=None, disease_labels=None):
    """
    다중 레이블 분류 모델의 AUC 계산 및 ROC 곡선 이미지 저장
    
    Args:
        pred (torch.Tensor | np.ndarray): 예측 확률
        target (torch.Tensor | np.ndarray): 실제 타겟 레이블
        thr (float, optional): 임계값
        k (int, optional): 상위 k개
        save_dir (str, optional): ROC 곡선 이미지 저장 경로
        disease_labels (list, optional): 질병 레이블 이름 목록
    
    Returns:
        float: 평균 AUC 점수
    """
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr
    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1
    print('0')
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    # 저장 디렉토리 생성
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        roc_curves_dir = os.path.join(save_dir, 'roc_curves')
        os.makedirs(roc_curves_dir, exist_ok=True)

    
    # 기본 질병 레이블 설정
    n_classes = target.shape[1]
    # if disease_labels is None or len(disease_labels) != n_classes:
    #     disease_labels = [f"Class_{i}" for i in range(n_classes)]

    # NIH ChestXray 데이터셋의 질병 레이블
    disease_labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 
        'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
        'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]

    total_auc = 0.
    auc_values = []


    for i in range(target.shape[1]):
        try:
            # ROC AUC 계산
            auc_score = roc_auc_score(target[:, i], pred[:, i])
            auc_values.append(auc_score)
            
            # ROC 곡선 이미지 저장
            if save_dir is not None:
                fpr, tpr, _ = roc_curve(target[:, i], pred[:, i])
                roc_auc = auc(fpr, tpr)

                # 그래프 생성
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {disease_labels[i]}')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)

                # 저장될 파일 이름 생성
                safe_label = disease_labels[i].replace(' ', '_').replace('/', '_')
                roc_curve_path = os.path.join(roc_curves_dir, f'roc_curve_{safe_label}.png')

                # 저장 전 확인 출력 (중요)
                print(f"Saving ROC curve for {safe_label} at {roc_curve_path}")

                plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
                plt.close()

        except ValueError:
            print(f"Warning: Class {disease_labels[i]} could not compute AUC (possibly one class only)")
            auc_score = 0.5
            auc_values.append(auc_score)
        
        total_auc += auc_score

    
    # 평균 AUC 계산
    multi_auc = total_auc / target.shape[1]

    
    # AUC 종합 이미지 생성 (전체 클래스 ROC 곡선)
    if save_dir is None :
        pass
        # save_dir = "/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/visualizations"
    else:
    # if save_dir is not None:
        # 1. 모든 클래스의 ROC 곡선을 하나의 그래프에 표시
        # print('here')
        plt.figure(figsize=(12, 10))
        for i in range(target.shape[1]):
            try:
                fpr, tpr, _ = roc_curve(target[:, i], pred[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1.5, alpha=0.7, 
                        label=f'{disease_labels[i]} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Error plotting ROC curve for {disease_labels[i]}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'all_roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # print('ttt')
        # 2. AUC 값 막대 그래프
        plt.figure(figsize=(14, 8))
        
        # AUC 값에 따라 정렬
        sorted_indices = np.argsort(auc_values)
        sorted_diseases = [disease_labels[i] for i in sorted_indices]
        sorted_aucs = [auc_values[i] for i in sorted_indices]
        
        bars = plt.bar(sorted_diseases, sorted_aucs, color='skyblue')
        plt.axhline(y=np.mean(auc_values), color='r', linestyle='-', 
                   label=f'Mean AUC: {np.mean(auc_values):.3f}')
        plt.xlabel('Disease Classes')
        plt.ylabel('AUC Score')
        plt.title('AUC Scores by Disease Class')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        # print('there')
        # 막대 위에 값 표시
        for bar in bars:
            print(bar)
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.savefig(os.path.join(save_dir, 'auc_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 결과 요약 텍스트 파일 저장
        with open(os.path.join(save_dir, 'auc_results.txt'), 'w') as f:
            f.write("Multi-Label ROC AUC Results\n")
            f.write("=========================\n\n")
            f.write(f"Mean AUC: {multi_auc:.4f}\n\n")
            f.write("Class-wise AUC:\n")
            for i, label in enumerate(disease_labels):
                f.write(f"{label}: {auc_values[i]:.4f}\n")
        
        print(f"ROC 곡선 이미지가 {save_dir}에 저장되었습니다.")
    
    return multi_auc


# 사용 예시:
if __name__ == "__main__":
    # 임의의 데이터로 테스트
    pred = np.random.rand(100, 14)  # 예시: 100개 샘플, 14개 클래스
    target = np.random.randint(0, 2, (100, 14))  # 이진 레이블
    
    # NIH ChestXray 데이터셋의 질병 레이블
    disease_labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 
        'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
        'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]
    
    # AUC 계산 및 ROC 곡선 이미지 저장
    save_dir = "/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/visualization"
    # multi_auc = add_multi_label_auc(pred, target, save_dir, disease_labels)
    
    # print(f"평균 AUC: {multi_auc:.4f}")
