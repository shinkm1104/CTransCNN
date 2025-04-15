import warnings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os

def average_performance(pred, target, thr=None, k=None, save_dir=None, disease_labels=None):
    """
    Calculate multilabel classification metrics according to exact formulas.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape (N, C)
        target (torch.Tensor | np.ndarray): The target with shape (N, C)
        thr (float): Threshold for prediction. Defaults to None.
        k (int): Top-k performance. Defaults to None.
        save_dir (str): Directory to save plots and metric summaries. Defaults to None.
        disease_labels (list): List of disease names for labels. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1, multi_auc, hamming_loss, ranking_loss, 
                multilabel_accuracy, multilabel_coverage, one_error, subset_accuracy, 
                macro_f1, micro_f1)
    """
    # Convert to numpy if input is torch.Tensor
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or np.ndarray')
    
    # Default threshold
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, using threshold in favor of top-k.')
    
    assert pred.shape == target.shape, 'pred and target should have the same shape.'
    n_samples, n_classes = target.shape
    eps = np.finfo(np.float32).eps
    
    # Replace -1 labels with 0
    target[target == -1] = 0
    
    # Use external disease_labels if provided, else default to NIH ChestXray labels
    if disease_labels is None:
        disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 
                          'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
                          'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
    
    # Compute predicted labels using threshold or top-k method
    if thr is not None:
        pred_labels = (pred >= thr).astype(int)
        pos_inds = pred >= thr
    else:
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        pred_labels = np.zeros_like(pred)
        inds = np.indices(sort_inds_.shape)
        pred_labels[inds[0], sort_inds_] = 1
        pos_inds = pred_labels.copy()
    
    # Calculate TP, FP, FN and per-class precision and recall
    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1
    precision_class = tp.sum(axis=0) / np.maximum(tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(tp.sum(axis=0) + fn.sum(axis=0), eps)
    
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    
    # Multi-label AUC and ROC curves for each class
    auc_values = []
    total_auc = 0.0
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        roc_curves_dir = os.path.join(save_dir, 'roc_curves')
        os.makedirs(roc_curves_dir, exist_ok=True)
    
    for i in range(n_classes):
        try:
            auc_score = roc_auc_score(target[:, i], pred[:, i])
        except ValueError:
            warnings.warn(f"Class {disease_labels[i]}: only one class present; AUC set to 0.5")
            auc_score = 0.5
        auc_values.append(auc_score)
        total_auc += auc_score
        
        if save_dir is not None:
            fpr, tpr, _ = roc_curve(target[:, i], pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {disease_labels[i]}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            safe_label = disease_labels[i].replace(' ', '_').replace('/', '_')
            roc_curve_path = os.path.join(roc_curves_dir, f'roc_curve_{safe_label}.png')
            print(f"Saving ROC curve for {safe_label} at {roc_curve_path}")
            plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    multi_auc = total_auc / n_classes
    
    # Plot overall ROC curves and AUC bar graph
    if save_dir is None:
        save_dir = "/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/visualizations"
    if save_dir is not None:
        # All ROC curves in one plot
        plt.figure(figsize=(12, 10))
        for i in range(n_classes):
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
        
        # AUC bar graph
        plt.figure(figsize=(14, 8))
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
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom')
        plt.savefig(os.path.join(save_dir, 'auc_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save AUC results to text file
        with open(os.path.join(save_dir, 'auc_results.txt'), 'w') as f:
            f.write("Multi-Label ROC AUC Results\n")
            f.write("=========================\n\n")
            f.write(f"Mean AUC: {multi_auc:.4f}\n\n")
            f.write("Class-wise AUC:\n")
            for i, label in enumerate(disease_labels):
                f.write(f"{label}: {auc_values[i]:.4f}\n")
    
    # ---------------------------
    # Compute additional performance metrics
    # ---------------------------
    # Hamming Loss: misclassification rate over all labels
    hamming_loss_value = 0.0
    for i in range(n_samples):
        for j in range(n_classes):
            if target[i, j] != pred_labels[i, j]:
                hamming_loss_value += 1
    hamming_loss_value = (hamming_loss_value / (n_samples * n_classes)) * 100.0

    # One Error: percentage of samples where the top predicted label is not true
    one_error = 0.0
    for i in range(n_samples):
        top_label = np.argmax(pred[i])
        if target[i, top_label] == 0:
            one_error += 1
    one_error = (one_error / n_samples) * 100.0

    # Multilabel Accuracy (Jaccard similarity per sample)
    multilabel_accuracy = 0.0
    for i in range(n_samples):
        intersection = np.sum(np.logical_and(target[i], pred_labels[i]))
        union = np.sum(np.logical_or(target[i], pred_labels[i]))
        multilabel_accuracy += (intersection / union) if union > 0 else 1.0
    multilabel_accuracy = (multilabel_accuracy / n_samples) * 100.0

    # Subset Accuracy: percentage of samples with exact label match
    subset_accuracy = 0.0
    for i in range(n_samples):
        if np.array_equal(target[i], pred_labels[i]):
            subset_accuracy += 1
    subset_accuracy = (subset_accuracy / n_samples) * 100.0

    # Macro F1 Score and per-class F1 values (computed once here for reuse)
    f1_values = []
    for c in range(n_classes):
        true_pos = np.sum((target[:, c] == 1) & (pred_labels[:, c] == 1))
        false_pos = np.sum((target[:, c] == 0) & (pred_labels[:, c] == 1))
        false_neg = np.sum((target[:, c] == 1) & (pred_labels[:, c] == 0))
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_values.append(f1)
    macro_f1 = np.mean(f1_values) * 100.0

    # Micro F1 Score calculation
    micro_tp = np.sum((target == 1) & (pred_labels == 1))
    micro_fp = np.sum((target == 0) & (pred_labels == 1))
    micro_fn = np.sum((target == 1) & (pred_labels == 0))
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0) * 100.0

    # Ranking Loss: percentage of mis-ranked label pairs per sample
    ranking_loss_value = 0.0
    for i in range(n_samples):
        relevant = np.where(target[i] == 1)[0]
        irrelevant = np.where(target[i] == 0)[0]
        if len(relevant) == 0 or len(irrelevant) == 0:
            continue
        wrong_pairs = 0
        for rel in relevant:
            for irr in irrelevant:
                if pred[i, rel] <= pred[i, irr]:
                    wrong_pairs += 1
        ranking_loss_value += wrong_pairs / (len(relevant) * len(irrelevant))
    ranking_loss_value = (ranking_loss_value / n_samples) * 100.0

    # Multilabel Coverage: average maximum rank among true labels per sample
    multilabel_coverage = 0.0
    valid_samples = 0
    for i in range(n_samples):
        relevant = np.where(target[i] == 1)[0]
        if len(relevant) == 0:
            continue
        ranks = np.zeros(n_classes)
        sorted_indices = np.argsort(-pred[i])
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        multilabel_coverage += np.max(ranks[relevant])
        valid_samples += 1
    multilabel_coverage = multilabel_coverage / valid_samples if valid_samples > 0 else 0.0

    # ---------------------------
    # Save all computed metrics to files
    # ---------------------------
    if save_dir is not None:
        # Per-class Accuracy
        class_accuracy = []
        for i in range(n_classes):
            acc = np.sum(pred_labels[:, i] == target[:, i]) / n_samples
            class_accuracy.append(acc)
        detailed_txt_path = os.path.join(save_dir, 'detailed_class_metrics.txt')
        with open(detailed_txt_path, 'w') as f:
            f.write("클래스별 AUC, F1 및 Accuracy 성능 추가:\n")
            for i in range(n_classes):
                f.write(f"  {disease_labels[i]}: AUC={auc_values[i]:.4f}, F1={f1_values[i]:.4f}, Accuracy={class_accuracy[i]:.4f}\n")
        
        # Overall metrics summary CSV
        import pandas as pd
        metrics_summary = {
            'Metric': ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'Multi AUC', 
                       'Hamming Loss', 'Ranking Loss', 'Multilabel Accuracy', 
                       'Multilabel Coverage', 'One Error', 'Subset Accuracy',
                       'Macro F1', 'Micro F1'],
            'Value': [CP, CR, CF1, OP, OR, OF1, multi_auc,
                      hamming_loss_value, ranking_loss_value, multilabel_accuracy,
                      multilabel_coverage, one_error, subset_accuracy, macro_f1, micro_f1]
        }
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
        
        # Per-class metrics CSV
        class_metrics = {
            'Disease': disease_labels,
            'AUC': auc_values,
            'F1': f1_values,
            'Precision': precision_class.tolist(),
            'Recall': recall_class.tolist()
        }
        class_df = pd.DataFrame(class_metrics)
        class_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False)
        
        # F1 Score bar plot
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(f1_values)
        sorted_diseases = [disease_labels[i] for i in sorted_indices]
        sorted_f1s = [f1_values[i] for i in sorted_indices]
        bars = plt.bar(sorted_diseases, sorted_f1s, color='lightgreen')
        plt.axhline(y=np.mean(f1_values), color='r', linestyle='-', label=f'Mean F1: {np.mean(f1_values):.3f}')
        plt.xlabel('Disease Classes')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores by Disease Class')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom')
        plt.savefig(os.path.join(save_dir, 'f1_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves, plots, and metric files are saved in {save_dir}")
    
    from collections import Counter

        # ---------------------------
    # 🔍 상위 20% 에러 샘플에 대한 클래스별 FP/FN 누적 막대 그래프
    # ---------------------------
    if save_dir is not None:
        print("Analyzing top 20% error samples for per-class error pattern visualization...")

        # 샘플 단위 에러 점수 계산 (Hamming Distance 기반)
        sample_errors = np.sum(pred_labels != target, axis=1)
        sorted_indices = np.argsort(sample_errors)[::-1]
        top_n = max(1, int(0.2 * len(sorted_indices)))
        high_error_indices = sorted_indices[:top_n]

        # 클래스별 FP/FN 계산
        false_positives = {cls: 0 for cls in disease_labels}
        false_negatives = {cls: 0 for cls in disease_labels}
        total_errors = {cls: 0 for cls in disease_labels}

        for idx in high_error_indices:
            for c, cls_name in enumerate(disease_labels):
                true_label = target[idx, c]
                pred_label = pred_labels[idx, c]

                if pred_label != true_label:
                    total_errors[cls_name] += 1
                    if true_label == 0 and pred_label == 1:
                        false_positives[cls_name] += 1
                    elif true_label == 1 and pred_label == 0:
                        false_negatives[cls_name] += 1

        # 시각화
        # 시각화 (색상: FP=파란색, FN=주황색)
        plt.figure(figsize=(15, 10))
        sorted_classes = sorted(total_errors.items(), key=lambda x: x[1], reverse=True)
        sorted_class_names = [item[0] for item in sorted_classes]
        fp_data = [false_positives[cls] for cls in sorted_class_names]
        fn_data = [false_negatives[cls] for cls in sorted_class_names]

        bar_width = 0.8
        plt.bar(sorted_class_names, fp_data, bar_width, label='False Positive', color='steelblue')
        plt.bar(sorted_class_names, fn_data, bar_width, bottom=fp_data, label='False Negative', color='darkorange')

        plt.xlabel('Disease Class')
        plt.ylabel('Number of Errors')
        plt.title('Error Patterns by Disease Class (Top 20% Error Samples)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # 저장
        error_pattern_path = os.path.join(save_dir, 'top20_error_class_pattern_colored.png')
        plt.savefig(error_pattern_path, dpi=300)
        plt.close()
        print(f"✅ Color-adjusted error pattern plot saved at {error_pattern_path}")

    return (CP, CR, CF1, OP, OR, OF1, multi_auc, hamming_loss_value, ranking_loss_value, 
            multilabel_accuracy, multilabel_coverage, one_error, subset_accuracy, macro_f1, micro_f1)


def add_multi_label_auc(pred, target, thr=None, k=None):
    """Calculate multi-label AUC."""
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or np.ndarray')
    
    total_auc = 0.0
    for i in range(target.shape[1]):
        try:
            auc_val = roc_auc_score(target[:, i], pred[:, i])
        except ValueError:
            auc_val = 0.5
        total_auc += auc_val
    multi_auc = total_auc / target.shape[1]
    return multi_auc
