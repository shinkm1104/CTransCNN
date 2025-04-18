#!/bin/bash

# 반복할 Fold 번호 설정
FOLD=002

CUDA_VISIBLE_DEVICES=1 python train.py \
  --config /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/configs/NIH_ChestX-ray14_CTransCNN.py \
  --work-dir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/${FOLD} \
  --cfg-options \
    data.train.ann_file="/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/${FOLD}/train_val_list.txt" \
    data.val.ann_file="/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/${FOLD}/train_val_list.txt" \
    data.test.ann_file="/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/${FOLD}/test_list.txt" \
  --resume-from /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/${FOLD}/epoch_2.pth \
  --seed 42

#     --resume-from /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/epoch_47.pth \
#     --seed 42 --deterministic --gpu-ids 1 2

# CUDA_VISIBLE_DEVICES=2 python test.py \
#     /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/configs/NIH_ChestX-ray14_CTransCNN.py \
#     /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/epoch_50.pth\
#     --out /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/result_skm_epoch50.json \
#     --out-items all \
#     --metrics all\
#     --gpu-collect \
#     --tmpdir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/tmp\
#     --gpu-id 0\
#     --device cuda \
    # --show \
    # --show-dir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save/visualizations
    # --metrics mAP CP CR CF1 OP OR OF1 multi_auc hamming_loss ranking_loss multilabel_accuracy \
    # one_error subset_accuracy macro_f1 micro_f1 multilabel_coverage \
    # --show \
    # --show-dir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save
    # --metrics accuracy precision recall f1_score support\


# python -m torch.distributed.launch --nproc_per_node=3 train.py \
#     --config /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/configs/NIH_ChestX-ray14_CTransCNN.py \
#     --work-dir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save \
#     --seed 42 \
#     --deterministic \
#     --launcher pytorch

# CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.run --nproc_per_node=3 train.py \
#     --config /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/configs/NIH_ChestX-ray14_CTransCNN.py \
#     --work-dir /userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save \
#     --seed 42 --deterministic --launcher pytorch