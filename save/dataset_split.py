import os
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# -------------------------
# 기본 설정
# -------------------------
SEED = 42
np.random.seed(SEED)

# 경로 설정
meta_file     = '/userHome/userhome4/kyoungmin/code/Xray/dataset/Data_Entry_2017.csv'
classes_file  = '/userHome/userhome4/kyoungmin/code/Xray/dataset/add72_chest14_classes.txt'
save_base_dir = '/userHome/userhome4/kyoungmin/code/Xray/CTransCNN/save'
images_subdir = 'images'  # 파일명 앞에 붙일 접두사

# -------------------------
# 메타데이터 로드 및 전처리
# -------------------------
df = pd.read_csv(meta_file)
df.columns = [c.strip() for c in df.columns]

# Finding_Labels 컬럼 확인
if 'Finding_Labels' in df.columns:
    finding_col = 'Finding_Labels'
elif 'Finding Labels' in df.columns:
    finding_col = 'Finding Labels'
else:
    raise ValueError("Finding_Labels 컬럼을 찾을 수 없습니다.")

# 클래스 순서 로드
with open(classes_file, 'r') as f:
    disease_labels = [line.strip() for line in f if line.strip()]

# One‑Hot 인코딩 컬럼 생성
for d in disease_labels:
    df[d] = df[finding_col].apply(lambda x: 1 if d in str(x) else 0)

# -------------------------
# 성능 최적화: One‑Hot 벡터 미리 매핑
# -------------------------
# Image Index를 인덱스로 설정한 후 각 행을 문자열로 결합
df_labels = df.set_index('Image Index')[disease_labels]
label_map = {
    img: ','.join(map(str, df_labels.loc[img].astype(int).tolist()))
    for img in df_labels.index
}

# -------------------------
# 10-Fold 분할 설정
# -------------------------
if 'Image Index' not in df.columns:
    raise ValueError("Image Index 컬럼을 찾을 수 없습니다.")

Y = df[disease_labels].values
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

splits = []
for _, (tr_idx, te_idx) in enumerate(mskf.split(df, Y)):
    train_list = df.iloc[tr_idx]['Image Index'].tolist()
    test_list  = df.iloc[te_idx]['Image Index'].tolist()
    splits.append((train_list, test_list))

# -------------------------
# 분할 결과 저장 (001~010)
# -------------------------
for i, (train_list, test_list) in enumerate(splits, start=1):
    exp_dir = os.path.join(save_base_dir, f"{i:03d}")
    os.makedirs(exp_dir, exist_ok=True)

    # train_val_list.txt 생성
    tv_path = os.path.join(exp_dir, 'train_val_list.txt')
    with open(tv_path, 'w') as fw:
        for img in train_list:
            fw.write(f"{images_subdir}/{img}\t{label_map[img]}\n")

    # test_list.txt 생성
    tst_path = os.path.join(exp_dir, 'test_list.txt')
    with open(tst_path, 'w') as fw:
        for img in test_list:
            fw.write(f"{images_subdir}/{img}\t{label_map[img]}\n")

    print(f"[{i:03d}] train_val saved to {tv_path} ({len(train_list)} samples)")
    print(f"[{i:03d}] test      saved to {tst_path} ({len(test_list)} samples)")
