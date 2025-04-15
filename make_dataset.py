import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 기본 경로 설정
base_dir = "/userHome/userhome4/kyoungmin/code/Xray/dataset"
csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")

# 하위 폴더(예: images_001, images_002, ...)에서 이미지 경로 찾기
img_mapping = {}
for subfolder in os.listdir(base_dir):
    if subfolder.startswith("images_"):
        # 예: /userHome/userhome4/kyoungmin/code/Xray/dataset/images_001/images
        images_path = os.path.join(base_dir, subfolder, "images")
        if os.path.isdir(images_path):
            for root, dirs, files in os.walk(images_path):
                for file in files:
                    # file 예: "00009985_000.png"
                    # 상대 경로: 예: images_001/images/00009985_000.png
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    img_mapping[file] = rel_path

# 클래스 리스트 정의
chest14_classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# CSV 파일 불러오기 및 전처리
df = pd.read_csv(csv_path)
df = df[["Image Index", "Finding Labels"]]

for c in chest14_classes:
    df[c] = df["Finding Labels"].apply(lambda x: int(c in x.split('|')))
df["labels"] = df[chest14_classes].values.tolist()

# 이미지 파일 경로를 img_mapping을 이용해 찾기
# 만약 해당 파일이 매핑에 없으면, 경고를 출력하고 파일명만 사용함.
def get_img_path(img_name):
    if img_name in img_mapping:
        return img_mapping[img_name]
    else:
        print(f"Warning: {img_name} not found in mapping. Using default path.")
        return "images/" + img_name

df["line"] = df["Image Index"].apply(lambda x: get_img_path(x))
# 라벨은 쉼표 구분하여 저장 (예: 0,0,1,0,...)
df["line"] = df.apply(lambda r: r["line"] + "\t" + ",".join(map(str, r["labels"])), axis=1)

# 데이터 분할
df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_trainval, test_size=0.111, random_state=42)

# 파일에 직접 저장 (to_csv 대신 open() 사용하여 쌍따옴표 문제가 발생하지 않도록)
with open(os.path.join(base_dir, "train_val_list.txt"), "w") as f:
    for line in df_trainval["line"]:
        f.write(line + "\n")

with open(os.path.join(base_dir, "add72_chest14_val_labels.txt"), "w") as f:
    for line in df_val["line"]:
        f.write(line + "\n")

with open(os.path.join(base_dir, "add72_chest14_test_labels.txt"), "w") as f:
    for line in df_test["line"]:
        f.write(line + "\n")

with open(os.path.join(base_dir, "add72_chest14_classes.txt"), "w") as f:
    for c in chest14_classes:
        f.write(c + "\n")
