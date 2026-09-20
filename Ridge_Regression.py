import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import RidgeClassifier
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
parser.add_argument("--data_root", type=str, default="./LSM_OUT", help="Root folder containing MNIST, FSDD, NMNIST subfolders")
parser.add_argument("--exp_root", type=str, default="./Train_Out", help="Output root for csv files")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Target datasets
dataset_list = ["MNIST", "FSDD", "NMNIST"]

for dataset_name in dataset_list:
    data_dir = os.path.join(args.data_root, dataset_name)
    out_dir = os.path.join(args.exp_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary_metrics_ridge_regression.csv")
    csv_header = ["pt_file", "acc", "precision", "recall", "f1"]

    # Create csv header
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # Collect only pt files with "all" in filename
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found, skip {dataset_name}")
        continue
    pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt") and "_all" in f]
    pt_files.sort()

    for pt_name in tqdm(pt_files, desc=f"[{dataset_name}] Processing"):
        pt_path = os.path.join(data_dir, pt_name)
        data_pairs = torch.load(pt_path, weights_only=False)  # fix torch.load warning

        X_list = []
        y_list = []
        for d in data_pairs:
            spk = d["encoded_image"]
            lab = d["label"]
            # sum over time dimension, then flatten to 1D vector to guarantee 1D feature
            feat = spk.sum(dim=0).cpu().numpy()
            feat = feat.flatten()  # <==== 核心修复：展平，强制一维特征向量
            X_list.append(feat)
            y_list.append(int(lab))

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        # 80/20 train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        ridge_clf = RidgeClassifier(alpha=args.alpha, random_state=args.seed)
        ridge_clf.fit(X_train, y_train)
        y_pred = ridge_clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Append record
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([pt_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

        print(f"{dataset_name} | {pt_name} | Acc: {acc:.4f} | F1: {f1:.4f}")

print("All datasets finished. Only CSV metrics saved.")
