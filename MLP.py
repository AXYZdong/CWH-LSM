import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--n_neurons", type=int, default=900)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")

parser.add_argument("--cuda_name", type=str, default='cuda:0')
parser.add_argument("--root_dir", type=str, default="./LSM_OUT", help="Root folder containing MNIST FSDD NMNIST subfolders")
parser.add_argument("--confusion_matrix", dest="confusion_matrix", action="store_true")
parser.add_argument("--detailed_report", dest="detailed_report", action="store_true",
                    help="Print detailed classification report")
parser.set_defaults(plot=False, gpu=True, confusion_matrix=False, detailed_report=False)

args = parser.parse_args()

# Fixed random seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Device setup
device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    device = torch.device("cpu")
print("Running on Device = ", device)

# Number of workers
if args.n_workers == -1:
    n_workers = 0
else:
    n_workers = args.n_workers
torch.set_num_threads(max(1, os.cpu_count() - 1))

# MLP Readout Model
class NN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_p=0.8):
        super(NN, self).__init__()
        h = int(input_size / 2)
        self.linear_1 = nn.Linear(input_size, h)
        self.bn1 = nn.BatchNorm1d(h)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        x = x.float()
        out = torch.relu(self.linear_1(x))
        out = self.dropout(out)
        out = self.linear_2(out)
        return out


def run_single_pt(pt_path, dataset_name, out_root):
    """
    Train & evaluate MLP readout on one _all.pt LSM spike dataset
    :param pt_path: path to xxx_all.pt
    :param dataset_name: MNIST / FSDD / NMNIST
    :param out_root: Train_MLP_Out/{dataset_name}
    :return: result dict with best metrics
    """
    exp_tag = os.path.basename(pt_path)
    print(f"\n===== Start processing {exp_tag} (dataset: {dataset_name}) =====")

    # Create output subdir for this dataset
    save_dir = os.path.join(out_root, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load spike dataset
    dataset = torch.load(pt_path, map_location=device, weights_only=False)
    total_length = len(dataset)
    train_length = int(total_length * 0.8)
    test_length = total_length - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, test_length])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers)
    test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)

    # Initialize model, loss, optimizer
    learning_model = NN(args.n_neurons, 10).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(
        learning_model.parameters(),
        lr=2e-4,
    )

    best_confusion_matrix = None
    best_val = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_epoch = 0
    train_val_data = []

    print("Training the readout MLP")
    for epoch in tqdm(range(args.n_epochs)):
        learning_model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        # Train
        for batch in train_dataloader:
            encoded_image = batch["encoded_image"].sum(1).squeeze(1).to(device)
            l = batch["label"].to(device).long().flatten()

            optimizer.zero_grad()
            outputs = learning_model(encoded_image)
            loss = criterion(outputs, l)
            epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == l).sum().item()
            total_train += l.size(0)

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        learning_model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                encoded_image = batch["encoded_image"].sum(1).squeeze(1).to(device)
                l = batch["label"].to(device).long().flatten()

                outputs = learning_model(encoded_image)
                loss = criterion(outputs, l)
                epoch_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == l).sum().item()
                total_val += l.size(0)

                all_labels.extend(l.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct_val / total_val
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        if args.detailed_report:
            print(classification_report(all_labels, all_predictions, zero_division=0))

        avg_val_loss = epoch_val_loss / len(test_dataloader)
        train_val_data.append([avg_train_loss, train_accuracy, avg_val_loss, val_accuracy])

        # Update best record
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_epoch = epoch
            if args.confusion_matrix:
                best_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    # Save confusion matrix plot if required
    if args.confusion_matrix and best_confusion_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(best_confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix | {exp_tag} | Best Acc:{best_val:.2f}%")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(save_dir, f"{exp_tag}_cm.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Save train/val loss & acc curve
    if args.plot:
        arr = np.array(train_val_data)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(arr[:,0], label="Train Loss")
        plt.plot(arr[:,2], label="Val Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(arr[:,1], label="Train Acc")
        plt.plot(arr[:,3], label="Val Acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{exp_tag}_curve.png"), dpi=300)
        plt.close()

    return {
        "pt_file": exp_tag,
        "best_acc": best_val,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "best_epoch": best_epoch
    }


if __name__ == "__main__":
    target_datasets = ["MNIST", "FSDD", "NMNIST"]
    out_root = "Train_MLP_Out"

    for ds_name in target_datasets:
        ds_folder = os.path.join(args.root_dir, ds_name)
        if not os.path.exists(ds_folder):
            print(f"Warning: folder {ds_folder} not found, skip.")
            continue

        # Collect only files ending with _all.pt
        all_pt_files = []
        for fname in os.listdir(ds_folder):
            if fname.endswith("_all.pt"):
                all_pt_files.append(os.path.join(ds_folder, fname))
        all_pt_files.sort()

        if len(all_pt_files) == 0:
            print(f"No *_all.pt found under {ds_folder}, skip")
            continue

        csv_path = os.path.join(out_root, ds_name, f"{ds_name}_mlp_summary_{args.seed}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Write csv header
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pt_file", "best_acc", "best_precision", "best_recall", "best_f1", "best_epoch"])

        # Run each pt file and append result to csv
        for pt_path in all_pt_files:
            res = run_single_pt(pt_path, ds_name, out_root)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    res["pt_file"],
                    f"{res['best_acc']:.4f}",
                    f"{res['best_precision']:.4f}",
                    f"{res['best_recall']:.4f}",
                    f"{res['best_f1']:.4f}",
                    res["best_epoch"]
                ])
        print(f"\nDataset {ds_name} finished, summary saved to {csv_path}")
