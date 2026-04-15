import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import gc
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_curve
)
import copy
from pipeline.config import (
    ADAM_LR,
    ADAM_WEIGHT_DECAY,
    COMBINED_CSV_PATH,
    COMBINED_NPY_PATH,
    FTRL_ALPHA,
    FTRL_BETA,
    FTRL_L1,
    FTRL_L2,
    TRAIN_BATCH_SIZE,
    TRAIN_D_MODEL,
    TRAIN_EPOCHS,
    TRAIN_NUM_WORKERS,
    TRAIN_PATIENCE,
    TRAIN_PERSISTENT_WORKERS,
    TRAIN_PIN_MEMORY,
    TRAIN_PREFETCH_FACTOR,
    TRAIN_SIZE,
    VAL_SIZE,
)
from model.model import FTRL, xDeepFM


class RecDataset(Dataset):
    def __init__(self, df, clip_memmap):
        self.user_ids   = df['userId'].values.astype(np.int32)
        self.tweet_ids  = df['tweetId'].values.astype(np.int32)
        self.npy_index  = df['npy_index'].values.astype(np.int64)
        self.labels     = df['target'].values.astype(np.uint8)
        self.clip_memmap = clip_memmap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_id   = torch.tensor(self.user_ids[idx],   dtype=torch.long)
        tweet_id  = torch.tensor(self.tweet_ids[idx],  dtype=torch.long)
        clip_tensor = torch.tensor(self.clip_memmap[self.npy_index[idx]], dtype=torch.float32)
        label     = torch.tensor(self.labels[idx],     dtype=torch.float32)

        return user_id, tweet_id, clip_tensor, label


def train_epoch(model, dataloader, criterion, optimizer_adam, optimizer_ftrl, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for user_id, tweet_id, clip_tensor, labels in pbar:
        user_id, tweet_id = user_id.to(device), tweet_id.to(device)
        clip_tensor = clip_tensor.to(device)
        labels = labels.to(device)
        
        optimizer_adam.zero_grad()
        optimizer_ftrl.zero_grad()
        
        logits = model(user_id, tweet_id, clip_tensor).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        
        optimizer_ftrl.step()
        optimizer_adam.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for user_id, tweet_id, clip_tensor, labels in pbar:
        user_id, tweet_id = user_id.to(device), tweet_id.to(device)
        clip_tensor = clip_tensor.to(device)
        labels = labels.to(device)
        
        logits = model(user_id, tweet_id, clip_tensor).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(probs.cpu().numpy())
        
    return total_loss / len(dataloader), np.array(all_targets), np.array(all_preds)


def calc_and_plot_metrics(targets, preds, history, title_suffix=""):
    pred_labels = (preds >= 0.5).astype(int)
    
    auc_roc = roc_auc_score(targets, preds)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, preds)
    pr_auc = auc(recall_vals, precision_vals)
    map_score = average_precision_score(targets, preds)
    
    prec = precision_score(targets, pred_labels, zero_division=0)
    rec = recall_score(targets, pred_labels, zero_division=0)
    f1 = f1_score(targets, pred_labels, zero_division=0) 
    cm = confusion_matrix(targets, pred_labels)

    print(f"""
        AUC-ROC: {auc_roc}
        Precision: {prec}
        Recall: {rec}
        F1-Score: {f1}
        PR-AUC: {pr_auc}
        mAP: {map_score}
    """)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    axs[0, 0].plot(history["train_loss"], label="Train Loss", marker='o')
    axs[0, 0].plot(history["val_loss"], label="Val Loss", marker='o')
    if "test_loss" in history and len(history["test_loss"]) > 0:
        axs[0, 0].plot(len(history["val_loss"]), history["test_loss"][0], label="Test Loss", marker='*', markersize=12)
    axs[0, 0].set_title(f"Loss Over Epochs {title_suffix}")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    fpr, tpr, _ = roc_curve(targets, preds)
    axs[0, 1].plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}", color='darkorange')
    axs[0, 1].plot([0, 1], [0, 1], 'k--')
    axs[0, 1].set_title(f"ROC Curve {title_suffix}")
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.3f}", color='purple')
    axs[1, 0].set_title(f"Precision-Recall Curve {title_suffix}")
    axs[1, 0].set_xlabel("Recall")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].legend(loc="lower left")
    axs[1, 0].grid(True)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1, 1], cbar=False)
    axs[1, 1].set_title(f"Confusion Matrix {title_suffix}")
    axs[1, 1].set_xlabel("Predicted Label")
    axs[1, 1].set_ylabel("True Label")
    
    plt.tight_layout()
    plt.savefig(f"metrics_plot_{title_suffix.strip().replace(' ', '_')}.png")
    plt.close(fig)


def train(model, pos_weight, train_loader, val_loader, test_loader, device, epochs, patience):
    pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)

    linear_params = []
    deep_params = []

    for name, param in model.named_parameters():
        if 'linear' in name:
            linear_params.append(param)
        else:
            deep_params.append(param)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_ftrl = FTRL(linear_params, alpha=FTRL_ALPHA, beta=FTRL_BETA, l1=FTRL_L1, l2=FTRL_L2)
    optimizer_adam = torch.optim.Adam(deep_params, lr=ADAM_LR, weight_decay=ADAM_WEIGHT_DECAY)
    
    history = {"train_loss": [], "val_loss": [], "test_loss": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_targets, best_preds = None, None

    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_adam, optimizer_ftrl, device)
        val_loss, val_targets, val_preds = evaluate_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_targets = val_targets
            best_preds = val_preds
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_xdeepfm.pth")
            print("  --> Saved new best model!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        if best_targets is not None and best_preds is not None:
            calc_and_plot_metrics(best_targets, best_preds, history, title_suffix="Validation Set")

    print("\n--- Validation Metrics on Best Epoch ---")
    calc_and_plot_metrics(best_targets, best_preds, history, title_suffix="Validation Set")

    model.load_state_dict(best_model_wts)
    
    print("\n--- Evaluating best model on Test Set ---")
    final_test_loss, final_test_targets, final_test_preds = evaluate_epoch(model, test_loader, criterion, device)
    history["test_loss"].append(final_test_loss)
    calc_and_plot_metrics(final_test_targets, final_test_preds, history, title_suffix="Test Set")
    
    return model


def run_training_pipeline():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Normalizing Dataset Headers...")
    temp_df = pd.read_csv(COMBINED_CSV_PATH, nrows=0)
    col_map = {}
    for col in temp_df.columns:
        if col in ["user_id", "userId"]: col_map[col] = "userId"
        if col in ["tweet_id", "tweetId"]: col_map[col] = "tweetId"
        if col in ["label", "target"]: col_map[col] = "target"
        if col == "timestamp": col_map[col] = col
        if col == "npy_index": col_map[col] = col

    print("Loading CSV into RAM (Downcasted)...")
    df = pd.read_csv(
        COMBINED_CSV_PATH,
        usecols=col_map.keys()
    ).rename(columns=col_map)

    if "npy_index" not in df.columns:
        df["npy_index"] = np.arange(len(df), dtype=np.int64)
    else:
        df["npy_index"] = df["npy_index"].astype(np.int64)

    for col in ["userId", "tweetId"]:
        if col in df.columns: df[col] = df[col].astype(np.int32)
    if "target" in df.columns: df["target"] = df["target"].astype(np.uint8)

    print("Calculating vocabulary sizes...")
    TOTAL_USERS = int(df['userId'].max()) + 1
    TOTAL_TWEETS = int(df['tweetId'].max()) + 1

    print("Sorting data chronologically...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Splitting data (70% Train / 15% Val / 15% Test)...")
    train_end = int(len(df) * TRAIN_SIZE)
    val_end = int(len(df) * (VAL_SIZE + TRAIN_SIZE))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    del df, temp_df
    gc.collect()

    pos_weight = (train_df["target"] == 0).sum() / (train_df["target"] == 1).sum()
    print(f"Dynamic pos_weight calculated as: {pos_weight:.4f}")

    if not os.path.exists(COMBINED_NPY_PATH):
        raise FileNotFoundError(f"Missing embeddings file: {COMBINED_NPY_PATH}")

    clip_memmap = np.load(COMBINED_NPY_PATH, mmap_mode="r")
    clip_dim = int(clip_memmap.shape[1])

    print("Creating DataLoaders...")
    train_dataset = RecDataset(train_df, clip_memmap)
    val_dataset = RecDataset(val_df, clip_memmap)
    test_dataset = RecDataset(test_df, clip_memmap)

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=TRAIN_NUM_WORKERS, pin_memory=TRAIN_PIN_MEMORY,
        prefetch_factor=TRAIN_PREFETCH_FACTOR, persistent_workers=TRAIN_PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False,
        num_workers=TRAIN_NUM_WORKERS, pin_memory=TRAIN_PIN_MEMORY,
        prefetch_factor=TRAIN_PREFETCH_FACTOR, persistent_workers=TRAIN_PERSISTENT_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False,
        num_workers=TRAIN_NUM_WORKERS, pin_memory=TRAIN_PIN_MEMORY,
        prefetch_factor=TRAIN_PREFETCH_FACTOR, persistent_workers=TRAIN_PERSISTENT_WORKERS
    )

    print("Initializing model...")
    model = xDeepFM(
        device=DEVICE,
        d_model=TRAIN_D_MODEL,
        num_users=TOTAL_USERS,
        num_tweets=TOTAL_TWEETS,
        clip_dim=clip_dim
    ).to(DEVICE)

    print("Started Training...")
    train(
        model=model,
        pos_weight=pos_weight,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=TRAIN_EPOCHS,
        patience=TRAIN_PATIENCE
    )

    print("Training Completed.")

    

if __name__ == "__main__":
    run_training_pipeline()