import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import apply_full_preprocessing

# =====================================================
# CONFIGURATION FLAGS
# =====================================================
DEBUG_PRINT = True
TARGETS_FS = [32, 64]              # Target resampling frequency in Hz
# TARGETS_FS = [64]              # Target resampling frequency in Hz
# WINDOWS_SECONDS = [1, 3, 5, 10, 20, 30]         # Length of each window (seconds)
WINDOWS_SECONDS = [1]         # Length of each window (seconds)
OVERLAP = 0.0              # 0 = no overlap, 0.5 = 50% overlap
BALANCEAR = True           # Enable or disable class balancing
PROCESAR_PAPER = False    # Apply preprocessing as per the referenced paper

# Small hyperparameter search space for the CNN+BiLSTM
HYPERPARAM_CONFIGS = [
    {"cnn_channels": 8,  "lstm_hidden": 16, "lstm_layers": 1, "dropout": 0.5, "lr": 1e-3},
    {"cnn_channels": 16, "lstm_hidden": 32, "lstm_layers": 1, "dropout": 0.5, "lr": 1e-3},
    {"cnn_channels": 16, "lstm_hidden": 64, "lstm_layers": 2, "dropout": 0.5, "lr": 1e-3},
    {"cnn_channels": 8,  "lstm_hidden": 32, "lstm_layers": 2, "dropout": 0.3, "lr": 5e-4},
]

# =====================================================
# DATA LOADING AND PREPROCESSING
# =====================================================

def debug(msg):
    if DEBUG_PRINT:
        print(msg)

def moving_average(acc_data):
    avg = 0
    prevX, prevY, prevZ = 0, 0, 0
    results = []
    for i in range(0, len(acc_data), 32):
        sum_ = 0
        buffX = acc_data[i:i+32, 0]
        buffY = acc_data[i:i+32, 1]
        buffZ = acc_data[i:i+32, 2]
        for j in range(len(buffX)):
            sum_ += max(abs(buffX[j] - prevX), abs(buffY[j] - prevY), abs(buffZ[j] - prevZ))
            prevX, prevY, prevZ = buffX[j], buffY[j], buffZ[j]
        avg = avg * 0.9 + (sum_ / 32) * 0.1
        results.append(avg)
    return np.array(results)

def read_signals(main_folder):
    signal_dict = {}
    fs_dict = {}
    subfolders = next(os.walk(main_folder))[1]
    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        signals, fs_subject = {}, {}
        # desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'ACC.csv']
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'ACC.csv']
        for file_name in desired_files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                fs = int(df.iloc[0,0])
                df = df.drop(0)
                data = df.values.astype(float)
                if file_name == 'ACC.csv':
                    signals[file_name.split('.')[0]+'_x'] = data[:,0]
                    signals[file_name.split('.')[0]+'_y'] = data[:,1]
                    signals[file_name.split('.')[0]+'_z'] = data[:,2]
                    fs_subject[file_name.split('.')[0]+'_x'] = fs
                    fs_subject[file_name.split('.')[0]+'_y'] = fs
                    fs_subject[file_name.split('.')[0]+'_z'] = fs
                else:
                    signals[file_name.split('.')[0]] = data.flatten()
                    fs_subject[file_name.split('.')[0]] = fs
        signal_dict[folder_name] = signals
        fs_dict[folder_name] = fs_subject
    return signal_dict, fs_dict

def resample_to_target(signals, fs_dict, target_fs):
    resampled = {}
    min_len = np.inf
    for name, data in signals.items():
        fs = fs_dict[name]
        new_len = int(len(data) * target_fs / fs)
        data_resampled = resample(data, new_len)
        resampled[name] = data_resampled
        min_len = min(min_len, len(data_resampled))
    for k in resampled.keys():
        resampled[k] = resampled[k][:min_len]
    return resampled, min_len

def load_all_states(base_path, window_seconds, target_fs):
    # STATES = ['AEROBIC', 'ANAEROBIC', 'STRESS']
    STATES = ['AEROBIC', 'ANAEROBIC']
    # LABELS = {'AEROBIC': 0, 'ANAEROBIC': 1, 'STRESS': 2}
    LABELS = {'AEROBIC': 0, 'ANAEROBIC': 1}
    X_all, y_all = [], []
    valid_channels_old = []

    for state in STATES:
        debug(f"\n=== Loading {state} ===")
        signal_data, fs_data = read_signals(os.path.join(base_path, state))
        for subj, signals in signal_data.items():
            # if all(k in signals for k in ['EDA', 'BVP', 'HR', 'TEMP', 'ACC_x', 'ACC_y', 'ACC_z']):
            if all(k in signals for k in ['EDA', 'BVP', 'HR', 'ACC_x', 'ACC_y', 'ACC_z']):
                if PROCESAR_PAPER:
                    processed, fs_data[subj] = apply_full_preprocessing(signals, fs_data[subj])
                    resampled, min_len = resample_to_target(processed, fs_data[subj], target_fs)
                else:
                    resampled, min_len = resample_to_target(signals, fs_data[subj], target_fs)

                debug(f"[{subj}/{state}] " + ", ".join([f"{k}({fs_data[subj][k]}→{target_fs})" for k in resampled.keys()]))
                # Get all channels in sorted order (so ordering is consistent)
                channels = sorted(resampled.keys())

                # Filter only 1D signals with correct length
                valid_channels = [
                    ch for ch in channels
                    if isinstance(resampled[ch], np.ndarray) and resampled[ch].ndim == 1
                ]

                if not valid_channels_old:
                    valid_channels_old = valid_channels
                else:
                    if valid_channels != valid_channels_old:
                        raise ValueError("Inconsistent channels across subjects.")

                # Stack them automatically
                combined = np.vstack([resampled[ch] for ch in valid_channels])

                window_size = int(window_seconds * target_fs)
                step = int(window_size * (1 - OVERLAP))
                for start in range(0, combined.shape[1] - window_size, step):
                    X_all.append(combined[:, start:start + window_size])
                    y_all.append(LABELS[state])
                debug(f"  -> {len(range(0, combined.shape[1] - window_size, step))} windows created ({window_size} samples each)")
    debug(f"\n=== Dataset Summary ===")
    debug(f"Total windows: {len(y_all)}")
    return np.array(X_all), np.array(y_all), len(valid_channels), STATES

# =====================================================
# DOWNSAMPLING (BALANCING)
# =====================================================
def balance_classes(X, y):
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    indices_balanced = []
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        indices_balanced.extend(cls_idx[:min_count])
    np.random.shuffle(indices_balanced)
    return X[indices_balanced], y[indices_balanced]

# =====================================================
# CNN-BiLSTM MODEL
# =====================================================
class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels, num_classes=3, cnn_channels=8, lstm_hidden=16, lstm_layers=1, bidirectional=True, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * (2 if bidirectional else 1), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.permute(0, 2, 1)
        out, _ = self.bilstm(z)
        out = out[:, -1, :]
        logits = self.classifier(out)
        return logits

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def plot_all_metrics(modelname, train, val, test, epoch):
    """
    train, val, test: tuples (cm, acc, report)
    report should be a string from classification_report()
    """
    sets = [("Train", *train), ("Validation", *val), ("Test", *test)]

    fig = plt.figure(figsize=(12, 18))
    outer_gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    for i, (name, cm, acc, report) in enumerate(sets):
        gs = outer_gs[i].subgridspec(1, 2, width_ratios=[1.2, 1])

        # ---- Confusion matrix ----
        ax0 = fig.add_subplot(gs[0])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax0)
        ax0.set_title(f"{name} Confusion Matrix - Epoch {epoch}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("True")

        # ---- Metrics text ----
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        text = f"""
{name} Metrics
-------------------------
Accuracy: {acc:.3f}

Classification Report:
{report}
"""
        ax1.text(0, 1, text, fontsize=11, va="top", family="monospace")

    plt.tight_layout()
    plt.savefig(f'figs/{modelname}_cm_last epoch.png')
    plt.close()

# =====================================================
# TRAINING FUNCTION
# =====================================================
def train_model(modelname, train_loader, val_loader, test_loader, model, device, epochs=20, lr=1e-3, patience=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs, test_accs = [], [], []
    stats = []
    best_val_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch, preds, trues = [], [], []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
        train_loss = np.mean(train_loss_epoch)
        train_acc = accuracy_score(trues, preds)
        train_cm = confusion_matrix(trues, preds)
        train_report = classification_report(trues, preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)


        model.eval()
        val_loss_epoch, vpreds, vtrues = [], [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                logits = model(Xv)
                loss = criterion(logits, yv)
                val_loss_epoch.append(loss.item())
                vpreds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                vtrues.extend(yv.cpu().numpy())
        val_loss = np.mean(val_loss_epoch)
        val_losses.append(val_loss)
        val_acc = accuracy_score(vtrues, vpreds)
        val_cm = confusion_matrix(vtrues, vpreds)
        val_report = classification_report(vtrues, vpreds)
        val_accs.append(val_acc)
        val_prec = precision_score(vtrues, vpreds, average='macro')
        val_rec = recall_score(vtrues, vpreds, average='macro')
        val_f1 = f1_score(vtrues, vpreds, average='macro')

        # --- Test subset evaluation ---
        tpreds, ttrues = [], []
        with torch.no_grad():
            for Xt, yt in test_loader:
                Xt, yt = Xt.to(device), yt.to(device)
                logits = model(Xt)
                tpreds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                ttrues.extend(yt.cpu().numpy())
        test_acc = accuracy_score(ttrues, tpreds)
        test_cm = confusion_matrix(ttrues, tpreds)
        test_report = classification_report(ttrues, tpreds)
        test_prec = precision_score(ttrues, tpreds, average='macro')
        test_rec = recall_score(ttrues, tpreds, average='macro')
        test_f1 = f1_score(ttrues, tpreds, average='macro')
        test_accs.append(test_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
              f"Test Acc={test_acc:.4f}, Val Prec={val_prec:.4f}, Val Rec={val_rec:.4f}, "
              f"Val F1={val_f1:.4f}, Test F1={test_f1:.4f}")
        
        if epoch % 50 == 0 or epoch == epochs:
            plot_all_metrics(modelname,
                             (train_cm, train_acc, train_report),
                             (val_cm, val_acc, val_report),
                             (test_cm, test_acc, test_report),
                             epoch)
            
            # --- Plot Loss and Accuracy ---
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.legend(); plt.title('Loss Curve')

            plt.subplot(1,2,2)
            plt.plot(train_accs, label='Train Acc')
            plt.plot(val_accs, label='Val Acc')
            plt.plot(test_accs, label='Test Acc')
            plt.legend(); plt.title('Accuracy Curve')

            plt.tight_layout()
            plt.savefig(f'figs/{modelname}_training_curves.png')
            plt.close()

        scheduler.step(val_loss)
        stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'models/{modelname}_best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # --- Save metrics to CSV ---
    pd.DataFrame(stats).to_csv(f'csvs/{modelname}_training_stats.csv', index=False)

    # --- Plot Loss and Accuracy ---
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.legend(); plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(f'figs/{modelname}_training_curves.png')
    plt.close()


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    base_path = "PHYSIONET_Database/Wearable_Dataset"
    make_dirs = ['models', 'figs', 'csvs']
    for d in make_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    for target_fs in TARGETS_FS:
        for window_seconds in WINDOWS_SECONDS:
            print(f"\n=== Training with Target FS={target_fs} Hz and Window Size={window_seconds} s ===")
            X, y, in_channels, states = load_all_states(base_path, window_seconds, target_fs)

            if BALANCEAR:
                X, y = balance_classes(X, y)
                debug(f"After balancing: {np.unique(y, return_counts=True)}")

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=4096, shuffle=True)
            val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=4096, shuffle=False)
            test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=4096, shuffle=False)

            # model = CNNBiLSTM(in_channels=in_channels, num_classes=2).to(device)  # 7 channels: EDA, BVP, HR, TEMP, ACC_x, ACC_y, ACC_z
            # modelname = f"model_FS{target_fs}_WS{window_seconds}"
            # train_model(modelname, train_loader, val_loader, test_loader, model, device, epochs=2000, lr=1e-3)

            # ---- LOOP OVER HYPERPARAMETER CONFIGS ----
            for cfg_idx, cfg in enumerate(HYPERPARAM_CONFIGS):
                print(
                    f"\n--- Config {cfg_idx+1}/{len(HYPERPARAM_CONFIGS)} "
                    f"(FS={target_fs}, WS={window_seconds}) ---\n"
                    f"cnn_channels={cfg['cnn_channels']}, "
                    f"lstm_hidden={cfg['lstm_hidden']}, "
                    f"lstm_layers={cfg['lstm_layers']}, "
                    f"dropout={cfg['dropout']}, lr={cfg['lr']}"
                )

                model = CNNBiLSTM(
                    in_channels=in_channels,
                    num_classes=2,
                    cnn_channels=cfg["cnn_channels"],
                    lstm_hidden=cfg["lstm_hidden"],
                    lstm_layers=cfg["lstm_layers"],
                    dropout=cfg["dropout"],
                ).to(device)

                modelname = (
                    f"model_FS{target_fs}_WS{window_seconds}"
                    f"_cfg{cfg_idx}"
                    f"_cnn{cfg['cnn_channels']}"
                    f"_lstm{cfg['lstm_hidden']}"
                    f"_layers{cfg['lstm_layers']}"
                    f"_do{cfg['dropout']}"
                    f"_lr{cfg['lr']}"
                )

                # You can lower epochs here if this becomes too slow
                train_model(
                    modelname,
                    train_loader,
                    val_loader,
                    test_loader,
                    model,
                    device,
                    epochs=2000,
                    lr=cfg["lr"],
                )

    print("Training complete.")
