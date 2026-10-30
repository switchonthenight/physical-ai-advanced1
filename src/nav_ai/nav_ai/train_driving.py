"""
模倣学習（キー操作 + カメラ画像）用の学習スクリプト 改善版

元の train_model.py からの主な変更点:
  1. AdaptiveAvgPool2d((1,1)) を廃止し、空間情報（球が画面の左右どこにあるか）を残す
  2. 旋回を「左 / 直進 / 右」の3クラス分類にし、クラス重みで直進データの偏りを補正
     → MSE回帰で左右が平均されて「ほぼ直進」になり、赤玉に突っ込む問題への対策
  3. 人間の反応遅れを補正するラベルシフト（画像tに対して数フレーム先の操作を正解にする）
  4. 入力を 120x160 に縮小・正規化・明るさのデータ拡張・検証データ分割
  5. 画像番号が連続していない箇所（行の削除・別の走行）を区切りとして扱い、
     履歴フレームやラベルシフトが区切りをまたがないようにする
  6. 複数のデータセットフォルダをまとめて学習できる

CSV: image_name, linear_x, angular_z（data_collector.py の出力そのまま）
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------- 設定 ----------------
WS = os.path.expanduser("~/ros2_ws")
# 学習に使うデータセット（コマンドライン引数で上書き可: python3 train_driving.py dataset dataset.04）
DATASETS = sys.argv[1:] or ["dataset", "dataset.04"]
MODEL_PATH = os.path.join(WS, "driving_model_v2.pth")

HISTORY_LEN = 3          # 過去何フレームを入力に重ねるか
LABEL_SHIFT = 2          # 画像tの正解に使う操作のフレームずれ（10Hz収集なら2=0.2秒先）
ANG_THRESH = 0.1         # |angular_z| がこれ未満なら「直進」クラス
IMG_H, IMG_W = 120, 160  # 入力解像度
VAL_RATIO = 0.15         # 各データセットの末尾15%を検証用（時系列で分割＝リークしにくい）
EPOCHS = 30
LR = 3e-4
BATCH = 64
NUM_WORKERS = 4
CLASS_NAMES = ["left", "straight", "right"]   # 0: 左(angular>0), 1: 直進, 2: 右(angular<0)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def angular_to_class(a):
    if a > ANG_THRESH:
        return 0
    if a < -ANG_THRESH:
        return 2
    return 1


def load_datasets(names):
    """複数フォルダの CSV を結合し、連続区間ごとの segment 番号を付ける"""
    frames, val_mask = [], []
    seg_offset = 0
    for name in names:
        d = os.path.join(WS, name)
        df = pd.read_csv(os.path.join(d, "data.csv"))
        df.columns = ["image_name", "linear_x", "angular_z"]
        n_raw = len(df)
        # 停止・後退の行は除外（前進中の操舵だけを学習する）
        df = df[df["linear_x"] > 0].reset_index(drop=True)
        num = df["image_name"].str.extract(r"(\d+)")[0].astype(int)
        # 画像番号が +1 でつながっていない所で区切る（行の削除や別の走行）
        seg = (num.diff() != 1).cumsum() + seg_offset
        seg_offset = int(seg.max()) + 1
        df["path"] = [os.path.join(d, "images", f) for f in df["image_name"]]
        df["seg"] = seg.values
        n_val = int(len(df) * VAL_RATIO)
        val_mask.append(np.r_[np.zeros(len(df) - n_val, bool), np.ones(n_val, bool)])
        frames.append(df)
        print(f"  {name}: {n_raw}行 → 前進のみ {len(df)}行, 連続区間 {df['seg'].nunique()}")
    df = pd.concat(frames, ignore_index=True)
    return df, np.concatenate(val_mask)


class DrivingDataset(Dataset):
    def __init__(self, df, indices, train=True):
        self.paths = df["path"].values
        self.lin = df["linear_x"].values.astype(np.float32)
        self.ang = df["angular_z"].values.astype(np.float32)
        self.seg = df["seg"].values
        self.indices = indices
        self.base = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor()])
        # 色相は変えない（球の色が手がかりなので）。明るさ・コントラストのみ揺らす
        self.jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2) if train else None
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        s = self.seg[idx]
        imgs = []
        for h in range(HISTORY_LEN):
            j = idx - (HISTORY_LEN - 1 - h)
            # 先頭や区間の境界をまたぐ場合は、区間内の最古フレームで埋める
            while j < 0 or self.seg[j] != s:
                j += 1
            imgs.append(Image.open(self.paths[j]).convert("RGB"))

        # 同じ拡張パラメータを全履歴フレームに適用
        if self.jitter is not None:
            seed = torch.randint(0, 2**31, (1,)).item()
            out = []
            for im in imgs:
                torch.manual_seed(seed)
                out.append(self.jitter(im))
            imgs = out
        x = torch.cat([self.norm(self.base(im)) for im in imgs], dim=0)  # [3*H, IMG_H, IMG_W]

        # ラベルシフト（同じ区間内でのみ。区間末尾ではその区間の最後の操作）
        t = idx
        for _ in range(LABEL_SHIFT):
            if t + 1 < len(self.seg) and self.seg[t + 1] == s:
                t += 1
        return x, torch.tensor(angular_to_class(self.ang[t])), torch.tensor([self.lin[t]])


class DrivingCNN(nn.Module):
    """PilotNet系。最後に空間方向をつぶし切らない（左右の位置情報を保持）"""
    def __init__(self, history_len=HISTORY_LEN, n_cls=3):
        super().__init__()
        c = 3 * history_len
        self.features = nn.Sequential(
            nn.Conv2d(c, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2, padding=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2, padding=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),   # ★(1,1)ではなく横5分割を残す
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 3 * 5, 128), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, 64), nn.ReLU())
        self.head_cls = nn.Linear(64, n_cls)   # 旋回クラス
        self.head_lin = nn.Linear(64, 1)       # 前進速度

    def forward(self, x):
        h = self.fc(self.features(x))
        return self.head_cls(h), self.head_lin(h)


def main():
    print("データセット:", DATASETS)
    df, is_val = load_datasets(DATASETS)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    # 正解はラベルシフト後の値なので、Dataset 経由でクラスを数える
    probe = DrivingDataset(df, np.arange(len(df)), train=False)
    shifted_cls = np.empty(len(df), int)
    for i in range(len(df)):
        t, s = i, probe.seg[i]
        for _ in range(LABEL_SHIFT):
            if t + 1 < len(df) and probe.seg[t + 1] == s:
                t += 1
        shifted_cls[i] = angular_to_class(probe.ang[t])
    counts = np.bincount(shifted_cls[train_idx], minlength=3)
    print(f"学習 {len(train_idx)}件 / 検証 {len(val_idx)}件")
    print("クラス分布(train):", dict(zip(CLASS_NAMES, counts.tolist())))
    if counts.min() == 0:
        print("⚠ 一方向の旋回データがありません。ANG_THRESH かデータを見直してください")
    class_w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)

    train_dl = DataLoader(DrivingDataset(df, train_idx, train=True), batch_size=BATCH,
                          shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    val_dl = DataLoader(DrivingDataset(df, val_idx, train=False), batch_size=BATCH,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model = DrivingCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    class_w = class_w.to(device)

    best, best_conf = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, y, lin in train_dl:
            x, y, lin = x.to(device, non_blocking=True), y.to(device), lin.to(device)
            logit, lin_pred = model(x)
            loss = F.cross_entropy(logit, y, weight=class_w) + 0.5 * F.mse_loss(lin_pred, lin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        # 検証: クラスごとの再現率（「右に切るべき場面で右と言えたか」）を見る
        model.eval()
        conf = np.zeros((3, 3), dtype=int)
        with torch.no_grad():
            for x, y, _ in val_dl:
                p = model(x.to(device))[0].argmax(1).cpu().numpy()
                for t_, p_ in zip(y.numpy(), p):
                    conf[t_, p_] += 1
        recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
        bal_acc = recall[conf.sum(1) > 0].mean()
        print(f"epoch {ep+1:02d} loss {tot/len(train_idx):.4f}  val recall "
              + " ".join(f"{c}={r:.2f}" for c, r in zip(CLASS_NAMES, recall))
              + f"  bal_acc={bal_acc:.3f}", flush=True)
        if bal_acc > best:
            best, best_conf = bal_acc, conf.copy()
            torch.save(model.state_dict(), MODEL_PATH)

    with open(os.path.splitext(MODEL_PATH)[0] + "_config.json", "w") as f:
        json.dump({"history_len": HISTORY_LEN, "img_h": IMG_H, "img_w": IMG_W,
                   "mean": MEAN, "std": STD, "classes": CLASS_NAMES,
                   "datasets": DATASETS, "label_shift": LABEL_SHIFT,
                   "class_weight": class_w.cpu().tolist()}, f, indent=2)
    print(f"完了: best bal_acc={best:.3f} → {MODEL_PATH}")
    print("ベスト時の混同行列(行=正解 left/straight/right, 列=予測):\n", best_conf)


if __name__ == "__main__":
    main()
