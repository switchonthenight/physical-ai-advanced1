"""
簡易 ACT (Action Chunking with Transformers) の学習スクリプト

構成（元論文 ACT を本課題向けに簡略化）:
  画像 → ResNet18(ImageNet事前学習, layer4まで) → 1x1conv で d_model 次元のトークン列（2D正弦波位置埋め込み）
  [潜在zトークン + 画像トークン] → Transformer エンコーダ
  CHUNK 個の学習クエリ → Transformer デコーダ → 各ステップ「左/直進/右」3クラス + 前進速度
  CVAE（ACT_CVAE=1 のとき）: 学習時は行動チャンクから z を推定、推論時は z=0（平均）

ラベル:
  画像 t に対し、行動 t+LABEL_SHIFT ... t+LABEL_SHIFT+CHUNK-1 を予測（CNN と同じ反応遅れ補正）
  エピソード末尾を超える部分・前進していない行動は損失から除外（mask）

データ:
  新形式 dataset_act/data.csv : episode, frame, image_name, linear_x, angular_z, sim_time
  旧形式 dataset*/data.csv     : image_name, linear_x, angular_z（画像番号の連続区間を1エピソードとみなす）
  検証は「最後の VAL_RATIO のエピソード」（エピソード単位で分割＝リークしない）

使い方:
  python3 train_act.py                       # dataset_act で学習 → ~/ros2_ws/act_model.pth
  python3 train_act.py dataset_act dataset   # 複数データセット
  環境変数: ACT_MODEL(保存名) ACT_CVAE(0/1) ACT_EPOCHS ACT_MAX_STEPS(動作確認用)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------- 設定 ----------------
WS = os.path.expanduser("~/ros2_ws")
DATASETS = sys.argv[1:] or ["dataset_act"]
MODEL_PATH = os.path.join(WS, os.environ.get("ACT_MODEL", "act_model.pth"))

CHUNK = 10               # 予測する行動の個数（10Hz なら1秒分）
LABEL_SHIFT = 2          # 画像tの最初の正解を何フレーム先の操作にするか（CNNと同じ）
ANG_THRESH = 0.1
IMG_H, IMG_W = 192, 256  # ResNet18 の出力は 6x8=48 トークン
D_MODEL = 256
N_HEAD = 8
N_ENC, N_DEC = 2, 2      # 元論文は 4/7 層。データが少ないので小さめ
DROPOUT = 0.1
USE_CVAE = os.environ.get("ACT_CVAE", "0") == "1"
LATENT = 32
KL_W = 10.0
LIN_W = 0.5
VAL_RATIO = 0.15
EPOCHS = int(os.environ.get("ACT_EPOCHS", "40"))
MAX_STEPS = int(os.environ.get("ACT_MAX_STEPS", "0"))
USE_AUG = os.environ.get("ACT_AUG", "0") == "1"      # 1: 平行移動・拡大縮小の幾何拡張（過学習対策）
FREEZE = os.environ.get("ACT_FREEZE", "")            # 例 layer2: conv1〜layer2 を凍結（過学習対策）   # >0 なら各epochをこのステップ数で打ち切る（動作確認用）
LR, LR_BACKBONE = 1e-4, 1e-5
BATCH = 32
NUM_WORKERS = 4
CLASS_NAMES = ["left", "straight", "right"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def angular_to_class(a):
    if a > ANG_THRESH:
        return 0
    if a < -ANG_THRESH:
        return 2
    return 1


# ---------------- データ ----------------
def load_episodes(names):
    eps = []
    for name in names:
        d = os.path.join(WS, name)
        df = pd.read_csv(os.path.join(d, "data.csv"))
        if "episode" in df.columns:
            groups = [g.sort_values("frame") for _, g in df.groupby("episode")]
            kind = "エピソード"
        else:
            df.columns = ["image_name", "linear_x", "angular_z"]
            num = df["image_name"].str.extract(r"(\d+)")[0].astype(int)
            seg = (num.diff() != 1).cumsum()
            groups = [g for _, g in df.groupby(seg)]
            kind = "連続区間"
        n0 = len(eps)
        for g in groups:
            if len(g) < CHUNK + LABEL_SHIFT:
                continue
            eps.append(dict(
                paths=[os.path.join(d, "images", f) for f in g["image_name"]],
                lin=g["linear_x"].values.astype(np.float32),
                cls=np.array([angular_to_class(a) for a in g["angular_z"]], np.int64)))
        print(f"  {name}: {len(df)}行, {kind} {len(groups)} → 使用 {len(eps) - n0}")
    return eps


def make_index(eps, ep_ids):
    """(エピソード番号, フレーム) の組。最初の正解行動が前進していないサンプルは除外"""
    idx = []
    for e in ep_ids:
        ep = eps[e]
        n = len(ep["lin"])
        for t in range(n):
            j = min(t + LABEL_SHIFT, n - 1)
            if ep["lin"][j] > 0:
                idx.append((e, t))
    return idx


class ChunkDataset(Dataset):
    def __init__(self, eps, index, train=True):
        self.eps, self.index = eps, index
        self.base = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor()])
        # 色相は変えない（球の色が手がかり）
        self.jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2) if train else None
        # 左右の位置が操舵の手がかりなので、ずらしは小さく（横4%・縦5%）
        self.geom = transforms.RandomAffine(0, translate=(0.04, 0.05), scale=(0.95, 1.05)) if (train and USE_AUG) else None
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        e, t = self.index[k]
        ep = self.eps[e]
        im = Image.open(ep["paths"][t]).convert("RGB")
        if self.jitter is not None:
            im = self.jitter(im)
        if self.geom is not None:
            im = self.geom(im)
        x = self.norm(self.base(im))
        n = len(ep["lin"])
        cls = np.ones(CHUNK, np.int64)
        lin = np.zeros(CHUNK, np.float32)
        pad = np.ones(CHUNK, bool)
        for i in range(CHUNK):
            j = t + LABEL_SHIFT + i
            if j < n:
                cls[i], lin[i] = ep["cls"][j], ep["lin"][j]
                pad[i] = not (ep["lin"][j] > 0)   # 停止・後退は学習しない
        return x, torch.from_numpy(cls), torch.from_numpy(lin), torch.from_numpy(pad)


# ---------------- モデル ----------------
def pos2d(h, w, d, device):
    """2D 正弦波位置埋め込み [1, h*w, d]（前半 d/2 が縦、後半 d/2 が横）"""
    ys = torch.arange(h, device=device).float().repeat_interleave(w)
    xs = torch.arange(w, device=device).float().repeat(h)
    dq = d // 4
    freq = 1.0 / (10000 ** (torch.arange(dq, device=device).float() / dq))

    def enc(v):
        a = v[:, None] * freq[None]
        return torch.cat([a.sin(), a.cos()], 1)
    return torch.cat([enc(ys), enc(xs)], 1).unsqueeze(0)


class ACTPolicy(nn.Module):
    def __init__(self, chunk=CHUNK, d=D_MODEL, n_head=N_HEAD, n_enc=N_ENC, n_dec=N_DEC,
                 use_cvae=USE_CVAE, latent=LATENT, n_cls=3, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.chunk, self.d, self.use_cvae, self.latent = chunk, d, use_cvae, latent
        r = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])     # [B,512,H/32,W/32]
        self.proj = nn.Conv2d(512, d, 1)

        def enc_layer():
            return nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer(), n_enc)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, n_head, 4 * d, dropout, batch_first=True), n_dec)
        self.query = nn.Embedding(chunk, d)
        self.latent_proj = nn.Linear(latent, d)
        self.latent_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.head_cls = nn.Linear(d, n_cls)
        self.head_lin = nn.Linear(d, 1)
        if use_cvae:
            self.cvae_token = nn.Parameter(torch.zeros(1, 1, d))
            self.act_embed = nn.Linear(n_cls + 1, d)
            self.cvae_pos = nn.Parameter(torch.randn(1, chunk + 1, d) * 0.02)
            self.cvae_enc = nn.TransformerEncoder(enc_layer(), 2)
            self.to_latent = nn.Linear(d, 2 * latent)

    def forward(self, img, act_cls=None, act_lin=None, is_pad=None):
        B = img.size(0)
        f = self.proj(self.backbone(img))                       # [B,d,h,w]
        h, w = f.shape[-2:]
        tokens = f.flatten(2).transpose(1, 2) + pos2d(h, w, self.d, img.device)

        mu = logvar = None
        if self.use_cvae and act_cls is not None:
            a = torch.cat([F.one_hot(act_cls, 3).float(), act_lin.unsqueeze(-1)], -1)
            x = torch.cat([self.cvae_token.expand(B, -1, -1), self.act_embed(a)], 1) + self.cvae_pos
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=img.device), is_pad], 1)
            hcv = self.cvae_enc(x, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.to_latent(hcv).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = torch.zeros(B, self.latent, device=img.device)  # 推論時は事前分布の平均

        ztok = self.latent_proj(z).unsqueeze(1) + self.latent_pos
        mem = self.encoder(torch.cat([ztok, tokens], 1))
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(q, mem)                               # [B,CHUNK,d]
        return self.head_cls(out), self.head_lin(out).squeeze(-1), mu, logvar


# ---------------- 学習 ----------------
def evaluate(model, dl, device):
    """ステップ0（今まさに実行する行動）の混同行列と、全ステップの正解率"""
    model.eval()
    conf = np.zeros((3, 3), int)
    hit = tot = 0
    with torch.no_grad():
        for x, cls, lin, pad in dl:
            logit = model(x.to(device))[0].argmax(-1).cpu()
            valid = ~pad
            hit += ((logit == cls) & valid).sum().item()
            tot += valid.sum().item()
            m = valid[:, 0]
            for t_, p_ in zip(cls[m, 0].numpy(), logit[m, 0].numpy()):
                conf[t_, p_] += 1
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    bal = recall[conf.sum(1) > 0].mean() if conf.sum() else 0.0
    return conf, recall, bal, hit / max(tot, 1)


def main():
    print("データセット:", DATASETS, " CVAE:", USE_CVAE)
    # 検証は先頭データセットの最後 VAL_RATIO のエピソードのみ（追加データを混ぜても検証条件を揃える）
    eps = load_episodes(DATASETS[:1])
    n0 = len(eps)
    eps += load_episodes(DATASETS[1:]) if len(DATASETS) > 1 else []
    n_val = max(1, int(round(n0 * VAL_RATIO)))
    va_ids = list(range(n0 - n_val, n0))
    tr_ids = [i for i in range(len(eps)) if i not in va_ids]
    tr_idx, va_idx = make_index(eps, tr_ids), make_index(eps, va_ids)
    print(f"エピソード 学習{len(tr_ids)} / 検証{len(va_ids)}   サンプル 学習{len(tr_idx)} / 検証{len(va_idx)}")

    counts = np.zeros(3, int)
    for e, t in tr_idx:
        for i in range(CHUNK):
            j = t + LABEL_SHIFT + i
            if j < len(eps[e]["lin"]) and eps[e]["lin"][j] > 0:
                counts[eps[e]["cls"][j]] += 1
    print("クラス分布(学習, 全ステップ):", dict(zip(CLASS_NAMES, counts.tolist())))
    class_w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)

    tr_dl = DataLoader(ChunkDataset(eps, tr_idx, True), batch_size=BATCH, shuffle=True,
                       num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    va_dl = DataLoader(ChunkDataset(eps, va_idx, False), batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model = ACTPolicy().to(device)
    if FREEZE:
        names = ["0", "1", "2", "3", "4", "5", "6", "7"]   # conv1,bn1,relu,maxpool,layer1..4
        upto = {"layer1": 5, "layer2": 6, "layer3": 7}[FREEZE]
        for n_, p_ in model.backbone.named_parameters():
            if n_.split(".")[0] in names[:upto]:
                p_.requires_grad = False
        print("凍結:", FREEZE, " 学習パラメータ数:", sum(p_.numel() for p_ in model.parameters() if p_.requires_grad))
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": LR}, {"params": bb, "lr": LR_BACKBONE}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    class_w = class_w.to(device)

    cfg = {"chunk": CHUNK, "img_h": IMG_H, "img_w": IMG_W, "mean": MEAN, "std": STD,
           "classes": CLASS_NAMES, "d_model": D_MODEL, "n_head": N_HEAD, "n_enc": N_ENC,
           "n_dec": N_DEC, "use_cvae": USE_CVAE, "latent": LATENT, "label_shift": LABEL_SHIFT,
           "ang_thresh": ANG_THRESH, "datasets": DATASETS, "class_weight": class_w.cpu().tolist(),
           "episodes": {"train": len(tr_ids), "val": len(va_ids)}, "aug": USE_AUG, "freeze": FREEZE, "epochs": EPOCHS}

    best, best_conf = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        s_ce = s_lin = s_kl = 0.0
        nb = 0
        for x, cls, lin, pad in tr_dl:
            x, cls, lin, pad = (v.to(device, non_blocking=True) for v in (x, cls, lin, pad))
            logit, lin_pred, mu, logvar = model(x, cls, lin, pad)
            valid = (~pad).float()
            ce = F.cross_entropy(logit.reshape(-1, 3), cls.reshape(-1), weight=class_w, reduction="none")
            ce = (ce * valid.reshape(-1)).sum() / valid.sum().clamp(min=1)
            l_lin = (((lin_pred - lin) ** 2) * valid).sum() / valid.sum().clamp(min=1)
            loss = ce + LIN_W * l_lin
            kl = torch.zeros((), device=device)
            if mu is not None:
                kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(-1).mean()
                loss = loss + KL_W * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            s_ce, s_lin, s_kl, nb = s_ce + ce.item(), s_lin + l_lin.item(), s_kl + kl.item(), nb + 1
            if MAX_STEPS and nb >= MAX_STEPS:
                break
        sched.step()

        conf, recall, bal, acc_all = evaluate(model, va_dl, device)
        print(f"epoch {ep+1:02d} ce {s_ce/nb:.4f} lin {s_lin/nb:.4f}"
              + (f" kl {s_kl/nb:.3f}" if USE_CVAE else "")
              + "  val(step0) " + " ".join(f"{c}={r:.2f}" for c, r in zip(CLASS_NAMES, recall))
              + f"  bal_acc={bal:.3f}  acc(全step)={acc_all:.3f}", flush=True)
        if bal > best:
            best, best_conf = bal, conf.copy()
            torch.save(model.state_dict(), MODEL_PATH)
            with open(os.path.splitext(MODEL_PATH)[0] + "_config.json", "w") as f:
                json.dump(dict(cfg, best_epoch=ep + 1, best_bal_acc=float(bal)), f, indent=2, ensure_ascii=False)

    print(f"完了: best bal_acc={best:.3f} → {MODEL_PATH}")
    print("ベスト時の混同行列(step0, 行=正解 left/straight/right, 列=予測):\n", best_conf)


if __name__ == "__main__":
    main()
