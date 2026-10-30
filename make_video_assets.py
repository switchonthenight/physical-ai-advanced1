import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib import font_manager as fm
names = {f.name for f in fm.fontManager.ttflist}
JP = next(n for n in ["Noto Sans CJK JP", "Noto Serif CJK JP", "TakaoGothic", "TakaoMincho"] if n in names)
plt.rcParams.update({"font.family": JP, "text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white", "axes.edgecolor": "#888"})
BG = "#14171c"; L = os.path.expanduser("~/ros2_ws/logs"); OUT = os.path.expanduser("~/ros2_ws/video_assets")
os.makedirs(OUT, exist_ok=True)
BALLS = [("red", 3.5, 0.5), ("green", 7.0, -0.5), ("blue", 10.5, 0.5)]

def fig():
    f = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG); return f

# ---------- 1. 結果表 ----------
def table(fname, title, header, rows, hl=None, note=None):
    f = fig(); ax = f.add_axes([0.06, 0.08, 0.88, 0.72]); ax.axis("off")
    f.text(0.5, 0.9, title, ha="center", fontsize=44, weight="bold")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(28); t.scale(1, 3.4)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#555"); cell.get_text().set_color("white")
        cell.set_facecolor("#2b3a55" if r == 0 else ("#3d3418" if hl is not None and r - 1 in hl else "#1f232a"))
        if r == 0 or (hl is not None and r - 1 in hl): cell.get_text().set_weight("bold")
    if note: f.text(0.5, 0.06, note, ha="center", fontsize=24, color="#bbb")
    f.savefig(f"{OUT}/{fname}", facecolor=BG); plt.close(f)

table("table_normal.png", "通常スタート（各10回・80秒）",
      ["モデル", "ゴール", "平均時間", "球との最小距離（赤/緑/青）"],
      [["CNN r3", "8/10", "34.6秒", "–（失敗2回は青玉に接触）"],
       ["ACT v3", "10/10", "32.5秒", "1.04 / 1.04 / 1.34 m"],
       ["ACT v4（最終）", "10/10", "33.2秒", "1.09 / 1.20 / 1.28 m"]], hl=[2])
table("table_recovery.png", "ずれた姿勢からの復帰（固定15姿勢・各60秒）",
      ["モデル", "正しい側でゴール", "逆走", "周回・停滞"],
      [["CNN r3", "3/15", "3", "8"], ["ACT v3", "5/15", "4", "4"], ["ACT v4（最終）", "6/15", "3", "6"]],
      hl=[2], note="後ろ向きのスタート（球が見えない）では3モデルとも逆走 ＝ カメラ1台だけで学ぶ方式の限界")
table("table_act_progress.png", "ACT の改善過程（通常スタート10回）",
      ["版", "変更点", "ゴール"],
      [["v1", "ベースライン（過学習）", "0/10"],
       ["v2", "データ拡張＋layer2まで凍結", "0/10"],
       ["v2 案A", "TE を新しい予測重視に変更", "3/10"],
       ["v3", "旧 CNN データを追加", "10/10"],
       ["v4", "復帰デモを追加", "10/10"]], hl=[3, 4],
      note="オフラインの bal_acc は 0.666→0.687 とほぼ同じでも、実走は 3/10→10/10")

# ---------- 2. 構成図 ----------
f = fig(); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 19.2); ax.set_ylim(0, 10.8); ax.axis("off")
ax.text(9.6, 9.8, "ACT（Action Chunking with Transformers）簡易版の構成", ha="center", fontsize=40, weight="bold")
boxes = [("車載カメラ画像\n640×480", "#355c7d"), ("ResNet18\n（layer2まで凍結）", "#4a6fa5"),
         ("Transformer\nEnc×2 / Dec×2", "#6c5b7b"), ("10ステップの\nアクションチャンク", "#8e5572"),
         ("Temporal\nEnsembling", "#a0634a"), ("安全層・\nスタック検出", "#5b7a4a")]
w, h, y = 2.55, 2.0, 5.4; xs = np.linspace(0.5, 19.2 - 0.5 - w, len(boxes))
for i, ((txt, c), x) in enumerate(zip(boxes, xs)):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="white", lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=23, weight="bold")
    if i < len(boxes) - 1:
        ax.annotate("", xy=(xs[i + 1] - 0.08, y + h / 2), xytext=(x + w + 0.08, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=25))
ax.annotate("", xy=(xs[-1] + w / 2, 3.2), xytext=(xs[-1] + w / 2, y - 0.1),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=25))
ax.text(xs[-1] + w / 2, 2.8, "/cmd_vel（速度・旋回）", ha="center", fontsize=24, weight="bold")
notes = [(xs[3] + w / 2, "各ステップ：左／直進／右 の3クラス ＋ 速度"),
         (xs[4] + w / 2, "重なった予測を新しいもの重視で平均")]
for x, t in notes: ax.text(x, 4.75, t, ha="center", fontsize=18, color="#ddd")
ax.text(9.6, 1.4, "学習データ：手動操作のデモ（通常 41本＋復帰 32本＋旧CNNデータ）／入力はカメラ画像のみ",
        ha="center", fontsize=24, color="#ccc")
f.savefig(f"{OUT}/method_diagram.png", facecolor=BG); plt.close(f)

# ---------- 3. 軌跡の比較 ----------
def traj(label):
    d = np.genfromtxt(f"{L}/eval_{label}.csv", delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    return d[:, 1] + 0.5 * np.cos(d[:, 3]), d[:, 2] + 0.5 * np.sin(d[:, 3])

def course(ax, x0=-1.5, x1=14.5):
    ax.set_facecolor("#1f232a")
    for yy in (2.4, -2.4): ax.axhline(yy, color="#e0c040", lw=6)
    ax.axvline(13.0, color="white", lw=3, ls="--"); ax.text(13.1, 2.0, "GOAL", fontsize=20, weight="bold")
    for c, bx, by in BALLS: ax.add_patch(Circle((bx, by), 0.3, color=c, zorder=5))
    ax.set_xlim(x0, x1); ax.set_ylim(-2.9, 2.9); ax.set_aspect("equal"); ax.set_xlabel("x [m]", fontsize=18); ax.set_ylabel("y [m]", fontsize=18)

def compare(fname, title, pairs, start=None):
    f = fig(); ax = f.add_axes([0.06, 0.12, 0.9, 0.68])
    f.text(0.5, 0.88, title, ha="center", fontsize=42, weight="bold"); course(ax)
    for label, name, col in pairs:
        x, y = traj(label); ax.plot(x, y, color=col, lw=4, label=name, zorder=6)
        ax.plot(x[-1], y[-1], "o", color=col, ms=14, mec="white", zorder=7)
    if start is not None: ax.plot(*start, "*", color="white", ms=30, mec="black", zorder=8, label="スタート")
    ax.legend(fontsize=22, loc="lower right", facecolor="#2b2f36", labelcolor="white")
    f.savefig(f"{OUT}/{fname}", facecolor=BG); plt.close(f)

compare("traj_normal_v4_vs_cnn.png", "通常スタートの軌跡：ACT v4 と CNN",
        [("video_cnn_normal", "CNN r3（最小距離 0.95m）", "#ff8c42"), ("video_v4_normal", "ACT v4（最小距離 1.07m）", "#4fc3f7")],
        start=(0.5, 0.0))
compare("traj_rec14_v4_vs_cnn.png", "復帰 #14（青玉手前で大きくずれた姿勢）",
        [("video_cnn_rec14", "CNN r3（壁に激突してスタック）", "#ff8c42"), ("video_v4_rec14", "ACT v4（13秒でゴール）", "#4fc3f7")],
        start=(9.24 + 0.5 * np.cos(np.radians(-81)), 0.56 + 0.5 * np.sin(np.radians(-81))))
print("font:", JP); print("\n".join(sorted(os.listdir(OUT))))
