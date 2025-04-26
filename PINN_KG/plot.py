import matplotlib.pyplot as plt
import torch

def result(i, x, yh, y=None, xp=None):
    "Pretty plot training results with axes and ticks"
    plt.figure(figsize=(8,4))
    
    # ニューラルネットの予測
    plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    
    # 正解データがあれば描画
    if y is not None:
        plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    
    # 物理ロスのサンプリングポイントがあれば描画
    if xp is not None:
        plt.scatter(xp, torch.zeros_like(xp), s=60, color="tab:green", alpha=0.4, label='Physics loss training locations')
    
    # 凡例
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    
    # 軸設定
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-1.1, 1.1)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    
    # トレーニングステップ数を右上に表示
    plt.text(1.07, 0.9, f"Step: {i+1}", fontsize="x-large", color="k")
    
    # グリッド追加
    plt.grid(True, linestyle="--", alpha=0.5)

    # 軸を消さずにそのまま表示



def loss_history(*loss_histories, labels=None, title="Loss History", log_scale=True, filename=None):
    """
    複数の損失履歴をログスケールで描画する汎用関数。

    Parameters:
    - *loss_histories: 可変長のリスト群（それぞれ1つのロス履歴）
    - labels: 各ロス履歴に対応する凡例ラベルのリスト（省略可）
    - title: グラフタイトル
    - log_scale: Y軸をlogスケールにするかどうか
    - filename: ファイルに保存する場合はファイル名（Noneなら表示のみ）
    """
    plt.figure(figsize=(10,6))
    
    styles = ['-', '--', ':', '-.']  # スタイルを順番に使う
    for i, history in enumerate(loss_histories):
        label = labels[i] if labels and i < len(labels) else f"Loss {i+1}"
        linestyle = styles[i % len(styles)]
        plt.plot(history, label=label, linestyle=linestyle, linewidth=2 if i == 0 else 1.5)
    
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()
