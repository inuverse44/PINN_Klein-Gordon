import matplotlib.pyplot as plt
import torch

def result(i, x, yh, y=None, xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    if y is not None:
        plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")


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
