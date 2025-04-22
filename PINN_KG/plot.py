import matplotlib.pyplot as plt
import torch

def plot_result(x,y,x_data,y_data,yh,xp=None):
        "Pretty plot training results"
        plt.figure(figsize=(8,4))
        plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(x_data[0], y_data[0], s=60, color="tab:orange", alpha=0.4, label='Training data')
        if xp is not None:
            plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                        label='Physics loss training locations')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-1.1, 1.1)
        # plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
        plt.axis("off")