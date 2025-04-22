import numpy as np
import torch
import matplotlib.pyplot as plt

from PINN_KG.gif import save_gif_PIL
from PINN_KG.plot import plot_result
from PINN_KG.neural_network import FullyConnectedNetwork

def main_KG():
        
    def oscillator(d, w0, x):
        """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
        assert d < w0
        w = np.sqrt(w0**2-d**2)
        phi = np.arctan(-d/w)
        A = 1/(2*np.cos(phi))
        cos = torch.cos(phi+w*x)
        sin = torch.sin(phi+w*x)
        exp = torch.exp(-d*x)
        y  = exp*2*A*cos
        return y
    
    d, w0 = 2, 20

    # get the analytical solution over the full domain
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)
    print(x.shape, y.shape)

    # slice out a small number of points from the LHS of the domain
    x_data = x[0:200:20]
    y_data = y[0:200:20]
    print(x_data.shape, y_data.shape)

    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()

    # """=================================================="""
    # """                        NN                        """
    # """=================================================="""

    # # train standard neural network to fit training data
    # torch.manual_seed(123)
    # model = FullyConnectedNetwork(1,1,32,3)
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    # files = []
    # for i in range(1000):
    #     optimizer.zero_grad()
    #     yh = model(x_data)
    #     loss = torch.mean((yh-y_data)**2)# use mean squared error
    #     loss.backward()
    #     optimizer.step()
        
        
    #     # plot the result as training progresses
    #     if (i+1) % 10 == 0: 
            
    #         yh = model(x).detach()
            
    #         plot_result(x,y,x_data,y_data,yh)
            
    #         file = "plots/nn_%.8i.png"%(i+1)
    #         plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    #         files.append(file)
        
    #         if (i+1) % 500 == 0: plt.show()
    #         else: plt.close("all")
                
    # save_gif_PIL("nn.gif", files, fps=20, loop=0)


    """=================================================="""
    """                      PINN                        """
    """=================================================="""

    x_physics = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)# sample locations over the problem domain
    mu, k = 2*d, w0**2

    torch.manual_seed(123)
    model = FullyConnectedNetwork(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-3)
    files = []
    final_loss = 0
    for i in range(20000):
        optimizer.zero_grad()
                
        # compute the "physics loss"
        yhp = model(x_physics)
        dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
        dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
        physics_loss = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
    
        # compute the "initial condition loss"
        x0 = torch.zeros(1,1, requires_grad=True)
        y0 = model(x0)
        dy0 = torch.autograd.grad(y0, x0, torch.ones_like(y0), create_graph=True)[0]
        initial_loss = (y0 - 1.0)**2 + (dy0 - 0.0)**2

        # final loss
        loss = 1e-4*torch.mean(physics_loss**2) + 1e0*initial_loss
        
        # backpropagate joint loss
        loss.backward()
        final_loss = loss
        optimizer.step()
        
        # plot the result as training progresses
        if (i+1) % 150 == 0: 

            yh = model(x).detach()
            xp = x_physics.detach()
            
            plot_result(x,y,x_data,y_data,yh,xp)
            
            file = "plots/pinn_%.8i.png"%(i+1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            
            if (i+1) % 6000 == 0: plt.show()
            else: plt.close("all")
                
    print(final_loss)
    save_gif_PIL("pinn.gif", files, fps=20, loop=0)