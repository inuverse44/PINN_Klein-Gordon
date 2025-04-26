import numpy as np
import torch
import matplotlib.pyplot as plt

from PINN_KG.gif import save_gif_PIL
from PINN_KG import plot
from PINN_KG.neural_network import FullyConnectedNetwork
from PINN_KG.physics.physics import klein_gordon_equation
from PINN_KG.physics.physics import harmonic_oscillator_equation

def main_KG():


    # get the analytical solution over the full domain
    x = torch.linspace(0,70,1000).view(-1,1)
    # y = oscillator(d, w0, x).view(-1,1)
    
    # slice out a small number of points from the LHS of the domain
    # x_data = x[0:200:20]
    # y_data = y[0:200:20]

    """=================================================="""
    """                      PINN                        """
    """=================================================="""


    x_physics = torch.linspace(0,70,1000).view(-1,1).requires_grad_(True)# sample locations over the problem domain

    torch.manual_seed(123)
    model = FullyConnectedNetwork(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    files = []

    # for loss history
    total_loss_history = []
    physics_loss_history = []
    initial_loss_history = []
    for i in range(10000):
        optimizer.zero_grad()
                
        # compute the "physics loss"
        yhp = model(x_physics)
        dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
        dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
        residual = klein_gordon_equation(yhp, dx, dx2)
        physics_loss = torch.mean(residual**2)
    
        # compute the "initial condition loss"
        x0 = torch.zeros(1,1, requires_grad=True)
        y0 = model(x0)
        dy0 = torch.autograd.grad(y0, x0, torch.ones_like(y0), create_graph=True)[0]
        initial_loss = (y0 - 16.0)**2 + (dy0 - 0.0)**2

        if i < 2000:
            loss = 1e0*initial_loss + 0*physics_loss
        else:
            loss = 1e-4*physics_loss + 1e0*initial_loss

        
        # backpropagate joint loss
        loss.backward()
        final_loss = loss
        optimizer.step()   

        # loss history
        physics_loss_history.append(physics_loss.item())
        initial_loss_history.append(initial_loss.item())
        total_loss_history.append(loss.item())

        # plot the result as training progresses
        if (i+1) % 100 == 0: 
            yh = model(x).detach()
            xp = x_physics.detach()
            
            #plot.result(i, x, y, yh, xp)
            plot.result(i, x, yh=yh, xp=xp)
            
            file = "plots/pinn_%.8i.png"%(i+1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            
            # if (i+1) % 6000 == 0: plt.show()
            # else: plt.close("all")
            plt.close("all")
            print(f"Epoch: {i+1}")


    """=================================================="""
    """                     OUTPUT                       """
    """=================================================="""
                
    save_gif_PIL("pinn.gif", files, fps=20, loop=0)

    plot.loss_history(
        total_loss_history,
        physics_loss_history,
        initial_loss_history,
        labels=["Total Loss", "Physics Loss (Residual)", "Initial Condition Loss"],
        title="Loss History (Decomposed)",
        filename="loss_history_decomposed.png"
    )
