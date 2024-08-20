import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import math
import numpy
import copy
from torch import autograd
from torch.optim import Adam
from tqdm import trange
import argparse

from generate_data import *
from network_class import *
from model import *
from utils import *

def log_Gaussian(x,d):
    return -d/2*math.log(2*math.pi)-torch.sum(x**2,dim=1).reshape(-1,1)/2

def grad_log_Gaussian(x):
    return -x


def stepRK4(odefun, z, net,t0, t1):
    h = t1 - t0 # step size
    z0 = z
    K = h * odefun(z0, t0, net)
    z_temp = z0 + (1.0/6.0) * K
    K = h * odefun( z0 + 0.5*K , t0+(h/2) ,net)
    z_temp =z_temp+ (2.0/6.0) * K
    K = h * odefun( z0 + 0.5*K , t0+(h/2) ,net)
    z_temp = z_temp+(2.0/6.0) * K
    K = h * odefun( z0 + K , t0+h ,net)
    z_temp = z_temp+(1.0/6.0) * K
    return z_temp



def stepRK1(odefun, z, net,t0, t1):
    z0 = z
    z_temp =z0 + (t1 - t0) * odefun(z0, t0, net)

    return z_temp

def odefun(x, t, net):
    dx = net(x,t)
    return dx


def odefun_backward(cat_var, t, net):
    d=net.d
    x=cat_var[:,0:d]
    score=cat_var[:,d+1:-1]
    
    divergence_f=torch.zeros(x.shape[0],device=cat_var.device)
    
    f=net(x, t).reshape(x.shape[0],-1)

    jacobian = []
    for i in range(d):
        jacobian_i = autograd.grad(f[:,i].sum(), x, create_graph=True,retain_graph=True)[0]
        # torch.autograd.functional.jacobian
        jacobian.append(jacobian_i)
    stacked_jacobbian = torch.stack(jacobian, dim=2)

    #精确计算divergence_f
    # for i in range(d):
    #     grad = autograd.grad(f[:,i].sum(), x, create_graph=True)[0]
    #     divergence_f+= grad[:,i]
    divergence_f = torch.diagonal(stacked_jacobbian, dim1=-2, dim2=-1).sum(-1).reshape(-1,1)
    
    # # 近似计算divergence_f
    # v=torch.randn(f.shape).to(device)
    # vf=torch.sum(f*v, dim=1)
    # a=autograd.grad(vf.sum(),x,create_graph=True)[0].reshape(x.shape[0],-1)
    # divergence_f=torch.sum(a*v, dim=1)

    d_log=-divergence_f.reshape(-1,1)

    #calculate jacobbian 
    grad_div_f=autograd.grad(divergence_f.sum(), x, create_graph=True,retain_graph=True)[0]
    J_mul_f=torch.bmm(stacked_jacobbian,score.unsqueeze(2)).squeeze()
    d_score=(-grad_div_f-J_mul_f)
    d_penalty=torch.mean(torch.sum((f+score)**2,dim=1))*torch.ones(x.shape[0],1).to(cat_var.device)

    # d_score=torch.zeros(x.shape[0],d).to(device)
    # d_penalty=torch.zeros(x.shape[0],1).to(device)

    return torch.cat((f,d_log,d_score,d_penalty),dim=1)


def train(args):
    device="cuda"
    tspan=[0,1]
    
    # h=(tspan[1]-tspan[0])/args.nt
    model = Bridge_2d(d=args.d,nt=args.nt).to(device)
    optim = Adam(model.parameters(), lr=args.lr)


    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - 0.995
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    # scheduler=OneCycleLR(optim,args.lr,total_steps=args.epochs*1000,pct_start=0.25,anneal_strategy='cos')


    tqdm_epoch = trange(args.epochs)
    llh_record=[]
    v_pen_record=[]
    x=torch.tensor(generate_2d(batch_size=args.batch_size,data_name=args.dataset)).float().to(device)

    

    for epoch in tqdm_epoch:
        
        #resample
        if epoch %50==0:
            x=torch.tensor(generate_2d(batch_size=args.batch_size,data_name=args.dataset)).float()
            x = x.to(device)
        loss,log_temp,bridge_pen=model(x)

        loss.backward()
        optim.step()
        # scheduler.step()
        # if epoch%args.model_ema_steps==0:
        #     model_ema.update_parameters(model)


        llh_record.append(-torch.mean(log_temp).detach().cpu())
        v_pen_record.append(bridge_pen.detach().cpu())
        tqdm_epoch.set_description('Loss: {:5f}'.format(-torch.mean(log_temp).detach().cpu()))
        torch.save(model.state_dict(), 'experiment/'+args.dataset+'_2d.pth')
        # if (epoch+1) % 200 == 0:
        #         for p in optim.param_groups:
        #             p['lr'] /= 2
        #         print("lr: ", p['lr'])

    # x=torch.tensor(generate_2d(batch_size=10000,data_name=args.dataset)).float().to(device)
    # zFull = torch.zeros(*x.shape , args.nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
    # zFull[:,:,0] = x

    # tk=0
    # for k in range(args.nt):
    #     tk=h*k
    #     zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , f_theta, tk, tk+h)
    #     z=zFull.detach().cpu().numpy()

    # plot_zfull(zFull)

    # x=torch.randn(10000,2).to(device)
    # nt_val=16
    # h_val=1/nt_val
    # zFull = torch.zeros(*x.shape , nt_val+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
    # zFull[:,:,0] = x

    # for k in range(nt_val):
    #     tk=1-h_val*k
    #     zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , f_theta, tk, tk-h_val)

    # plot_generate_sequence(zFull)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.004)
    parser.add_argument('--d',type = int ,default=2)
    parser.add_argument('--nt',type = int ,default=8)
    parser.add_argument('--dataset',type = str ,default="checkerboard")
    parser.add_argument('--batch_size',type = int ,default=1000)    
    parser.add_argument('--epochs',type = int,default=400)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='results/ckpt.pt')
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)

    args = parser.parse_args()
    train(args)