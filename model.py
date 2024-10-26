import torch.nn as nn
import torch
import math
from torch import autograd
import gc

from network_class import *
from utils import *


class Bridge_2d(nn.Module):
    def __init__(self,d,nt,net_class="ConcatSquash"):
        super().__init__()
        self.d=d
        if net_class=="MLP_cat":
            self.f_theta=velocity_net(d=self.d)
        if net_class=="ConcatSquash":
            self.f_theta=ConcatSquash_net(d=self.d)
            
        
        self.nt=nt
        self.h=1/nt
    
    def forward(self,x):
        z_temp,tk=x,0
        for k in range(self.nt):
            tk=self.h*k
            z_temp = self.stepRK4(self.odefun, z_temp , self.f_theta, tk, tk+self.h)

        z_terminal=z_temp
        log_terminal=self.log_Gaussian(x=z_terminal,d=self.d)
        log_temp=log_terminal
    
        score_terminal=self.grad_log_Gaussian(z_terminal)
        score_temp=score_terminal
        

        tk=1
        z_back=z_terminal
        cat_var=torch.cat((z_back,log_temp,score_temp),dim=1)
        cat_var=pad(cat_var,(0,1,0,0),value=0)
        
        # print("begin---------------")
        # print("Allocated memory:", torch.cuda.memory_allocated(3)/1024**3)

        for k in  range(self.nt):
            cat_var= self.stepRK4(self.odefun_backward, cat_var , self.f_theta, tk, tk-self.h)
            tk-=self.h
        # print("Allocated memory:", torch.cuda.memory_allocated(3)/1024**3)
        # print("end---------------")



        z,log_temp,bridge_pen=cat_var[:,0:self.d],cat_var[:,self.d],-cat_var[0,-1]
        loss=-torch.mean(log_temp)*10+bridge_pen
        return loss,log_temp,bridge_pen

    def log_Gaussian(self,x,d):
        return -d/2*math.log(2*math.pi)-torch.sum(x**2,dim=1).reshape(-1,1)/2

    def grad_log_Gaussian(self,x):
        return -x

    def stepRK4(self,odefun, z, net,t0, t1):
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


    def odefun(self,x, t, net):
        dx = net(x,t)
        return dx


    def odefun_backward(self,cat_var, t, net):
        device=cat_var.device
        d=net.d
        x=cat_var[:,0:d]
        score=cat_var[:,d+1:-1]
    
        divergence_f=torch.zeros(x.shape[0],device=device)
        
        f=net(x, t).reshape(x.shape[0],-1)

        jacobian = []
        for i in range(d):
            jacobian_i = autograd.grad(f[:,i].sum(), x, create_graph=True,retain_graph=True)[0]
            # torch.autograd.functional.jacobian
            jacobian.append(jacobian_i)
        stacked_jacobbian = torch.stack(jacobian, dim=2)
        #精确计算divergence_f
        # divergence_f = torch.diagonal(stacked_jacobbian, dim1=-2, dim2=-1).sum(-1).reshape(-1,1)

        # for i in range(d):
        #     grad = autograd.grad(f[:,i].sum(), x, create_graph=True)[0]
        #     divergence_f+= grad[:,i]
    
        # 近似计算divergence_f
        v=torch.randn(f.shape).to(device)
        vf=torch.sum(f*v, dim=1)
        a=autograd.grad(vf.sum(),x,create_graph=True)[0].reshape(x.shape[0],-1)
        divergence_f=torch.sum(a*v, dim=1)


        d_log=-divergence_f.reshape(-1,1)
        #calculate jacobbian 
        grad_div_f=autograd.grad(divergence_f.sum(), x, create_graph=True,retain_graph=True)[0]
        J_mul_f=torch.bmm(stacked_jacobbian,score.unsqueeze(2)).squeeze()
        d_score=(-grad_div_f-J_mul_f)
        d_penalty=torch.mean(torch.sum((f+score)**2,dim=1))*torch.ones(x.shape[0],1).to(device)


        # d_score=torch.zeros(x.shape[0],d).to(device)
        # d_penalty=torch.zeros(x.shape[0],1).to(device)
        return torch.cat((f,d_log,d_score,d_penalty),dim=1)

    def forward_evolution(self,x,save_path=None):
        zFull = torch.zeros(*x.shape , self.nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = x

        tk=0
        for k in range(self.nt):
            tk=self.h*k
            zFull[:,:,k+1] = self.stepRK4(self.odefun, zFull[:,:,k] , self.f_theta, tk, tk+self.h)
        
        if save_path:
            plot_zfull(zFull,save_path=save_path+'/forward_process.pdf')

    
    def generate_evolution(self,x,save_path=None):
        nt_val=15
        h_val=1/nt_val
        zFull = torch.zeros(*x.shape , nt_val+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = x

        for k in range(nt_val):
            tk=1-h_val*k
            zFull[:,:,k+1] = self.stepRK4(self.odefun, zFull[:,:,k] , self.f_theta, tk, tk-h_val)

        if save_path:
            plot_generate_sequence(zFull,save_path=save_path+'/generate_process.pdf')

    
    def forward_backward_eval(self,x_data,x_random,save_path):
        nt_val=8
        h_val=1/nt_val
        x_forward=x_data.clone()
        for k in range(nt_val):
            tk=h_val*k
            x_forward = self.stepRK4(self.odefun, x_forward , self.f_theta, tk, tk+h_val)

        x_backward=x_forward.clone()
        for k in range(nt_val):
            tk=1-h_val*k
            x_backward = self.stepRK4(self.odefun, x_backward , self.f_theta, tk, tk-h_val)

        x_generate=x_random.clone()
        for k in range(nt_val):
            tk=1-h_val*k
            x_generate = self.stepRK4(self.odefun, x_generate , self.f_theta, tk, tk-h_val)

        if save_path:
            plot_2d_eval(x_data.detach().cpu().numpy(),x_forward.detach().cpu().numpy(),x_backward.detach().cpu().numpy(),x_generate.detach().cpu().numpy(),save_path=save_path+'/plot4_val.pdf')

        



if __name__=="__main__":
    device="cuda"
    model=Bridge_2d(d=2,nt=8).to(device)
    x=torch.ones(128,2).to(device)
    print(model(x).shape)
    print(model(x))
