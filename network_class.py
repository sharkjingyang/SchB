import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.functional import pad
import copy
from utils import count_parameters
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x * self.W * 2 * np.pi
    # print(x_proj.shape)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self,hidden_unit=32):
        super(ResidualBlock, self).__init__()
        self.hidden_unit=hidden_unit
        self.conv1 = nn.Linear(hidden_unit,hidden_unit,bias=True)
        self.conv2 = nn.Linear(hidden_unit,hidden_unit,bias=True)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_unit),
            nn.Linear(hidden_unit, hidden_unit))
        self.activation = nn.Softplus()
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        time_embed = self.act(self.embed(t)).squeeze()
        out += time_embed
        out = self.conv2(out)
        out = self.activation(out)

        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self,d=2,hidden_unit=32,num_resblocks=3):
        super(ResNet, self).__init__()
        self.num_resblocks = num_resblocks
        self.hidden_unit=hidden_unit
        self.d=d
        self.init_conv = nn.Linear(self.d,hidden_unit,bias=True)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_unit=self.hidden_unit)
            for _ in range(self.num_resblocks)
        ])
        self.final_conv = nn.Linear(hidden_unit,self.d,bias=False)
        self.activation = nn.Softplus()

    def forward(self, x, t):
        x = self.init_conv(x)
        # x = self.activation(x)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_conv(x)
        
        return x
    

class velocity_net(nn.Module):
    def __init__(self,d):
        super().__init__()
        hidden_unit=32
        self.d=d
        self.net=nn.Sequential(
            nn.Linear(d+1,hidden_unit,bias=True),
            nn.Softplus(),
            nn.Linear(hidden_unit,hidden_unit,bias=True),
            nn.Softplus(),
            nn.Linear(hidden_unit,hidden_unit,bias=True),
            nn.Softplus(),
            nn.Linear(hidden_unit,hidden_unit,bias=True),
            nn.Softplus(),
            nn.Linear(hidden_unit,d,bias=False),
        )

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=hidden_unit),
         nn.Linear(hidden_unit, hidden_unit))
        
    def forward(self,x,t):
        # return  x[:,:self.d]
        z = pad(x, (0, 1, 0, 0), value=t)
        return  self.net(z)
    

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x



class Phi(nn.Module):
    def __init__(self, nTh=2, m=32, d=2, r=10):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
      

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d, m, nTh=nTh)

        # set initial values
        # self.w.weight.data = torch.ones(self.w.weight.data.shape)/m
        self.w.weight.data = torch.zeros(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)



    def forward(self, x,t):
        """ calculating Phi(s, theta)...not used in OT-Flow """
        z = pad(x, (0, 1, 0, 0), value=t)
        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A

        # return self.w(self.N(x))
        return self.w( self.N(z)) + 0.5 * torch.sum( torch.matmul(z , symA) * z , dim=1, keepdims=True) + self.c(z)


    def trHess(self, x, t, justGrad=False ):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        x = pad(x, (0, 1, 0, 0), value=t)

        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()

# new grad
#         grad = z[0]

        if justGrad:
            return grad.t()

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t(), trH+ torch.trace(symA[0:d,0:d])
        # return grad.t(), trH + torch.trace(symA[0:d,0:d])

        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, x, t):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
            + self._hyper_bias(t.view(-1, 1))
    
class ConcatSquash_net(nn.Module):
    def __init__(self,d):
        super(ConcatSquash_net, self).__init__()
        self.d=d
        self.net1=ConcatSquashLinear(dim_in=self.d, dim_out=64)
        self.net2=ConcatSquashLinear(dim_in=64, dim_out=64)
        self.net3=ConcatSquashLinear(dim_in=64, dim_out=64)
        self.net4=ConcatSquashLinear(dim_in=64, dim_out=self.d)
        self.act=nn.Softplus()
      
    def forward(self, x, t):
        t=torch.tensor(t).float().to(x.device)
        x=self.net1(x,t)
        x=self.act(x)
        x=self.net2(x,t)
        x=self.act(x)
        x=self.net3(x,t)
        x=self.act(x)
        x=self.net4(x,t)
        return x

      
    

if __name__ == "__main__":

    device="cuda"
    model=ConcatSquash_net().to(device)
    x=torch.ones(128,2).to(device)
    t=torch.ones(128,1).to(device)
    print("total parameter is: ", count_parameters(model))
    
    print(model(x,t).shape)