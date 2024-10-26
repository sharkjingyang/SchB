import torch
import matplotlib.pyplot as plt
import numpy as np
#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_zfull(zFull,save_path=None):
    z=zFull.detach().cpu().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    axs[0,0].hist2d(z[:,0,0],z[:,1,0],bins=66, range=[[-4, 4], [-4, 4]])
    axs[0,1].hist2d(z[:,0,1],z[:,1,1],bins=66, range=[[-4, 4], [-4, 4]])
    axs[0,2].hist2d(z[:,0,2],z[:,1,2],bins=66, range=[[-4, 4], [-4, 4]])
    axs[1,0].hist2d(z[:,0,3],z[:,1,3],bins=66, range=[[-4, 4], [-4, 4]])
    axs[1,1].hist2d(z[:,0,4],z[:,1,4],bins=66, range=[[-4, 4], [-4, 4]])
    axs[1,2].hist2d(z[:,0,5],z[:,1,5],bins=66, range=[[-4, 4], [-4, 4]])
    axs[2,0].hist2d(z[:,0,6],z[:,1,6],bins=66, range=[[-4, 4], [-4, 4]])
    axs[2,1].hist2d(z[:,0,7],z[:,1,7],bins=66, range=[[-4, 4], [-4, 4]])
    axs[2,2].hist2d(z[:,0,8],z[:,1,8],bins=66, range=[[-4, 4], [-4, 4]])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_generate_sequence(zFull,save_path=None):
    z=zFull.detach().cpu().numpy()
    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(2):
        for j in range(8):
            axs[i,j].hist2d(z[:,0,i*8+j],z[:,1,i*8+j],bins=66, range=[[-4, 4], [-4, 4]])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_2d_eval(x_data,x_foward,x_backward,x_generate,save_path=None):

    xticks = yticks = [-4, 0, 4]

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs[0,0].hist2d(x_data[:,0],x_data[:,1],bins=66, range=[[-4, 4], [-4, 4]])
    axs[0, 0].set_xticks(xticks)
    axs[0, 0].set_yticks(yticks)
    axs[0, 0].set_title('Data samples')

    axs[0,1].hist2d(x_foward[:,0],x_foward[:,1],bins=66, range=[[-4, 4], [-4, 4]])
    axs[0, 1].set_xticks(xticks)
    axs[0, 1].set_yticks(yticks)
    axs[0, 1].set_title('Forward transform')

    axs[1,0].hist2d(x_backward[:,0],x_backward[:,1],bins=66, range=[[-4, 4], [-4, 4]])
    axs[1, 0].set_xticks(xticks)
    axs[1, 0].set_yticks(yticks)
    axs[1, 0].set_title('Direct inverse transform')

    axs[1,1].hist2d(x_generate[:,0],x_generate[:,1],bins=66, range=[[-4, 4], [-4, 4]])
    axs[1, 1].set_xticks(xticks)
    axs[1, 1].set_yticks(yticks)
    axs[1, 1].set_title('Generate sanples')
  

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()







def plot_loss_all(llh_record,v_pen_record):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].plot(llh_record, label='likelihood')
    axs[0].set_yscale("log")

    axs[1].plot(v_pen_record, label='v_penalty')
    axs[1].set_yscale("log")

    axs[2].plot(np.array(llh_record)*10+np.array(v_pen_record), label='totol loss')
    axs[2].set_yscale("log")

    plt.legend()
    plt.show()
    
def plot_loss(llh_record,log_flag=True):

    plt.plot(llh_record, label='likelihood')
    if log_flag:
        plt.yscale("log")
    plt.legend()
    plt.show()