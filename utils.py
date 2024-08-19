import torch
import matplotlib.pyplot as plt
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

def plot_zfull(zFull):
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
    plt.savefig("exp_pic/forward_process.png")

def plot_generate_sequence(zFull):
    z=zFull.detach().cpu().numpy()
    fig, axs = plt.subplots(1, 17, figsize=(30, 2))
    for i in range(17):
        axs[i].hist2d(z[:,0,i],z[:,1,i],bins=66, range=[[-4, 4], [-4, 4]])
    plt.tight_layout()
    plt.savefig("exp_pic/forward_sequence.png")
    