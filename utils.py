import torch
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
def generate_gaussian_mixture_data(num_samples, num_components, means, stds, weights):
    # 随机选择每个样本的高斯分布
    components = torch.multinomial(weights, num_samples, replacement=True)
    # 生成样本
    samples = []
    for i in range(num_samples):
        component = components[i]
        mean = means[component]
        std = stds[component]
        sample = torch.normal(mean, std)
        samples.append(sample)
    return torch.stack(samples)

# 定义高斯混合分布的参数


# 生成一维高斯混合数据
def gaussian_1d_data(num_samples=100000):
    num_components = 2
    means = torch.tensor([-3.0, 3.0])
    stds = torch.tensor([1,1])
    weights = torch.tensor([0.5, 0.5])

    components = torch.multinomial(weights, num_samples, replacement=True)
    samples = []
    for i in range(num_samples):
        component = components[i]
        mean = means[component]
        std = stds[component]
        sample = torch.normal(mean, std)
        samples.append(sample)
    
    data = torch.stack(samples)

    print("Data shape:", data.shape)
    file_path = './data/mix_gaussian.pt'
    torch.save(data, file_path)


def generate_2d(batch_size,data_name):
    rng = np.random.RandomState()

    if data_name=="moon":
        samples = datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        samples = samples.astype("float32")
        samples = samples * 2 + np.array([-1, -0.2])
        return samples
    
    if data_name=="double_well":
        mean_initial = np.array([-1, 0])
        sigma_initial = np.array([[0.0125, 0], [0, 0.15]])
        return np.random.randn(batch_size, 2) @ np.sqrt(sigma_initial) + mean_initial
    

        

    elif data_name == "circles":
        samples = datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        samples = samples.astype("float32")
        samples *= 3
        return samples

    elif data_name == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return x
    elif data_name == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset,dtype=np.float64)
        dataset /= 1.414
        return dataset


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
    n_para=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total parameter is: ", n_para)


def plot_z_state(state_forward,state_backward,save_path):
    z_forward=state_forward.detach().cpu().numpy()
    z_backward=state_backward.detach().cpu().numpy()
    time_plot=z_forward.shape[0]
    fig, axs = plt.subplots(2, time_plot, figsize=(18, 5))
    for i in range(time_plot):
        axs[0,i].hist2d(z_forward[i,:,0],z_forward[i,:,1],bins=100, range=[[-4, 4], [-4, 4]])
        axs[1,i].hist2d(z_backward[i,:,0],z_backward[i,:,1],bins=100, range=[[-4, 4], [-4, 4]])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ode_trajectory(state,save_path):
    z=state.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    for i in range(50):
        plt.plot(z[:,i,0],z[:,i,1],color="white",marker='o', linestyle='-',markersize=3)
    plt.hist2d(z[-1,:,0],z[-1,:,1],bins=100, range=[[-4, 4], [-4, 4]])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ode_trajectory_doublewell(u,state,save_path):
    
    x0 = np.linspace(-2, 2, 100)
    x1 = np.linspace(-2, 2, 100)
    x0, x1 = np.meshgrid(x0, x1)

    x_grid = torch.tensor(np.stack([x0, x1], axis=-1).reshape(-1, 2)).float()
    u_values = u(x_grid).reshape(100, 100)
    z=state.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.contourf(x0, x1, u_values, levels=12)
    for i in range(50):
        plt.plot(z[:,i,0],z[:,i,1],color="white",marker='o', linestyle='-',markersize=3)
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.tight_layout()
    plt.savefig(save_path+"trajectories.png")
    plt.close()

    fig, axs = plt.subplots(1, 6, figsize=(15, 2.5))
    for i in range(6):
        axs[i].hist2d(z[4*i,:,0],z[4*i,:,1],bins=100, range=[[-2, 2], [-2, 2]])
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.tight_layout()
    plt.savefig(save_path+"forward.png")
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