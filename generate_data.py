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
    if data_name=="moon":
        samples = datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        samples = samples.astype("float32")
        samples = samples * 2 + np.array([-1, -0.2])
        return samples

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



def plot_2d(x):
    fig, axs = plt.subplots()
    fig.set_size_inches(5, 5)
    im1 , _, _, map1 = axs.hist2d(x[:, 0], x[:, 1], range=[[-4, 4], [-4, 4]], bins=66)
    axs.set_title('x from rho_0')


# gaussian_1d_data(num_samples=100000)