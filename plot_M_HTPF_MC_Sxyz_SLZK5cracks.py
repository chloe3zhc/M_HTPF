import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# 假设原始数据已经加载为numpy数组（需要根据实际情况替换数据加载方式）
# sigma_ucc5 = np.loadtxt('sigma_ucc5.txt')
# sigma_utc5 = np.loadtxt('sigma_utc5.txt')
# sigma_uvc5 = np.loadtxt('sigma_uvc5.txt')
# CIf_utc5 = np.loadtxt('CIf_utc5.txt')

# 定义百分比格式化函数
def percentage_formatter(x, pos):
    return f"{x * 100:.0f}%"


# 创建绘图函数
def plot_stress_histograms(data, title, save_path):
    plt.figure(figsize=(15, 10))

    # Sxx
    plt.subplot(2, 3, 1)
    plt.hist(data[:, 0], bins=10, density=True, color='k', edgecolor='none',
             range=(0, 20), weights=np.ones(len(data[:, 0])) / len(data[:, 0]))
    plt.xlabel('Stress $\sigma_{x}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(0, 20)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.title(title)

    # Syy
    plt.subplot(2, 3, 2)
    plt.hist(data[:, 1], bins=10, density=True, color='k', edgecolor='none',
             range=(0, 20), weights=np.ones(len(data[:, 1])) / len(data[:, 1]))
    plt.xlabel('Stress $\sigma_{y}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(0, 20)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Szz
    plt.subplot(2, 3, 3)
    plt.hist(data[:, 2], bins=10, density=True, color='k', edgecolor='none',
             range=(0, 20), weights=np.ones(len(data[:, 2])) / len(data[:, 2]))
    plt.xlabel('Stress $\sigma_{z}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(0, 20)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Sxy
    plt.subplot(2, 3, 4)
    plt.hist(data[:, 3], bins=10, density=True, color='k', edgecolor='none',
             range=(-10, 10), weights=np.ones(len(data[:, 3])) / len(data[:, 3]))
    plt.xlabel('Stress $\tau_{xy}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(-10, 10)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Syz
    plt.subplot(2, 3, 5)
    plt.hist(data[:, 4], bins=10, density=True, color='k', edgecolor='none',
             range=(-10, 10), weights=np.ones(len(data[:, 4])) / len(data[:, 4]))
    plt.xlabel('Stress $\tau_{yz}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(-10, 10)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Sxz
    plt.subplot(2, 3, 6)
    plt.hist(data[:, 5], bins=10, density=True, color='k', edgecolor='none',
             range=(-10, 10), weights=np.ones(len(data[:, 5])) / len(data[:, 5]))
    plt.xlabel('Stress $\tau_{xz}$ /MPa')
    plt.ylabel('Probability')
    plt.xlim(-10, 10)
    plt.ylim(0, 0.5)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.savefig(f"{save_path}.fig")
    plt.close()


# 绘制UCS控制图
plot_stress_histograms(sigma_ucc5, 'UCS control', './DataMC_SLZK/M_HTPF_MC_SLZK_5cracks_ucc_v2_sxyz')

# 绘制T控制图
plot_stress_histograms(sigma_utc5, 'T0 control', './DataMC_SLZK/M_HTPF_MC_SLZK_5cracks_utc_v2_sxyz')

# 绘制SV控制图
plot_stress_histograms(sigma_uvc5, 'T0 & S_v control', './DataMC_SLZK/M_HTPF_MC_SLZK_5cracks_uvc_v2_sxyz')

# 数据提取部分
NN = len(sigma_utc5[:, 0])
sigma_ut = {k: [] for k in range(1, 16)}  # 使用字典存储不同分量的数据

for k in range(15):
    mask = (sigma_utc5[:, k] >= CIf_utc5[0, k]) & (sigma_utc5[:, k] <= CIf_utc5[1, k])
    sigma_ut[k + 1] = sigma_utc5[mask, k].tolist()
