import matplotlib.pyplot as plt
import numpy as np


def plot_stress_axes(sigma_vector_1_, sigma_vector_2_, sigma_vector_3_, plot_file):
    N = sigma_vector_1_.shape[1]

    x_sigma_1, y_sigma_1 = [], []
    x_sigma_2, y_sigma_2 = [], []
    x_sigma_3, y_sigma_3 = [], []

    for i in range(N):
        # 处理每个应力向量方向...
        # (保持原有方向计算逻辑，转换为Python语法)

        # 计算投影坐标
        x_sigma_1.append(np.sqrt(2) * radius_sigma_1 * np.cos(azimuth_sigma_1 * np.pi / 180))
        y_sigma_1.append(np.sqrt(2) * radius_sigma_1 * np.sin(azimuth_sigma_1 * np.pi / 180))
        # 对 sigma_2 和 sigma_3 进行类似操作...

    # 绘制极坐标图
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.axis('equal')
    plt.xlim(-1.05, 1.70)
    plt.ylim(-1.05, 1.05)

    plt.plot(y_sigma_1, x_sigma_1, 'r.', markersize=20, label='$\sigma_1$')
    plt.plot(y_sigma_2, x_sigma_2, 'g.', markersize=20, label='$\sigma_2$')
    plt.plot(y_sigma_3, x_sigma_3, 'b.', markersize=20, label='$\sigma_3$')

    # 添加网格和边界...
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(plot_file + '.png')
    plt.savefig(plot_file + '.fig')
    plt.close()