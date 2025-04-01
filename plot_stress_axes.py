import numpy as np
import matplotlib.pyplot as plt


def plot_stress_axes(sigma_vector_1_, sigma_vector_2_, sigma_vector_3_, plot_file):
    """
    绘制主应力方向的置信度图。

    参数：
    sigma_vector_1_, sigma_vector_2_, sigma_vector_3_：numpy数组
        每个数组的形状为 (3, N)，表示三个主应力方向的分量（x, y, z）。
    plot_file：str
        保存绘图结果的文件名（不带扩展名）。
    """
    # 获取每个应力方向向量的数量
    N = sigma_vector_1_.shape[1]

    # 初始化用于存储应力方向投影结果的列表
    x_sigma_1, y_sigma_1 = [], []
    x_sigma_2, y_sigma_2 = [], []
    x_sigma_3, y_sigma_3 = [], []

    # 遍历每个应力方向向量
    for i in range(N):
        # 提取当前向量
        sigma_vector_1 = sigma_vector_1_[:, i]
        sigma_vector_2 = sigma_vector_2_[:, i]
        sigma_vector_3 = sigma_vector_3_[:, i]

        # 确保 z 分量为正（如果不是，则翻转向量的方向）
        if sigma_vector_1[2] < 0:
            sigma_vector_1 *= -1
        if sigma_vector_2[2] < 0:
            sigma_vector_2 *= -1
        if sigma_vector_3[2] < 0:
            sigma_vector_3 *= -1

        def get_azimuth_theta(vector):
            """
            计算向量的方位角（azimuth）和极角（theta）。

            参数：
            vector：numpy数组，形状为 (3,)
                表示应力方向向量。

            返回：
            azimuth：float
                方位角（0 到 360 度）。
            theta：float
                极角（0 到 90 度）。
            """
            # 计算方位角（与 x 轴的夹角）
            fi = np.degrees(np.arctan(np.abs(vector[1] / vector[0])))
            if vector[1] >= 0 and vector[0] >= 0:
                azimuth = fi
            elif vector[1] >= 0 and vector[0] < 0:
                azimuth = 180 - fi
            elif vector[1] < 0 and vector[0] < 0:
                azimuth = 180 + fi
            else:
                azimuth = 360 - fi

            # 计算极角（与 z 轴的夹角）
            theta = np.degrees(np.arccos(np.abs(vector[2])))
            return azimuth, theta

        # 获取每个应力方向向量的方位角和极角
        azimuth_sigma_1, theta_sigma_1 = get_azimuth_theta(sigma_vector_1)
        azimuth_sigma_2, theta_sigma_2 = get_azimuth_theta(sigma_vector_2)
        azimuth_sigma_3, theta_sigma_3 = get_azimuth_theta(sigma_vector_3)

        # 根据极角计算投影的半径（在单位球坐标上）
        radius_sigma_1 = np.sin(np.radians(theta_sigma_1))
        radius_sigma_2 = np.sin(np.radians(theta_sigma_2))
        radius_sigma_3 = np.sin(np.radians(theta_sigma_3))

        # 将球坐标转换为平面投影坐标（x, y），并保存到列表中
        x_sigma_1.append(np.sqrt(2) * radius_sigma_1 * np.cos(np.radians(azimuth_sigma_1)))
        y_sigma_1.append(np.sqrt(2) * radius_sigma_1 * np.sin(np.radians(azimuth_sigma_1)))

        x_sigma_2.append(np.sqrt(2) * radius_sigma_2 * np.cos(np.radians(azimuth_sigma_2)))
        y_sigma_2.append(np.sqrt(2) * radius_sigma_2 * np.sin(np.radians(azimuth_sigma_2)))

        x_sigma_3.append(np.sqrt(2) * radius_sigma_3 * np.cos(np.radians(azimuth_sigma_3)))
        y_sigma_3.append(np.sqrt(2) * radius_sigma_3 * np.sin(np.radians(azimuth_sigma_3)))

    # 创建绘图
    plt.figure(figsize=(8, 8))
    plt.title('Confidence of principal stress axes', fontsize=16)
    plt.axis('off')  # 不显示坐标轴
    plt.axis('equal')  # 设置坐标轴比例相等
    plt.xlim(-1.05, 1.70)  # 设置 x 轴范围
    plt.ylim(-1.05, 1.05)  # 设置 y 轴范围

    # 绘制三个主应力方向的点
    plt.plot(y_sigma_1, x_sigma_1, 'r.', markersize=20, linewidth=1.5, label='sigma 1')
    plt.plot(y_sigma_2, x_sigma_2, 'g.', markersize=20, linewidth=1.5, label='sigma 2')
    plt.plot(y_sigma_3, x_sigma_3, 'b.', markersize=20, linewidth=1.5, label='sigma 3')

    # 绘制单位圆的轮廓
    fi = np.arange(0, 360, 0.1)
    plt.plot(np.cos(np.radians(fi)), np.sin(np.radians(fi)), 'k', linewidth=2.0)
    plt.plot(0, 0, 'k+', markersize=10)  # 绘制圆心

    # 绘制径向网格线（每隔 15 度绘制一条）
    for theta_grid in np.arange(0, 91, 15):
        radius_grid = np.sin(np.radians(theta_grid))
        x_grid = np.sqrt(2) * radius_grid * np.cos(np.radians(fi))
        y_grid = np.sqrt(2) * radius_grid * np.sin(np.radians(fi))
        plt.plot(y_grid, x_grid, 'k:', linewidth=0.5)

    # 绘制方位角网格线（每隔 15 度绘制一条）
    for fi_grid in np.arange(0, 361, 15):
        theta_grid = np.arange(0, 91, 15)
        radius_grid = np.sin(np.radians(theta_grid))
        x_grid = np.sqrt(2) * radius_grid * np.cos(np.radians(fi_grid))
        y_grid = np.sqrt(2) * radius_grid * np.sin(np.radians(fi_grid))
        plt.plot(y_grid, x_grid, 'k:', linewidth=0.5)

    # 添加图例
    plt.legend(loc='southeast', fontsize=14)

    # 保存图像为 PNG 和 MATLAB 格式
    plt.savefig(plot_file + '.png', format='png')
    plt.savefig(plot_file + '.fig')

    # 显示图像
    plt.show()

# 示例用法：
# 假设 sigma_vector_1_, sigma_vector_2_, sigma_vector_3_ 是已定义的 numpy 数组
# plot_stress_axes(sigma_vector_1_, sigma_vector_2_, sigma_vector_3_, 'stress_axes_plot')