import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(sigma_data, filename_prefix, title):
    plt.figure(figsize=(12, 10))

    # 绘制主应力直方图
    for idx, col in enumerate([6, 9, 12], start=1):
        plt.subplot(2, 2, idx)
        counts, bins, _ = plt.hist(
            sigma_data[:, col],
            bins=10,
            weights=np.ones(len(sigma_data)) / len(sigma_data),
            facecolor='k',
            edgecolor='w',
            alpha=0.75
        )
        plt.xlabel('Stress / MPa')
        plt.ylabel('Probability')
        plt.xlim(0, 20)
        plt.ylim(0, 0.5)
        plt.title(title)

        # 转换Y轴为百分比
        yticks = plt.gca().get_yticks()
        plt.gca().set_yticklabels([f"{y * 100:.0f}%" for y in yticks])

    # 绘制极坐标图
    plt.subplot(2, 2, 4, projection='polar')
    angles = np.deg2rad(sigma_data[:, [8, 11, 14]])
    radii = 90 - sigma_data[:, [7, 10, 13]]

    for i in range(3):
        plt.scatter(
            angles[:, i],
            radii[:, i],
            s=5,
            c=['r', 'g', 'b'][i],
            alpha=0.5
        )
    plt.view_init(90, -90)
    plt.title('Orientation of Stress')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./DataMC_SLZK/{filename_prefix}_v2.png')
    plt.savefig(f'./DataMC_SLZK/{filename_prefix}_v2.fig')
    plt.close()

# 示例调用（需加载sigma_ucc5等实际数据）
# plot_histograms(sigma_ucc5, 'M_HTPF_MC_SLZK_5cracks_ucc', 'UCS Control')
# plot_histograms(sigma_utc5, 'M_HTPF_MC_SLZK_5cracks_utc', 'T Control')
# plot_histograms(sigma_uvc5, 'M_HTPF_MC_SLZK_5cracks_uvc', 'T0 & S_v Control')