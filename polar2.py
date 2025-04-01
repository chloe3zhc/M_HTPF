import numpy as np
import matplotlib.pyplot as plt


def polar2(theta, rho, r=None, line_style='auto', ax=None):
    """
    创建极坐标图。

    参数:
    - theta: array-like，角度（弧度）。
    - rho: array-like，与每个角度对应的半径。
    - r: list 或 None，径向和角度范围 [rmin, rmax, thmin, thmax]。
    - line_style: str，绘图的线型。
    - ax: matplotlib.axes.Axes，可选，绘图的坐标轴。

    返回:
    - hpol: matplotlib.lines.Line2D，绘制的线对象。
    """
    if ax is None:
        ax = plt.gca(projection='polar')  # 获取当前的极坐标轴或创建新的极坐标轴

    # 验证输入
    if isinstance(theta, str) or isinstance(rho, str):
        raise ValueError("输入参数必须为数值类型。")
    if theta.shape != rho.shape:
        raise ValueError("THETA 和 RHO 的大小必须相同。")

    # 处理径向范围
    if r is not None:
        if len(r) == 2:
            rmin, rmax = r
            thmin, thmax = 0, 2 * np.pi
        elif len(r) == 4:
            rmin, rmax, thmin, thmax = r
        else:
            raise ValueError("R 必须是长度为 2 或 4 的数组。")
    else:
        rmin, rmax = 0, np.max(np.abs(rho))
        thmin, thmax = 0, 2 * np.pi

    # 根据范围筛选数据
    subset = (rho >= rmin) & (rho <= rmax) & (theta >= thmin) & (theta <= thmax)
    theta, rho = theta[subset], rho[subset]

    # 绘制网格和背景（如果需要）
    ax.grid(True)
    ax.set_facecolor(ax.get_xcolor())  # 将背景颜色匹配为 x 轴文本颜色

    # 设置角度刻度
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels([f'{int(i * 30)}' for i in range(12)])

    # 设置径向刻度
    radial_ticks = np.arange(rmin, rmax, (rmax - rmin) / 5)
    ax.set_rticks(radial_ticks)
    ax.set_rlabel_position(-22.5)  # 设置径向标签的位置

    # 绘制数据
    if line_style == 'auto':
        hpol = ax.plot(theta, rho - rmin)  # 在极坐标轴上绘制数据
    else:
        hpol = ax.plot(theta, rho - rmin, line_style)

    # 设置绘图范围
    ax.set_ylim(rmin - (rmax - rmin) * 0.15, rmax + (rmax - rmin) * 0.15)

    plt.show()

    return hpol[0]

'''
# 示例用法
if __name__ == "__main__":
    t = np.linspace(0, 2 * np.pi, 100)
    polar2(t, np.sin(2 * t) * np.cos(2 * t), line_style='--r')
'''