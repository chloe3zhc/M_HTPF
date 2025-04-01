import numpy as np


# 假设sigma_uvc是输入的二维数组（类似MATLAB矩阵）
# sampledata = sigma_uvc[:, 0]  # 取第一列数据（根据实际数据结构可能需要调整）

def calculate_confidence_interval(sampledata, a, method):
    """
    计算置信区间
    参数：
        sampledata: 输入样本数据（一维数组）
        alpha: 显著性水平（默认0.05对应95%置信区间）
        method: 计算方法，'z-score'或'mc'（蒙特卡洛）
    返回：
        包含上下限的numpy数组
    """
    # 方法1：基于Z分数的置信区间（正态分布假设）
    if method == 'z-score':
        # 确定Z值（根据alpha选择对应临界值）
        if a == 0.01:
            z = 2.576
        elif a == 0.05:
            z = 1.96
        elif a == 0.1:
            z = 1.645
        else:
            # 使用erfinv函数计算任意alpha对应的临界值
            from scipy.special import erfinv
            z = np.sqrt(2) * erfinv(1 - a)

        mean_val = np.mean(sampledata)
        std_val = np.std(sampledata, ddof=1)  # ddof=1对应MATLAB的样本标准差计算
        ci = np.array([mean_val - z * std_val, mean_val + z * std_val])

    # 方法2：蒙特卡洛方法（基于排序的百分位数）
    elif method == 'mc':
        sorted_data = np.sort(sampledata)
        n = len(sorted_data)
        lower_idx = int(np.round(n * a / 2)) - 1  # Python索引从0开始
        upper_idx = int(np.round(n * (1 - a / 2))) - 1
        ci = np.array([sorted_data[lower_idx], sorted_data[upper_idx]])

    else:
        raise ValueError("Invalid method. Choose 'z-score' or 'mc'")

    return ci

'''
# 使用示例
if __name__ == "__main__":
    # 生成示例数据（替换为实际数据）
    np.random.seed(42)
    sample_data = np.random.normal(loc=50, scale=10, size=100)  # 正态分布样本

    # 计算95%置信区间（两种方法）
    ci_zscore = calculate_confidence_interval(sample_data, a=0.05, method='z-score')
    ci_mc = calculate_confidence_interval(sample_data, a=0.05, method='mc')

    print(f"Z-score方法 95%置信区间: [{ci_zscore[0]:.2f}, {ci_zscore[1]:.2f}]")
    print(f"蒙特卡洛方法 95%置信区间: [{ci_mc[0]:.2f}, {ci_mc[1]:.2f}]")
'''