import os
import numpy as np
import pandas as pd
from confidence_interval import calculate_confidence_interval
from solve_equation import solve_equation


def Components2Principle(stress_tensor):
    """将应力张量转换为主应力和方向（需实现特征值分解）"""
    # 实现MATLAB的Components2Principle功能
    eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)
    sort_idx = np.argsort(eigenvalues)[::-1]  # 降序排列
    sigma123 = eigenvalues[sort_idx]
    degS123 = np.degrees(np.arctan2(eigenvectors[1, sort_idx], eigenvectors[0, sort_idx])) % 180
    return sigma123, degS123


# 主程序
def main():
    # 创建文件夹
    os.makedirs('./DataMC_SLZK', exist_ok=True)

    # 读取CSV数据
    DB = pd.read_csv('SLZK_5_cracks.csv', header=None).values

    N0 = DB.shape[0]  # 测试段数
    Zu = 1  # 分组数
    rg = 2.65
    Tr = [1.0, 20]
    Cr = [20, 200]
    CZWfangweijiao = 0
    Nmc = 100000

    sigma_utc5 = np.empty((0, 15))
    sigma_ucc5 = np.empty((0, 15))
    sigma_uvc5 = np.empty((0, 15))

    for i in range(Zu):
        g1 = i * N0
        g2 = (i + 1) * N0
        B2 = DB[g1:g2, :]

        sigma_utc, sigma_ucc, sigma_uvc = solve_equation(
            B2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc
        )

        sigma_utc5 = np.vstack((sigma_utc5, sigma_utc))
        sigma_ucc5 = np.vstack((sigma_ucc5, sigma_ucc))
        sigma_uvc5 = np.vstack((sigma_uvc5, sigma_uvc))

    # 计算平均应力张量
    SE_mean_utc5 = np.array([
        [np.mean(sigma_utc5[:, 0]), np.mean(sigma_utc5[:, 3]), np.mean(sigma_utc5[:, 5])],
        [np.mean(sigma_utc5[:, 3]), np.mean(sigma_utc5[:, 1]), np.mean(sigma_utc5[:, 4])],
        [np.mean(sigma_utc5[:, 5]), np.mean(sigma_utc5[:, 4]), np.mean(sigma_utc5[:, 2])]
    ])
    sigma123_Emean_utc5, degS123_Emean_utc5 = Components2Principle(SE_mean_utc5)

    # 对sigma_uvc5进行角度修正
    for i in range(len(sigma_uvc5[:, 8])):
        if sigma_uvc5[i, 8] < 180:
            sigma_uvc5[i, 8] += 180
        if sigma_uvc5[i, 11] < 210:
            sigma_uvc5[i, 11] += 180

    # 计算置信区间（示例，实际需循环所有列）
    mean_utc5 = np.zeros((15, 1))
    std_utc5 = np.zeros((15, 1))
    CIf_utc5 = np.zeros((2, 15))
    CIa_utc5 = np.zeros((2, 15))

    for i in range(15):
        meana, stda, CIf, CIa = calculate_confidence_interval(sigma_utc5[:, i], 0.1, method='mc')
        mean_utc5[i] = meana
        std_utc5[i] = stda
        CIf_utc5[:, i] = CIf
        CIa_utc5[:, i] = CIa

    # 保存结果
    np.savez('./DataMC_SLZK/result_utc5_v2.npz',
             sigma_utc5=sigma_utc5,
             SE_mean_utc5=SE_mean_utc5,
             sigma123_Emean_utc5=sigma123_Emean_utc5,
             mean_utc5=mean_utc5,
             std_utc5=std_utc5,
             CIf_utc5=CIf_utc5,
             CIa_utc5=CIa_utc5,
             DB=DB,
             N0=N0,
             Nmc=Nmc,
             Tr=Tr,
             Cr=Cr,
             CZWfangweijiao=CZWfangweijiao)

    # 类似地保存sigma_ucc5和sigma_uvc5的结果...


if __name__ == "__main__":
    main()
