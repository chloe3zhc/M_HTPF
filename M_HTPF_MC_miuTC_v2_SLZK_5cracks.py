import numpy as np
import pandas as pd
import os
from solve_equation import solve_equation
import confidence_interval

def M_HTPF_MC():
    # 初始化变量
    DB = pd.read_csv('HTPF.csv')
    N0 = len(DB)  # 测试段数
    Zu = 1  # 分组数
    rg = 2.65  # 岩石重力
    Tr = [1.0, 20]  # 抗拉强度范围
    Cr = [20, 200]  # 抗压强度范围
    CZWfangweijiao = 0  # 参照物方位角
    Nmc = 100000  # 随机计算循环次数
    sigma_utc5 = []
    sigma_ucc5 = []
    sigma_uvc5 = []

    # 创建数据目录
    if not os.path.exists('./DataMC_SLZK/'):
        os.makedirs('./DataMC_SLZK/')

    # 分组处理数据
    group_size = N0 // Zu
    for i in range(Zu):
        start = i * group_size
        end = start + group_size if i < Zu - 1 else N0  # 处理余数
        B2 = DB.iloc[start:end, :]

        if B2.empty:
            print(f"警告：第 {i} 组数据为空，跳过计算")
            continue

        A2 = B2.values  # 转换为NumPy数组
        [sigma_utc, sigma_ucc, sigma_uvc] = solve_equation(A2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc)
        sigma_utc5.extend(sigma_utc)
        sigma_ucc5.extend(sigma_ucc)
        sigma_uvc5.extend(sigma_uvc)

    # 计算均值应力张量
    SE_mean_utc5 = np.array([
        [np.mean(sigma_utc5[:, 0]), np.mean(sigma_utc5[:, 3]), np.mean(sigma_utc5[:, 5])],
        [np.mean(sigma_utc5[:, 3]), np.mean(sigma_utc5[:, 1]), np.mean(sigma_utc5[:, 4])],
        [np.mean(sigma_utc5[:, 5]), np.mean(sigma_utc5[:, 4]), np.mean(sigma_utc5[:, 2])]
    ])
    # 假设Components2Principle函数已在其他地方定义
    [sigma123_Emean_utc5, degS123_Emean_utc5] = Components2Principle(SE_mean_utc5)

    SE_mean_ucc5 = np.array([
        [np.mean(sigma_ucc5[:, 0]), np.mean(sigma_ucc5[:, 3]), np.mean(sigma_ucc5[:, 5])],
        [np.mean(sigma_ucc5[:, 3]), np.mean(sigma_ucc5[:, 1]), np.mean(sigma_ucc5[:, 4])],
        [np.mean(sigma_ucc5[:, 5]), np.mean(sigma_ucc5[:, 4]), np.mean(sigma_ucc5[:, 2])]
    ])
    [sigma123_Emean_ucc5, degS123_Emean_ucc5] = Components2Principle(SE_mean_ucc5)

    SE_mean_uvc5 = np.array([
        [np.mean(sigma_uvc5[:, 0]), np.mean(sigma_uvc5[:, 3]), np.mean(sigma_uvc5[:, 5])],
        [np.mean(sigma_uvc5[:, 3]), np.mean(sigma_uvc5[:, 1]), np.mean(sigma_uvc5[:, 4])],
        [np.mean(sigma_uvc5[:, 5]), np.mean(sigma_uvc5[:, 4]), np.mean(sigma_uvc5[:, 2])]
    ])
    [sigma123_Emean_uvc5, degS123_Emean_uvc5] = Components2Principle(SE_mean_uvc5)

    # 调整角度
    for i in range(len(sigma_uvc5)):
        if sigma_uvc5[i, 8] > 180:
            sigma_uvc5[i, 8] = sigma_uvc5[i, 8]
        else:
            sigma_uvc5[i, 8] = sigma_uvc5[i, 8] + 180
        if sigma_uvc5[i, 11] > 210:
            sigma_uvc5[i, 11] = sigma_uvc5[i, 11]
        else:
            sigma_uvc5[i, 11] = sigma_uvc5[i, 11] + 180

    # 计算置信区间
    mean_utc5 = np.zeros((3, 18 + N0))
    std_utc5 = np.zeros((3, 18 + N0))
    CIf_utc5 = np.zeros((3, 18 + N0))
    CIa_utc5 = np.zeros((3, 18 + N0))
    mean_ucc5 = np.zeros((3, 18 + N0))
    std_ucc5 = np.zeros((3, 18 + N0))
    CIf_ucc5 = np.zeros((3, 18 + N0))
    CIa_ucc5 = np.zeros((3, 18 + N0))
    mean_uvc5 = np.zeros((3, 18 + N0))
    std_uvc5 = np.zeros((3, 18 + N0))
    CIf_uvc5 = np.zeros((3, 18 + N0))
    CIa_uvc5 = np.zeros((3, 18 + N0))

    for i in range(18 + N0):
        [mean_utc5[:, i], std_utc5[:, i], CIf_utc5[:, i], CIa_utc5[:, i]] = confidence_interval.confidence_interval(sigma_utc5[:, i], 0.1)
        [mean_ucc5[:, i], std_ucc5[:, i], CIf_ucc5[:, i], CIa_ucc5[:, i]] = confidence_interval.confidence_interval(sigma_ucc5[:, i], 0.1)
        [mean_uvc5[:, i], std_uvc5[:, i], CIf_uvc5[:, i], CIa_uvc5[:, i]] = confidence_interval.confidence_interval(sigma_uvc5[:, i], 0.2)

    # 保存结果
    np.savez('./DataMC_SLZK/result_utc5_v2', sigma_utc5=sigma_utc5, SE_mean_utc5=SE_mean_utc5,
             sigma123_Emean_utc5=sigma123_Emean_utc5, mean_utc5=mean_utc5, std_utc5=std_utc5,
             CIf_utc5=CIf_utc5, CIa_utc5=CIa_utc5, DB=DB.values, N0=N0, Nmc=Nmc, Tr=Tr, Cr=Cr,
             CZWfangweijiao=CZWfangweijiao)

    np.savez('./DataMC_SLZK/result_ucc5_v2', sigma_ucc5=sigma_ucc5, SE_mean_ucc5=SE_mean_ucc5,
             sigma123_Emean_ucc5=sigma123_Emean_ucc5, mean_ucc5=mean_ucc5, std_ucc5=std_ucc5,
             CIf_ucc5=CIf_ucc5, CIa_ucc5=CIa_ucc5, DB=DB.values, N0=N0, Nmc=Nmc, Tr=Tr, Cr=Cr,
             CZWfangweijiao=CZWfangweijiao)

    np.savez('./DataMC_SLZK/result_uvc5_v2', sigma_uvc5=sigma_uvc5, SE_mean_uvc5=SE_mean_uvc5,
             sigma123_Emean_uvc5=sigma123_Emean_uvc5, mean_uvc5=mean_uvc5, std_uvc5=std_uvc5,
             CIf_utc5=CIf_utc5, CIa_uvc5=CIa_uvc5, DB=DB.values, N0=N0, Nmc=Nmc, Tr=Tr, Cr=Cr,
             CZWfangweijiao=CZWfangweijiao)

# 假设Components2Principle函数在其他地方定义
def Components2Principle(stress_tensor):
    # 将应力张量从应力分量转换为主应力和主方向
    eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sigma123 = eigenvalues[sorted_indices]
    degS123 = np.degrees(np.arctan2(eigenvectors[1, sorted_indices], eigenvectors[0, sorted_indices]))
    return sigma123, degS123


if __name__ == "__main__":
    M_HTPF_MC()
