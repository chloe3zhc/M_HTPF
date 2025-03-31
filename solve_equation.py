import numpy as np
from scipy.linalg import lstsq, eig
from numpy.random import rand, normal
from math import atan, asin, acos, sqrt, cos, sin, pi, degrees


def solve_equation(A2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc):
    """
    主计算函数
    参数：
        A2: 测试数据矩阵（二维数组）
        N0: 测试段数
        rg: 岩石重度
        Tr: 抗拉强度范围[Tmin, Tmax]
        Cr: 抗压强度范围[Cmin, Cmax]
        CZWfangweijiao: 参照物方位角（度）
        Nmc: 蒙特卡洛循环次数
    返回：
        sigma_utc, sigma_ucc, sigma_uvc: 三个结果矩阵
    """
    # 计算垂直应力
    sigmaV1 = (A2[np.nonzero(A2[:, 0])[0][0], 0] * 9.8 * rg) / 1000
    sigmaV2 = (np.max(A2[:, 0]) * 9.8 * rg) / 1000

    bet0 = np.deg2rad(CZWfangweijiao)  # 参照物方位角（弧度）
    Dip = np.deg2rad(A2[:, 2])  # 倾角（弧度）
    Strike = np.deg2rad(A2[:, 3])  # 走向（弧度）

    sigma_utc = np.empty((0, 22))  # 预分配内存
    sigma_ucc = np.empty((0, 22))
    sigma_uvc = np.empty((0, 22))

    # 并行循环（这里使用简单循环，实际可改用joblib或multiprocessing）
    for Num in range(Nmc):
        # 随机生成参数
        T0 = Tr[0] + (Tr[1] - Tr[0]) * rand()
        C0 = Cr[0] + (Cr[1] - Cr[0]) * rand()

        # 初始化矩阵
        # 原错误行
        # b1 = np.zeros((3 * (N0 - 1) + 1, 7))
        # 修改后
        b1 = np.zeros((3 * N0, 7))  # 确保每个测试段有3行存储空间

        for i in range(N0):
            # 生成mu值（根据文件名不同有两种分布）
            # solve_equation.m 使用正态分布
            mu_i = normal(0.4, 0.11)
            # solve_equation0.m 使用均匀分布（取消下方注释切换）
            # mu_i = 0.1 + 0.75 * rand()

            al = Dip[i]
            betai = Strike[i]
            bet = bet0 - betai

            k = 3 * i
            # 填充矩阵b1（具体公式与原文一致）
            b1[k, :] = [sin(bet) ** 2 * sin(al) ** 2, cos(bet) ** 2 * sin(al) ** 2, cos(al) ** 2,
                        -sin(2 * bet) * sin(al) ** 2, cos(bet) * sin(2 * al), -sin(bet) * sin(2 * al),
                        A2[i, 1]]

            b1[k + 1, :] = [-0.5 * sin(2 * bet) * sin(al), 0.5 * sin(2 * bet) * sin(al), 0,
                            cos(2 * bet) * sin(al), sin(bet) * cos(al), cos(bet) * cos(al),
                            A2[i, 1] * mu_i]

            b1[k + 2, :] = [0, 0, 0, 0, 0, 0, 0]

        # 构建方程组矩阵
        MATRIX1 = b1[:, :6]
        MATRIX2 = b1[:, 6].reshape(-1, 1)

        # 最小二乘法求解
        result1, _, _, _ = lstsq(MATRIX1, MATRIX2, cond=None)
        result2 = result1.flatten()

        # 构建应力张量矩阵
        ST = np.array([[result2[0], result2[3], result2[5]],
                       [result2[3], result2[1], result2[4]],
                       [result2[5], result2[4], result2[2]]])

        '''
        # 添加调试代码：打印矩阵维度
        print(f"ST矩阵维度: {ST.shape}")
        print(f"ST矩阵内容:\n{ST}")
        '''

        # 特征值分解
        eigenvalues, eigenvectors = eig(ST)
        V = eigenvectors

        # 强制转换为实数（当虚部足够小时）
        if np.max(np.abs(np.imag(eigenvalues))) < 1e-6:
            eigenvalues = np.real(eigenvalues)

        D = np.diag(eigenvalues)
        # print("D:", D)

        # 坐标旋转处理
        Vrot = np.rot90(V, k=1)
        Drot = np.rot90(D, k=2)

        # 计算主应力方向
        bta = np.zeros(3)
        arf = np.zeros(3)
        val1 = np.zeros(3)

        for i in range(3):
            # 计算方位角（具体公式与原文一致）
            bta[i] = degrees(acos(Vrot[i, 0] / sqrt(Vrot[i, 0] ** 2 + Vrot[i, 1] ** 2)))
            if cos(np.deg2rad(bta[i])) * Vrot[i, 0] > 0 and sin(np.deg2rad(bta[i])) * Vrot[i, 1] > 0:
                bta[i] = 90 - bta[i]
            else:
                bta[i] = 270 - bta[i]

            # 计算倾角
            arf[i] = degrees(asin(Vrot[i, 2]))
            if arf[i] < 0:
                arf[i] = -arf[i]
                bta[i] += 180

            # 确保方位角在0-360度之间
            bta[i] %= 360
            val1[i] = Drot[i, i]

        # 计算平面最大最小应力
        SHmax = (result2[0] + result2[1]) / 2 + sqrt(((result2[0] - result2[1]) / 2) ** 2 + result2[3] ** 2)
        Shmin = (result2[0] + result2[1]) / 2 - sqrt(((result2[0] - result2[1]) / 2) ** 2 + result2[3] ** 2)

        # 主应力方向计算（具体公式与原文一致）
        if result2[0] == result2[1]:
            betaSHmax = 45
        else:
            tan2D = -2 * result2[3] / (result2[0] - result2[1])
            twoD1 = degrees(atan(tan2D))
            betaSHmax = twoD1 / 2 if abs(SHmax - ((result2[0] + result2[1]) / 2 -
                                                  (result2[0] - result2[1]) / 2 * cos(np.deg2rad(twoD1)) +
                                                  result2[3] * sin(np.deg2rad(twoD1)))) < \
                                     abs(SHmax - ((result2[0] + result2[1]) / 2 -
                                                  (result2[0] - result2[1]) / 2 * cos(np.deg2rad(twoD1 + 180)) +
                                                  result2[3] * sin(np.deg2rad(twoD1 + 180)))) else twoD1 / 2 + 90
            betaSHmax %= 180

        # 抗拉强度条件判断
        if (val1[0] > val1[1] > val1[2] > 0) and (SHmax - 3 * Shmin < T0):
            temp = np.hstack((result2, val1, arf, bta, SHmax, Shmin, betaSHmax, mu_i))
            sigma_utc = np.vstack((sigma_utc, temp))

        # 抗压强度条件判断
        if (val1[0] > val1[1] > val1[2] > 0) and (3 * SHmax - Shmin < C0):
            temp = np.hstack((result2, val1, arf, bta, SHmax, Shmin, betaSHmax, mu_i))
            sigma_ucc = np.vstack((sigma_ucc, temp))

        # 联合条件判断
        if (val1[0] > val1[1] > val1[2] > 0) and (SHmax - 3 * Shmin < T0) and \
                ((arf[0] < 25 and 0.8 < abs(val1[0]) / sigmaV2 < 1.2) or
                 (abs(arf[1]) > 65 and 0.8 < abs(val1[1]) / sigmaV1 < 1.2) or
                 (abs(arf[2]) > 65 and 0.8 < abs(val1[2]) / sigmaV1 < 1.2)):
            temp = np.hstack((result2, val1, arf, bta, SHmax, Shmin, betaSHmax, mu_i))
            sigma_uvc = np.vstack((sigma_uvc, temp))

    return sigma_utc, sigma_ucc, sigma_uvc

'''
# 使用示例
if __name__ == "__main__":
    # 生成示例输入数据（需根据实际需求调整）
    A2 = np.array([[100, 50, 30, 45],
                   [200, 60, 40, 60],
                   [150, 55, 35, 50]])
    N0 = 2
    rg = 2.7  # 岩石重度示例值
    Tr = [5, 10]  # 抗拉强度范围
    Cr = [100, 200]  # 抗压强度范围
    CZWfangweijiao = 45  # 参照物方位角
    Nmc = 100  # 蒙特卡洛次数

    # 调用函数
    utc, ucc, uvc = solve_equation(A2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc)

    # 打印结果摘要
    print(f"UTC结果形状: {utc.shape}")
    print(f"UCC结果形状: {ucc.shape}")
    print(f"UVC结果形状: {uvc.shape}")
    print("\n前5行UTC结果示例:")
    print(utc[:5])
'''