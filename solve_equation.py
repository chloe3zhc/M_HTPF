import numpy as np
from scipy.linalg import eig
from numpy.linalg import pinv


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
    # print(f"输入参数-------A2:{A2}, N0:{N0}, rg:{rg}, Tr:{Tr}, Cr:{Cr}, CZW:{CZWfangweijiao}, Nmc:{Nmc}")

    # 初始化结果列表
    sigma_utc = [0]
    sigma_ucc = [0]
    sigma_uvc = [0]

    # 参照物方位角（弧度）
    bet0 = CZWfangweijiao * np.pi / 180

    # 岩石重度和抗拉、抗压强度
    sigmaV1 = (np.min(A2[A2[:, 0] != 0, 0]) * 9.8 * rg) / 1000
    sigmaV2 = (np.max(A2[:, 0]) * 9.8 * rg) / 1000
    # print(f"sigmaV1={sigmaV1}, sigmaV2={sigmaV2}")

    parfor = range(Nmc)  # Python 中没有直接的 parfor，可以用多线程或 multiprocessing 替代
    # print("parfor =", parfor)

    for Num in parfor:
        # print(f"-------------------------------Num:{Num}---------------------------------")
        # 随机生成抗拉和抗压强度
        T0 = Tr[0] + (Tr[1] - Tr[0]) * np.random.rand()
        C0 = Cr[0] + (Cr[1] - Cr[0]) * np.random.rand()

        b1 = np.zeros((3 * (N0 - 1) + 1, 7))
        mu = np.random.normal(0.4, 0.11, N0)

        # print(f"T0:{T0}, C0:{C0}, b1:{b1}, mu:{mu}")

        for i in range(N0):
            # print(f"---------------------i={i}----------------------")
            al = A2[i, 2] * np.pi / 180  # 倾角
            betai = A2[i, 3] * np.pi / 180  # 方位角
            bet = bet0 - betai
            # print(f"----------al={al}, betai={betai}, bet0={bet0}, bet={bet}")

            k = 3 * (i - 1)
            b1[k, 0] = np.sin(bet) ** 2 * np.sin(al) ** 2
            b1[k, 1] = np.cos(bet) ** 2 * np.sin(al) ** 2
            b1[k, 2] = np.cos(al) ** 2
            b1[k, 3] = -np.sin(2 * bet) * np.sin(al) ** 2
            b1[k, 4] = np.cos(bet) * np.sin(2 * al)
            b1[k, 5] = -np.sin(bet) * np.sin(2 * al)
            b1[k, 6] = A2[i, 1]
            # print(f"----------k={k}, b1[k]={b1[k]}")

            k += 1
            b1[k, 0] = -0.5 * np.sin(2 * bet) * np.sin(al)
            b1[k, 1] = 0.5 * np.sin(2 * bet) * np.sin(al)
            b1[k, 2] = 0
            b1[k, 3] = np.cos(2 * bet) * np.sin(al)
            b1[k, 4] = np.sin(bet) * np.cos(al)
            b1[k, 5] = np.cos(bet) * np.cos(al)
            b1[k, 6] = A2[i, 1] * mu[i]
            # print(f"----------k={k}, b1[k]={b1[k]}")

            k += 1
            b1[k, :] = 0
            # print(f"----------k={k}, b1[k]={b1[k]}")

        # 矩阵求解
        MATRIX1 = b1[:, :-1]
        MATRIX2 = b1[:, -1].reshape(-1, 1)
        result1 = pinv(MATRIX1) @ MATRIX2
        result2 = result1.flatten()
        # print(f"MATRIX1={MATRIX1}")
        # print(f"MATRIX2={MATRIX2}")
        # print(f"result1={result1}")
        # print(f"result2={result2}")

        ST = np.array([
            [result2[0], result2[3], result2[5]],
            [result2[3], result2[1], result2[4]],
            [result2[5], result2[4], result2[2]]
        ])
        # print(ST)

        eigvals, eigvecs = eig(ST)
        Vrot = np.rot90(eigvecs)
        Drot = np.rot90(np.diag(eigvals.real), k=2)
        # print(f"eigvals={eigvals}, eigvecs={eigvecs}, Vrot={Vrot}, Drot={Drot}")

        bta = np.zeros(3)
        arf = np.zeros(3)
        val1 = np.zeros(3)

        for i in range(3):
            bta[i] = np.degrees(np.arctan2(Vrot[i, 1], Vrot[i, 0]))
            if Vrot[i, 0] * np.cos(np.deg2rad(bta[i])) + Vrot[i, 1] * np.sin(np.deg2rad(bta[i])) < 0:
                bta[i] += 180

            arf[i] = np.degrees(np.arcsin(Vrot[i, 2]))
            if Vrot[i, 2] < 0:
                arf[i] = -arf[i]
                bta[i] += 180

            val1[i] = Drot[i, i].real
            # print(f"val1[{i}]={val1[i]}")
        # print(f"val1={val1}")

        # 计算最大最小主应力
        SHmax = (result2[0] + result2[1]) / 2 + np.sqrt(((result2[0] - result2[1]) / 2) ** 2 + result2[3] ** 2)
        Shmin = (result2[0] + result2[1]) / 2 - np.sqrt(((result2[0] - result2[1]) / 2) ** 2 + result2[3] ** 2)
        # print(f"SHmax={SHmax}, SHmin={Shmin}")

        if result2[0] == result2[1]:
            twoD1 = 90
            twoD2 = 270
        else:
            tan2D = -2 * result2[3] / (result2[0] - result2[1])
            twoD1 = np.degrees(np.arctan(tan2D))
            twoD2 = twoD1 + 180

        D2D1 = (result2[0] + result2[1]) / 2 - (result2[0] - result2[1]) / 2 * np.cos(np.deg2rad(twoD1)) + result2[
            3] * np.sin(np.deg2rad(twoD1))
        D2D2 = (result2[0] + result2[1]) / 2 - (result2[0] - result2[1]) / 2 * np.cos(np.deg2rad(twoD2)) + result2[
            3] * np.sin(np.deg2rad(twoD2))

        if abs(SHmax - D2D1) < abs(SHmax - D2D2):
            betaSHmax = twoD1 / 2
        else:
            betaSHmax = twoD2 / 2

        if betaSHmax < 0:
            betaSHmax += 180
        elif betaSHmax > 180:
            betaSHmax -= 180

        # 应力判别
        sigma1, sigma2, sigma3 = sorted(val1, reverse=True)
        Alph1, Alph2, Alph3 = sorted(arf, reverse=True)
        Beta1, Beta2, Beta3 = sorted(bta, reverse=True)
        # print(f"sigma1={sigma1}, sigma2={sigma2}, sigma3={sigma3}")
        # print(f"Alph1={Alph1}, Alph2={Alph2}, Alph3={Alph3}")
        # print(f"Beta1={Beta1}, Beta2={Beta2}, Beta3={Beta3}")

        conditions_met = (
                (sigma1 > sigma2 > sigma3 > 0) and
                (SHmax - 3 * Shmin < T0) and
                ((Alph1 < 25 and (abs(sigma1) / sigmaV2 > 0.8 and abs(sigma1) / sigmaV2 < 1.2)) or
                 (abs(Alph2) > 65 and (abs(sigma2) / sigmaV1 > 0.8 and abs(sigma2) / sigmaV1 < 1.2)) or
                 (abs(Alph3) > 65 and (abs(sigma3) / sigmaV1 > 0.8 and abs(sigma3) / sigmaV1 < 1.2)))
        )
        # print(f"conditions_met={conditions_met}")

        if conditions_met:
            temp = np.concatenate(([result2],
                                   [sigma1, Alph1, Beta1, sigma2, Alph2, Beta2, sigma3, Alph3, Beta3, SHmax, Shmin,
                                    betaSHmax, mu]))
            sigma_utc.append(temp)
        # print(f"sigma_utc={sigma_utc}")
    return sigma_utc, sigma_ucc, sigma_uvc


'''
# 示例用法
# A2, N0, rg, Tr, Cr, CZWfangweijiao, N
if __name__ == "__main__":
    # 生成示例输入数据（需根据实际需求调整）
    A2 = np.array([[150, 60, 45, 60], [200, 70, 50, 75]])
    N0 = 2
    rg = 2.7  # 岩石重度示例值
    Tr = [5, 10]  # 抗拉强度范围
    Cr = [100, 200]  # 抗压强度范围
    CZWfangweijiao = 45  # 参照物方位角
    Nmc = 100  # 蒙特卡洛次数

    # 调用函数
    utc, ucc, uvc = solve_equation(A2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc)

    # 打印结果摘要
    print(f"UTC结果: {utc}")
    print(f"UCC结果: {ucc}")
    print(f"UVC结果: {uvc}")
    print("\n前5行UTC结果示例:")
    print(utc[:5])
'''