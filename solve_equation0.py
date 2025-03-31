import numpy as np
from scipy.linalg import eig


def solve_equation(A2, N0, rg, Tr, Cr, CZWfangweijiao, Nmc):
    """
    Solve stress tensor using least squares method (alternative version).
    Inputs:
        A2: Test results
        N0: Number of test intervals
        rg: Rock density
        Tr: Tensile strength range
        Cr: Compressive strength range
        CZWfangweijiao: Reference azimuth
        Nmc: Number of Monte Carlo iterations
    Outputs:
        sigma_utc, sigma_ucc, sigma_uvc: Stress solutions
    """
    sigmaV1 = (A2[np.where(A2[:, 0] != 0)[0][0], 0] * 9.8 * rg) / 1000
    sigmaV2 = (np.max(A2[:, 0]) * 9.8 * rg) / 1000

    bet0 = np.deg2rad(CZWfangweijiao)

    sigma_utc = []
    sigma_ucc = []
    sigma_uvc = []

    Dip = np.deg2rad(A2[:, 2])
    Strike = np.deg2rad(A2[:, 3])

    for _ in range(Nmc):
        T0 = np.random.uniform(Tr[0], Tr[1])
        C0 = np.random.uniform(Cr[0], Cr[1])

        b1 = np.zeros((3 * N0, 7))

        for i in range(N0):
            mu = np.random.normal(0.5, 0.1)
            al = Dip[i]
            betai = Strike[i]
            bet = bet0 - betai

            k = 3 * i
            b1[k, :] = [np.sin(bet) ** 2 * np.sin(al) ** 2,
                        np.cos(bet) ** 2 * np.sin(al) ** 2,
                        np.cos(al) ** 2,
                        -np.sin(2 * bet) * np.sin(al) ** 2,
                        np.cos(bet) * np.sin(2 * al),
                        -np.sin(bet) * np.sin(2 * al),
                        A2[i, 1]]

            b1[k + 1, :] = [-0.5 * np.sin(2 * bet) * np.sin(al),
                            0.5 * np.sin(2 * bet) * np.sin(al),
                            0,
                            np.cos(2 * bet) * np.sin(al),
                            np.sin(bet) * np.cos(al),
                            np.cos(bet) * np.cos(al),
                            A2[i, 1] * mu]

            b1[k + 2, :] = [0, 0, 0, 0, 0, 0, 0]

        MATRIX1 = b1[:, :6]
        MATRIX2 = b1[:, 6].reshape(-1, 1)

        result1 = np.linalg.lstsq(MATRIX1, MATRIX2, rcond=None)[0]
        ST = np.array([[result1[0], result1[3], result1[5]],
                       [result1[3], result1[1], result1[4]],
                       [result1[5], result1[4], result1[2]]])

        eigvals, eigvecs = eig(ST)
        V = eigvecs
        D = np.diag(eigvals)

        Vrot = np.rot90(V, axes=(1, 0))
        Drot = np.rot90(D, k=2, axes=(1, 0))

        bta = np.zeros(3)
        arf = np.zeros(3)
        val1 = np.zeros(3)

        for i in range(3):
            bta[i] = np.rad2deg(np.arctan2(Vrot[i, 1], Vrot[i, 0]))
            if np.cos(np.deg2rad(bta[i])) * Vrot[i, 0] > 0 and np.sin(np.deg2rad(bta[i])) * Vrot[i, 1] > 0:
                bta[i] = 90 - bta[i]
            else:
                bta[i] = 270 - bta[i]

            arf[i] = np.rad2deg(np.arcsin(Vrot[i, 2]))
            if arf[i] < 0:
                arf[i] = -arf[i]
                bta[i] += 180

            bta[i] %= 360
            val1[i] = Drot[i, i]

        Smax = (result1[0] + result1[1]) / 2 + np.sqrt(((result1[0] - result1[1]) / 2) ** 2 + result1[3] ** 2)
        Smin = (result1[0] + result1[1]) / 2 - np.sqrt(((result1[0] - result1[1]) / 2) ** 2 + result1[3] ** 2)

        # Stress criteria checks
        if (val1[0] > val1[1] > val1[2] > 0 and Smax - 3 * Smin < T0):
            temp = np.concatenate([result1, val1, arf, bta, [Smax, Smin, betaSHmax, mu]])
            sigma_utc.append(temp)

        if (val1[0] > val1[1] > val1[2] > 0 and 3 * Smax - Smin < C0):
            temp = np.concatenate([result1, val1, arf, bta, [Smax, Smin, betaSHmax, mu]])
            sigma_ucc.append(temp)

        if (val1[0] > val1[1] > val1[2] > 0 and Smax - 3 * Smin < T0 and
                ((arf[0] < 25 and 0.8 < abs(val1[0]) / sigmaV2 < 1.2) or
                 (abs(arf[1]) > 65 and 0.8 < abs(val1[1]) / sigmaV1 < 1.2))):
            temp = np.concatenate([result1, val1, arf, bta, [Smax, Smin, betaSHmax, mu]])
            sigma_uvc.append(temp)

    return np.array(sigma_utc), np.array(sigma_ucc), np.array(sigma_uvc)