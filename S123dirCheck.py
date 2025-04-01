import numpy as np


def S123dirCheck(S123):
    """
    Coordinate transformation function.
    Inputs:
        S123 = [sigma1 trend1 plunge1 sigma2 trend2 plunge2 sigma3 trend3 plunge3]
    Outputs:
        degS123 = [degS12 degS23 degS13]
    """
    # 提取走向和倾角
    S1tr = S123[1]
    S1pl = S123[2]
    S2tr = S123[4]
    S2pl = S123[5]
    S3tr = S123[7]
    S3pl = S123[8]

    # 将角度转换为弧度
    def deg_to_rad(deg):
        return deg * np.pi / 180.0

    # 计算笛卡尔坐标分量
    z11 = np.cos(deg_to_rad(S1pl)) * np.sin(deg_to_rad(S1tr))
    z21 = np.cos(deg_to_rad(S2pl)) * np.sin(deg_to_rad(S2tr))
    z31 = np.cos(deg_to_rad(S3pl)) * np.sin(deg_to_rad(S3tr))

    z12 = np.cos(deg_to_rad(S1pl)) * np.cos(deg_to_rad(S1tr))
    z22 = np.cos(deg_to_rad(S2pl)) * np.cos(deg_to_rad(S2tr))
    z32 = np.cos(deg_to_rad(S3pl)) * np.cos(deg_to_rad(S3tr))

    z13 = np.sin(deg_to_rad(S1pl))
    z23 = np.sin(deg_to_rad(S2pl))
    z33 = np.sin(deg_to_rad(S3pl))

    # 计算三个应力方向之间的夹角
    degS12 = np.arccos(abs(z11 * z12 + z21 * z22 + z31 * z32)) * 180.0 / np.pi
    degS23 = np.arccos(abs(z12 * z13 + z22 * z23 + z32 * z33)) * 180.0 / np.pi
    degS13 = np.arccos(abs(z11 * z13 + z21 * z23 + z31 * z33)) * 180.0 / np.pi

    # 返回三个夹角
    degS123 = [degS12, degS23, degS13]
    return degS123
'''
# 示例输入
S123 = [1, 30, 45, 2, 60, 30, 3, 90, 15]  # 这里填入具体的数值进行测试
result = S123dirCheck(S123)
print(result)
'''