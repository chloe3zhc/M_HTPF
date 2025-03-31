import numpy as np


def S123dirCheck(S123):
    """
    Coordinate transformation function.
    Inputs:
        S123 = [sigma1 trend1 plunge1 sigma2 trend2 plunge2 sigma3 trend3 plunge3]
    Outputs:
        degS123 = [degS12 degS23 degS13]
    """
    S1tr = S123[1]
    S1pl = S123[2]
    S2tr = S123[4]
    S2pl = S123[5]
    S3tr = S123[7]
    S3pl = S123[8]

    z11 = np.cos(np.deg2rad(S1pl)) * np.sin(np.deg2rad(S1tr))
    z21 = np.cos(np.deg2rad(S2pl)) * np.sin(np.deg2rad(S2tr))
    z31 = np.cos(np.deg2rad(S3pl)) * np.sin(np.deg2rad(S3tr))
    z12 = np.cos(np.deg2rad(S1pl)) * np.cos(np.deg2rad(S1tr))
    z22 = np.cos(np.deg2rad(S2pl)) * np.cos(np.deg2rad(S2tr))
    z32 = np.cos(np.deg2rad(S3pl)) * np.cos(np.deg2rad(S3tr))
    z13 = np.sin(np.deg2rad(S1pl))
    z23 = np.sin(np.deg2rad(S2pl))
    z33 = np.sin(np.deg2rad(S3pl))

    degS12 = np.degrees(np.arccos(np.abs(z11 * z12 + z21 * z22 + z31 * z32)))
    degS23 = np.degrees(np.arccos(np.abs(z12 * z13 + z22 * z23 + z32 * z33)))
    degS13 = np.degrees(np.arccos(np.abs(z11 * z13 + z21 * z23 + z31 * z33)))

    return np.array([degS12, degS23, degS13])