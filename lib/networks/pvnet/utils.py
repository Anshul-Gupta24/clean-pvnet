import torch

def quaternion2rotation(quat):
    assert (quat.shape[1] == 4)
    # normalize first
    quat = quat / quat.norm(p=2, dim=1).view(-1, 1)

    a = quat[:, 0]
    b = quat[:, 1]
    c = quat[:, 2]
    d = quat[:, 3]

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return torch.stack((m0, m1, m2, m3, m4, m5, m6, m7, m8), dim=1).view(-1, 3, 3)