from math import pi, sqrt
import numpy as np

# 注：Hamilton为右手系
#    JPL     为左手系

def quat_to_pos_matrix_hm(p_x, p_y, p_z, x, y, z, w):
    """
        右手系
        四元数转RT
    """
    T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)

    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z - w * x)

    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T


def quat_to_pos_matrix_JPL(p_x, p_y, p_z, x, y, z, w):
    """
        左手系
        四元数转RT
    """
    T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y + w * z)
    T[0, 2] = 2 * (x * z - w * y)

    T[1, 0] = 2 * (x * y - w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z + w * x)

    T[2, 0] = 2 * (x * z + w * y)
    T[2, 1] = 2 * (y * z - w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T
    # return T.tolist()


def pos_matrix_to_quat_hm(T):
    """
        右手系
        RT转四元数
    """
    r11 = T[0, 0]
    r12 = T[0, 1]
    r13 = T[0, 2]
    r21 = T[1, 0]
    r22 = T[1, 1]
    r23 = T[1, 2]
    r31 = T[2, 0]
    r32 = T[2, 1]
    r33 = T[2, 2]
    tx = T[0, 3]
    ty = T[0, 3]
    tz = T[0, 3]
    w = (1 / 2) * sqrt(1 + r11 + r22 + r33)
    x = (r32 - r23) / (4 * w)
    y = (r13 - r31) / (4 * w)
    z = (r21 - r12) / (4 * w)
    return tx,ty,tz,x, y, z, w


def pos_matrix_to_quat_JPL(T):
    """
          左手系
          RT转四元数
      """
    r11 = T[0, 0]
    r12 = T[0, 1]
    r13 = T[0, 2]
    r21 = T[1, 0]
    r22 = T[1, 1]
    r23 = T[1, 2]
    r31 = T[2, 0]
    r32 = T[2, 1]
    r33 = T[2, 2]
    tx  = T[0, 3]
    ty  = T[0, 3]
    tz  = T[0, 3]
    w = (1 / 2) * sqrt(1 + r11 + r22 + r33)
    x = (r23 - r32) / (4 * w)
    y = (r31 - r13) / (4 * w)
    z = (r12 - r21) / (4 * w)
    return tx,ty,tz,x, y, z, w


if __name__ == "__main__":
    # https://blog.csdn.net/gyxx1998/article/details/119636130
    # ----------------------------------------------------------------------------------
    # tx ty tz (3 个浮点数) 给出彩色摄像机的光学中心相对于运动捕捉系统定义的世界原点的位置。
    # qx qy qz qw（4 个浮点数）以单位四元数的形式给出彩色相机的光学中心相对于运动捕捉系统定义的世界原点的方向。
    # ----------------------------------------------------------------------------------
    arr=[-0.00473952,- 0.00107861,- 2.25009,0.000447757,- 4.2262e-05,0.000114325,1]
    tx,ty,tz,qx,qy,qz,qw=arr
    T_hm = quat_to_pos_matrix_hm(tx, ty, tz, qx, qy, qz, qw)
    print("T_hm",T_hm)
    T_JPL = quat_to_pos_matrix_JPL(tx, ty, tz, qx, qy, qz, qw)
    print("T_JPL",T_JPL)

    tx, ty, tz, qx, qy, qz, qw = pos_matrix_to_quat_hm(T_hm)
    print("q_hm")
    print(tx, ty, tz, qx, qy, qz, qw)

    tx, ty, tz, qx, qy, qz, qw  = pos_matrix_to_quat_JPL(T_JPL)
    print("q_JPL")
    print(tx, ty, tz, qx, qy, qz, qw)


