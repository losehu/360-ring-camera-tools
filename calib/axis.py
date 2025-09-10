import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === 颜色设定：X=蓝, Y=红, Z=绿 ===
COLOR_X = "blue"
COLOR_Y = "red"
COLOR_Z = "green"

def invert_w2c(T):
    """将 world->camera 外参 (X_c = R X_w + t) 取逆，得到 camera->world 位姿 Twc。"""
    R = T[:3, :3]
    t = T[:3, 3]
    Twc = np.eye(4)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = -R.T @ t
    return Twc

def plot_frame(ax, T, name="frame", axis_len=300.0, use_quiver=True):
    """
    画一个坐标系：给定 camera->world 位姿 T(4x4)。
    X=蓝, Y=红, Z=绿；可用 quiver 画箭头，或用 plot 画线段。
    """
    R = T[:3, :3]
    t = T[:3, 3]

    if use_quiver:
        # 用箭头表示三个轴
        ax.quiver(t[0], t[1], t[2], *(R @ np.array([axis_len, 0, 0])), color=COLOR_X, linewidth=2)
        ax.quiver(t[0], t[1], t[2], *(R @ np.array([0, axis_len, 0])), color=COLOR_Y, linewidth=2)
        ax.quiver(t[0], t[1], t[2], *(R @ np.array([0, 0, axis_len])), color=COLOR_Z, linewidth=2)
    else:
        # 仅线段（没有箭头）
        ex = t + R @ np.array([axis_len, 0.0, 0.0])
        ey = t + R @ np.array([0.0, axis_len, 0.0])
        ez = t + R @ np.array([0.0, 0.0, axis_len])
        ax.plot([t[0], ex[0]], [t[1], ex[1]], [t[2], ex[2]], color=COLOR_X, linewidth=2)
        ax.plot([t[0], ey[0]], [t[1], ey[1]], [t[2], ey[2]], color=COLOR_Y, linewidth=2)
        ax.plot([t[0], ez[0]], [t[1], ez[1]], [t[2], ez[2]], color=COLOR_Z, linewidth=2)

    ax.scatter([t[0]], [t[1]], [t[2]], s=30)
    ax.text(t[0], t[1], t[2], f" {name}", fontsize=9)

def set_equal_3d(ax, pts, margin=0.2):
    """让 3D 轴均匀缩放，避免失真。pts: (N,3)"""
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    x_rng, y_rng, z_rng = x_max - x_min, y_max - y_min, z_max - z_min
    max_rng = max(x_rng, y_rng, z_rng, 1.0)
    pad = max_rng * margin
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    ax.set_xlim(x_mid - max_rng/2 - pad, x_mid + max_rng/2 + pad)
    ax.set_ylim(y_mid - max_rng/2 - pad, y_mid + max_rng/2 + pad)
    ax.set_zlim(z_mid - max_rng/2 - pad, z_mid + max_rng/2 + pad)

if __name__ == "__main__":
    # ===== 在此粘贴你的外参（4x4） =====
    T1 = np.array([
# [0.999799 , -0.019251 , 0.00560416,  113.747],
# [0.00633393 , 0.0380586,  -0.999255,  -49.4812],
# [0.0190234 , 0.99909 , 0.0381729  ,566.309],
# [0,  0 , 0  ,1],
[0.980777  ,-0.166422  ,0.101885  ,766.093],
[-0.0850568  ,-0.834527  ,-0.544363  ,3584.3],
[0.17562  ,0.525232  ,-0.83264  ,-1603.95],
[0  ,0  ,0  ,1],



    ], dtype=float)




    # 第二个外参（示例用单位阵，替换为你的矩阵）
    T2 = np.eye(4, dtype=float)

    # 如果上面是 world->camera（常见），保持 True；若本来是 camera->world，改为 False
    extrinsics_are_world_to_cam = True
    # 相机坐标系原点（即相机位置）
    camera_pos = T1[:3, 3]

    print("相机位置 (世界坐标系):", camera_pos)
    # 想画更多相机，继续往这里加
    T_list = [("Cam1", T1), ("Cam2", T2)]

    # ===== 生成 camera->world 位姿 =====
    poses = []
    for name, T in T_list:
        if T.shape != (4, 4):
            raise ValueError(f"{name} 不是 4x4 矩阵，实际形状={T.shape}")
        Twc = invert_w2c(T) if extrinsics_are_world_to_cam else T.copy()
        poses.append((name, Twc))

    # ===== 画图 =====
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 世界坐标系（也按同样的配色）
    plot_frame(ax, np.eye(4), name="World", axis_len=400.0, use_quiver=True)

    for name, Twc in poses:
        plot_frame(ax, Twc, name=name, axis_len=300.0, use_quiver=True)

    # 等比例
    pts = [np.zeros(3)] + [Twc[:3, 3] for _, Twc in poses]
    set_equal_3d(ax, np.vstack(pts), margin=0.25)

    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")
    ax.set_title("Extrinsic Coordinate Frames (X=Blue, Y=Red, Z=Green)")
    plt.tight_layout()
    plt.show()
