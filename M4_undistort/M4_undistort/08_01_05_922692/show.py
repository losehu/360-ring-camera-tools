import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取XML文件并提取相机参数
def read_camera_params(xml_file):
    fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    distortion_coeffs = fs.getNode("distortion_coefficients").mat()
    extrinsic_params = fs.getNode("extrinsic_parameters").mat()
    fs.release()
    return camera_matrix, distortion_coeffs, extrinsic_params

# 将鱼眼图像投影到经纬度坐标系
def fisheye_to_latlon(image, K, D):
    h, w = image.shape[:2]
    lon_range = np.linspace(-180, 180, w)
    lat_range = np.linspace(-90, 90, h)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    # 将经纬度坐标转换为球面坐标
    r = np.sqrt(lon_grid**2 + lat_grid**2)  # 简单的球面模型
    spherical_coords = np.stack([r, lon_grid, lat_grid], axis=-1)

    return spherical_coords

# 渲染投影图像到球面
def render_spherical_projection(images, camera_matrices, distortion_coeffs, extrinsic_params):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, image in enumerate(images):
        K = camera_matrices[i]
        D = distortion_coeffs[i]
        
        # 将图像投影到经纬度
        spherical_coords = fisheye_to_latlon(image, K, D)

        # 假设我们在此渲染球面，这里只是示意
        ax.plot_surface(spherical_coords[..., 1], spherical_coords[..., 2], spherical_coords[..., 0], rstride=1, cstride=1, alpha=0.6)

    plt.show()

# 读取4个XML文件
xml_files = ['camera1.xml', 'camera2.xml', 'camera3.xml', 'camera4.xml']
images = [cv2.imread(f'fisheye_{i}.jpg') for i in range(1, 5)]  # 读取4张鱼眼图像

camera_matrices = []
distortion_coeffs = []
extrinsic_params = []

# 读取每个XML文件的相机参数
for xml_file in xml_files:
    K, D, extrinsic = read_camera_params(xml_file)
    camera_matrices.append(K)
    distortion_coeffs.append(D)
    extrinsic_params.append(extrinsic)

# 渲染4张图像到球面
render_spherical_projection(images, camera_matrices, distortion_coeffs, extrinsic_params)
