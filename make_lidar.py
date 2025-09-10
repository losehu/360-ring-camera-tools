# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import math

""" 本程序为相机展开所使用到的工具函数，主程序位于ocam_unfold.py中 """

CMV_MAX_BUF = 1024
MAX_POL_LENGTH = 64
import numpy as np

def transform_3d_point(point_3d, rotation_matrix, translation_vector):
    """
    对3D点进行旋转和平移变换
    
    Args:
        point_3d: 输入3D点，形状为(3,)或(3,1)的numpy数组
        rotation_matrix: 旋转矩阵，形状为(3,3)的numpy数组
        translation_vector: 平移向量，形状为(3,)或(3,1)的numpy数组
    
    Returns:
        numpy.ndarray: 变换后的3D点，形状为(3,)
    """
    # 确保输入是numpy数组
    # 
    # point_3d=np.array([-585, 656, -475])# [ 654.  685. -476.]
    point_3d = np.array(point_3d, dtype=np.float64).flatten()
    translation_vector = np.array(translation_vector, dtype=np.float64).flatten()
    
    # 验证形状
    if point_3d.shape != (3,):
        raise ValueError(f"point_3d的形状应为(3,)，但得到{point_3d.shape}")
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix的形状应为(3,3)，但得到{rotation_matrix.shape}")
    if translation_vector.shape != (3,):
        raise ValueError(f"translation_vector的形状应为(3,)，但得到{translation_vector.shape}")
    transformed_point = point_3d - translation_vector

    # 应用变换：先旋转，后平移
    rotated_point = rotation_matrix.T @ transformed_point



    # exit(0)
    return rotated_point


class ocam_model:
    """  ocam_model  """

    def __init__(self):
        # 初始化ocam_model类的各个参数
        self.pol = []  # 存储多项式系数
        self.length_pol = 0  # 多项式系数的长度
        self.invpol = []  # 存储逆多项式系数
        self.length_invpol = 0  # 逆多项式系数的长度
        self.xc = 0.0  # 相机的中心点横坐标
        self.yc = 0.0  # 相机的中心点纵坐标
        self.c = 0.0  # 仿射变换参数
        self.d = 0.0  # 仿射变换参数
        self.e = 0.0  # 仿射变换参数
        self.width = 0  # 图像的宽度
        self.height = 0  # 图像的高度

def get_ocam_model(myocam_model, filename):
    try:
        with open(filename) as f:
            # Read polynomial coefficients
            f.readline()
            f.readline()
            str_poly = f.readline()
            poly = str_poly.split(" ")

            myocam_model.length_pol = int(poly[0])
            myocam_model.pol = [float(pp) for pp in poly[1:-1]]
            # Read inverse polynomial coefficients
            f.readline()
            f.readline()
            f.readline()
            str_invpoly = f.readline()
            invpoly = str_invpoly.split(" ")
            myocam_model.length_invpol = int(invpoly[0])
            myocam_model.invpol = [float(pp) for pp in invpoly[1:-1]]
            # Read center coordinates
            f.readline()
            f.readline()
            f.readline()
            str_cent = f.readline()
            cent = str_cent.split(" ")
            myocam_model.xc = float(cent[0])
            myocam_model.yc = float(cent[1])
            # print("Center: ({}, {})".format(myocam_model.xc, myocam_model.yc))
            # Read affine coefficients
            f.readline()
            f.readline()
            f.readline()
            str_aff = f.readline()
            aff = str_aff.split(" ")
            myocam_model.c = float(aff[0])
            myocam_model.d = float(aff[1])
            myocam_model.e = float(aff[2])
            # Read image size
            f.readline()
            f.readline()
            f.readline()
            str_size = f.readline()
            size = str_size.split(" ")
            myocam_model.height = float(size[0])
            myocam_model.width = float(size[1])
            # print(poly)

    except Exception as e:
        print(e)
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    If `p` is of length N, this function returns the value:

        ``p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]``

    If `x` is a sequence, then ``p(x)`` is returned for each element of ``x``.
    If `x` is another polynomial then the composite polynomial ``p(x(t))``
    is returned.

    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.

    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.

    If `x` is a subtype of `ndarray` the return value will be of the same type.

    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.

    Examples
    --------
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([76])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([76])

    """

    y = 0
    i=0
    for pv in p:
        y = y * x + pv
        i+=1
    return y


def cam2world(point3D, point2D, myocam_model):
    xc = myocam_model.xc
    yc = myocam_model.yc
    c = myocam_model.c
    d = myocam_model.d
    e = myocam_model.e
    length_pol = myocam_model.length_pol
    invdet = 1/(c-d*e)  # 1/det(A), where A = [c,d;e,1] as in the Matlab file
    xp = 1/(c-d*e)*((point2D[0] - xc) - d*(point2D[1] - yc))
    yp = 1/(c-d*e)*(-e*(point2D[0] - xc) + c*(point2D[1] - yc))
    #根据x，y轴像素变形程度，计算世界的x，y坐标系

    r = math.sqrt(xp*xp + yp*yp)  # distance of the point from image center
    
    zp = myocam_model.pol[0]
    r_i = 1
    for i in range(1,length_pol):
        r_i *= r
        zp += r_i*myocam_model.pol[i]
    #根据半径计算多项式得到距离（鱼眼模型特性）
    # normalize to unit norm
    invnorm = 1/math.sqrt(xp*xp + yp*yp + zp*zp)

    point3D[0] = invnorm*xp
    point3D[1] = invnorm*yp
    point3D[2] = invnorm*zp




def world2cam(point2D, point3D, myocam_model):
    xc = myocam_model.xc
    yc = myocam_model.yc
    c = myocam_model.c
    d = myocam_model.d
    e = myocam_model.e
    length_invpol = myocam_model.length_invpol
    #3D点在2D平面的投影长度
    norm = math.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1])
    #3D点与平面的夹角
    theta = math.atan(point3D[2]/norm)
    if norm != 0:
        invnorm = 1/norm
        t = theta
        rho = myocam_model.invpol[0]
        rho=polyval(myocam_model.invpol, t)

        # for i in range(1,length_invpol):
        #     t_i *= t
        #     rho += t_i*myocam_model.invpol[i]
        #rho是畸变半径
        x = point3D[0]*invnorm*rho
        y = point3D[1]*invnorm*rho
        point2D[0] = x*c + y*d + xc
        point2D[1] = x*e + y + yc
    else:
        point2D[0] = xc
        point2D[1] = yc


def create_perspective_undistortion_LUT(mapx, mapy, ocam_model, sf):
    """ Create Look Up Table for undistorting the image into a perspective one
     It assumes the final image plane is perpendicular to the camera axis """
    width = mapx.cols
    height = mapx.rows
    Nxc = height/2.0
    Nyc = width/2.0
    Nz = -width/sf

    M = np.zeros(3)
    m = np.zeros(2)

    for i in range(height):
        for j in range(width):
            M[0] = i - Nxc
            M[1] = j - Nyc
            M[2] = Nz
            world2cam(m, M, ocam_model)
            mapx[i*width+j] = m[1]
            mapy[i*width+j] = m[0]


def create_panoramic_undistortion_LUT(mapx, mapy, Rmin, Rmax, xc, yc):
    """  Create Look Up Table for undistorting the image into a panoramic image
    It computes a trasformation from cartesian to polar coordinates
    Therefore it does not need the calibration parameters
    The region to undistorted in contained between Rmin and Rmax
    xc, yc are the row and column coordinates of the image center """
    width = mapx.shape[0]
    height = mapx.shape[1]

    xnums = np.arange(height, dtype=np.float32)
    ynums = np.arange(width, dtype=np.float32)
    mapx, mapy = np.meshgrid(ynums, xnums)
    theta = -mapx/width*2*3.1416+3.1416
    rho = Rmax - (Rmax-Rmin)/height*mapy
    mapx = yc + rho*np.sin(theta)
    mapy = xc + rho*np.cos(theta)

    return mapx, mapy


def create_panoramic_CUBE(mapx, mapy, myocam_model,angle):
    width = mapx.shape[1]
    height = mapx.shape[0]

    size_x1 = math.tan(angle[0]/180*math.pi)
    size_x2 = math.tan(angle[1]/180*math.pi)
    size_y = math.tan(angle[2]/180*math.pi)+math.tan(angle[3]/180*math.pi)
    i = 0
    while i < height:
        j = 0
        while j < width:
            point3 = np.array([1, size_y*(1-2*j/width), -(size_x1-i*(size_x1+size_x2)/height)])
            point2 = np.array([1808, 1250])
            world2cam(point2, point3, myocam_model)

            mapx[i][j] = point2[1]
            mapy[i][j] = point2[0]
            j = j+1
        i = i+1

    return mapx, mapy


def create_panoramic_lonlat(mapx, mapy, myocam_model,angle):
    width = mapx.shape[1]
    height = mapx.shape[0]
    i = 0
    while i < height:
        j = 0
        while j < width:
            lat = (-angle[0]/180+i/height*(angle[0]+angle[1])/180) * np.pi
            lon = -(-angle[2]/180 + j/width*(angle[2]+angle[3])/180) * np.pi + np.pi
            point3 = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
            point2 = np.array([0, 0])
            point3D_1= np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
            world2cam(point2, point3, myocam_model)
            # print(point2)
            # cam2world(point3D_1,point2, myocam_model);
            # print()
            # print(point3D_1)
            # print(point3)
            # print()
            mapx[i][j] = point2[1]
            mapy[i][j] = point2[0]
            j = j+1
        i = i+1

    return mapx, mapy


def create_panoramic_CUBE_plus():
    pass
    return 0


def ocam_unfold(img):

    size_pan_img = [1200, 300]

    # cv2中的mat都变成了ndarray，所以直接创建ndarray就行
    mapx_pan = np.zeros([size_pan_img[0], size_pan_img[1], 1], np.float32)
    mapy_pan = np.zeros([size_pan_img[0], size_pan_img[1], 1], np.float32)
    Rmax = 600.78
    Rmin = 25.949
    mapx_pan, mapy_pan = create_panoramic_undistortion_LUT(mapx_pan, mapy_pan,
                                                           Rmin, Rmax,
                                                           o_cata.xc,
                                                           o_cata.yc)

    dst_pan = cv2.remap(img, mapx_pan, mapy_pan, cv2.INTER_LINEAR)
    dst_pan = cv2.flip(dst_pan, -1)
    return dst_pan


def get_lonlat_map(result_txt, fov, angle):
    o_cata = ocam_model()
    get_ocam_model(o_cata, result_txt)
    pixel_h = o_cata.height * (fov[1] - fov[0]) / 2 / fov[1]
    size_pan_img = [round(pixel_h*(angle[0]+angle[1])/(fov[1]-fov[0])),
                    round(pixel_h*(angle[2]+angle[3])/(fov[1]-fov[0]))]
    mapx_pan = np.zeros([size_pan_img[0], size_pan_img[1], 1], np.float32)
    mapy_pan = np.zeros([size_pan_img[0], size_pan_img[1], 1], np.float32)

    mapx_pan, mapy_pan = create_panoramic_lonlat(mapx_pan, mapy_pan, o_cata, angle)
    return mapx_pan, mapy_pan


def ocam_lonlat(img, mapx_pan, mapy_pan):
    dst_pan = cv2.remap(img, mapx_pan, mapy_pan, cv2.INTER_LINEAR)
    dst_pan = cv2.flip(dst_pan, 1)
    return dst_pan

def get_line_n(filename, n):
    """
    获取txt文件的第n行（内存效率更高）
    
    Args:
        filename: 文件名
        n: 行号（从1开始）
    
    Returns:
        str: 第n行的内容，如果行号超出范围返回None
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if i == n:
                    return line.strip()
            return None
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
import os

def reset_or_create_file(filename):
    """
    检查txt文件是否存在，存在则清空内容，不存在则创建
    
    Args:
        filename: 文件名
    
    Returns:
        bool: 操作成功返回True，失败返回False
    """
    try:
        if os.path.exists(filename):
            # 文件存在，清空内容
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('')  # 写入空字符串来清空内容
            print(f"已清空文件: {filename}")
        else:
            # 文件不存在，创建空文件
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('')  # 创建空文件
            print(f"已创建文件: {filename}")
        return True
    except Exception as e:
        print(f"操作失败: {e}")
        return False

# 使用示例
def parse_two_numbers(input_string):
    """
    解析包含两个数字的字符串，返回两个数字
    
    Args:
        input_string: 字符串，格式如 "1533                1633"
    
    Returns:
        tuple: 包含两个整数的元组 (num1, num2)
    
    Raises:
        ValueError: 如果字符串格式不正确或无法解析为两个数字
    """
    # 去除首尾空白字符
    cleaned_string = input_string.strip()
    
    # 使用空白字符分割字符串
    parts = cleaned_string.split()
    
    # 检查是否正好有两个部分
    if len(parts) != 2:
        raise ValueError(f"字符串应该包含两个数字，但找到 {len(parts)} 个部分: {parts}")
    
    try:
        num1 = int(parts[0])
        num2 = int(parts[1])
        return num1, num2
    except ValueError:
        raise ValueError(f"无法将部分转换为数字: {parts[0]}, {parts[1]}")
def append_line_to_file(filename, line_content, encoding='utf-8'):
    """
    向文件末尾追加一行内容
    
    Args:
        filename: 文件名
        line_content: 要追加的内容（字符串）
        encoding: 文件编码，默认为utf-8
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        with open(filename, 'a', encoding=encoding) as file:
            file.write(line_content + '\n')  # 自动添加换行符
        return True
    except Exception as e:
        print(f"追加文件时出错: {e}")
        return False
import numpy as np
import random

def generate_random_transform():
    """
    随机生成一个旋转矩阵和平移向量
    
    返回:
        rotation_matrix: 3x3 旋转矩阵
        translation_vector: 3维平移向量
    """
    # 随机生成三个欧拉角（弧度制）
    alpha = random.uniform(0, 2 * np.pi)  # 绕z轴旋转
    beta = random.uniform(0, 2 * np.pi)   # 绕y轴旋转
    gamma = random.uniform(0, 2 * np.pi)  # 绕x轴旋转
    
    # 计算旋转矩阵（使用ZYX欧拉角顺序）
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma), np.cos(gamma)]])
    
    # 组合旋转矩阵
    rotation_matrix = Rz @ Ry @ Rx
    
    # 随机生成平移向量（范围可根据需要调整）
    translation_vector = np.array([
        random.uniform(-100, 100),
        random.uniform(-100, 100),
        random.uniform(-100, 100)
    ])
    
    return rotation_matrix, translation_vector

# 使用示例
if __name__ == '__main__':


    rotation_matrix, translation_vector = generate_random_transform()
    
    print("旋转矩阵:")
    print(rotation_matrix)
    print("\n平移向量:")
    print(translation_vector)
    


    point3D_3 = np.array([0.0, 0.0, 0.0])
    rotation_matrix = rotation_matrix
    translation_vector = translation_vector


    result_txt = "a.txt"
    pic_point="calib/cam_point.txt"
    out_txt="fake_lidar.txt"
    o_cata = ocam_model()
    get_ocam_model(o_cata, result_txt)
    reset_or_create_file(out_txt)
    for i in range(20):
        pic_name=get_line_n(pic_point,1+i*9)
        point1_x,point1_y=parse_two_numbers(get_line_n(pic_point,3+i*9))
        point2_x, point2_y = parse_two_numbers(get_line_n(pic_point,5+i*9))
        point3_x,point3_y=parse_two_numbers(get_line_n(pic_point,7+i*9))
        point4_x,point4_y=parse_two_numbers(get_line_n(pic_point,9+i*9))
        point1=np.array([point1_x, point1_y])
        point2 = np.array([point2_x, point2_y])
        point3 = np.array([point3_x, point3_y])
        point4 = np.array([point4_x, point4_y])

        # cam2world
        point3D_1 = np.array([0.0,0.0,0.0])
        point3D_2 = np.array([0.0,0.0,0.0])
        point3D_3 = np.array([0.0,0.0,0.0])
        point3D_4 = np.array([0.0,0.0,0.0])

        cam2world(point3D_1,point1, o_cata)
        cam2world(point3D_2,point2, o_cata)
        cam2world(point3D_3,point3, o_cata)
        cam2world(point3D_4,point4, o_cata)
        point3D_1=point3D_1*1000
        point3D_2=point3D_2*1000
        point3D_3=point3D_3*1000
        point3D_4=point3D_4*1000
        point3D_1 = transform_3d_point(point3D_1, rotation_matrix, translation_vector)
        point3D_2 = transform_3d_point(point3D_2, rotation_matrix, translation_vector)
        point3D_3 = transform_3d_point(point3D_3, rotation_matrix, translation_vector)
        point3D_4 = transform_3d_point(point3D_4, rotation_matrix, translation_vector)

        append_line_to_file(out_txt, f"lidar{i+1}")
        append_line_to_file(out_txt, "1")
        append_line_to_file(out_txt, f"{(point3D_1[0])} {(point3D_1[1])} {(point3D_1[2])}")
        append_line_to_file(out_txt, "2")
        append_line_to_file(out_txt, f"{(point3D_2[0])} {(point3D_2[1])} {(point3D_2[2])}")
        append_line_to_file(out_txt, "3")
        append_line_to_file(out_txt, f"{(point3D_3[0])} {(point3D_3[1])} {(point3D_3[2])}")
        append_line_to_file(out_txt, "4")
        append_line_to_file(out_txt, f"{(point3D_4[0])} {(point3D_4[1])} {(point3D_4[2])}")
        # append_line_to_file(out_txt, "1")
        # append_line_to_file(out_txt, f"{(int)(point3D_1[0])} {int(point3D_1[1])} {int(point3D_1[2])}")
        # append_line_to_file(out_txt, "2")
        # append_line_to_file(out_txt, f"{(int)(point3D_2[0])} {int(point3D_2[1])} {int(point3D_2[2])}")
        # append_line_to_file(out_txt, "3")
        # append_line_to_file(out_txt, f"{(int)(point3D_3[0])} {int(point3D_3[1])} {int(point3D_3[2])}")
        # append_line_to_file(out_txt, "4")
        # append_line_to_file(out_txt, f"{(int)(point3D_4[0])} {int(point3D_4[1])} {int(point3D_4[2])}")

