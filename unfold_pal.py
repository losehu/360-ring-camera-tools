# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import math

""" 本程序为相机展开所使用到的工具函数，主程序位于ocam_unfold.py中 """

CMV_MAX_BUF = 1024
MAX_POL_LENGTH = 64


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
        print("i:",i,"y:", y)
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
    print(xp, yp)
    r = math.sqrt(xp*xp + yp*yp)  # distance of the point from image center
    
    zp = myocam_model.pol[0]
    r_i = 1
    for i in range(1,length_pol):
        r_i *= r
        zp += r_i*myocam_model.pol[i]
    print(xp, yp,zp)
    #根据半径计算多项式得到距离（鱼眼模型特性）
    # normalize to unit norm
    invnorm = 1/math.sqrt(xp*xp + yp*yp + zp*zp)
    point3D[0] = invnorm*xp
    point3D[1] = invnorm*yp
    point3D[2] = invnorm*zp
    print(invnorm,point3D)
    exit(0)

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
    print(norm, theta)
    if norm != 0:
        invnorm = 1/norm
        t = theta
        rho = myocam_model.invpol[0]
        rho=polyval(myocam_model.invpol, t)
        print(myocam_model.invpol)
        print(rho)

        # for i in range(1,length_invpol):
        #     t_i *= t
        #     rho += t_i*myocam_model.invpol[i]
        #rho是畸变半径
        x = point3D[0]*invnorm*rho
        y = point3D[1]*invnorm*rho
        point2D[0] = x*c + y*d + xc
        point2D[1] = x*e + y + yc
        print(point2D)
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
            point3D_1= np.array([0,0,0])
            world2cam(point2, point3, myocam_model)
            # print(point2)
            cam2world(point3D_1,point2, myocam_model);
            print(point3D_1)
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


if __name__ == '__main__':

    point33 = np.array([-0.28837 , -0.452965, 0.843603])

#            [-0.28837 ; 0, 1.83362, -0.552446, 0, 1, 0, 0]
#                                                                [-0.452965 ; -3.02666, -0.407817, -0.407817, -0.640589, 0, 1, 0]
#                                                                [0.843603 ; -0.088143, 0.407817, -0.407817, 1.19303, 0, 0, 1]
# world2cam1OUT:                                                                             [159.741 ; 5271.53, 3883.66, -161.147, 735.109, 1802.55, -1732.55, -314.11]
#                                                                              [-91.9385 ; -513.268, -3452.18, 1083.04, -706.371, -1732.08, 183.949, -493.31]
    # point34 = point33 / np.linalg.norm(point33)
    point2 = np.array([1517, 1650])
    o_cata = ocam_model()
    get_ocam_model(o_cata, "/Users/losehu/Downloads/Pano_Video/calib/intrinsic_pano.txt")
    print(point33)
    # world2cam(point2, point33, o_cata)
    print(point2)
    point33[0]=0

    cam2world(point33,point2,o_cata)
    print(point33)

    # world2cam(point2, point31, o_cata)
    # print(point2)
    # world2cam(point2, point32, o_cata)
    # print(point2)
    # world2cam(point2, point33, o_cata)
    # print(point2)
    exit(0)
    """ 本程序用于接受全景相机图片，进行展开，并返回展开结果 """

    result_txt = "/Users/losehu/Downloads/Pano_Video/a.txt"
    img_name = 'image4.jpg'
    img = cv2.imread(img_name)      # 2592 * 1944


    angle = [30, 10, 180, 180]    # upper fov bound, lower fov bound, right fov bound, left fov bound

    mapx, mapy = get_lonlat_map(result_txt, [0, 90], angle)       # 40, 120 = 486 * 2916 ; 10, 50 = 1166 * 6998 ; 0, 90 = 648 * 3888

    raw_img = ocam_lonlat(img, mapx, mapy)

    cv2.imshow('unfold.jpg', raw_img)
    cv2.waitKey(0)

