import cv2

# 棋盘格内角点数（每行 m、每列 n）
m, n = 7, 11
pattern_size = (m, n)

# 读取图像
img = cv2.imread("3.jpg")
if img is None:
    raise FileNotFoundError("找不到图像文件 a111.bmp")

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("chessboard corners1", gray)
cv2.waitKey(0)
# 检测角点
# flags 可按需增减：自适应阈值 + 归一化能提升鲁棒性
ret, corners = cv2.findChessboardCorners(
    gray, pattern_size,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

# （可选）亚像素精细化，效果更稳
if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# 绘制并显示
cv2.drawChessboardCorners(img, pattern_size, corners, ret)
cv2.imshow("chessboard corners", img)
cv2.waitKey(0)
