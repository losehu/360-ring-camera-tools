import cv2
import numpy as np

zoom_factor = 1.0
zoom_speed = 0.05
move_step = 20  # 平移步长
center_x = 0
center_y = 0
ZOOM_MIN = 1.0
ZOOM_MAX = 10.0

def auto_white_balance_gray_world(image):
    image = image.astype(np.float32)
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    image[:, :, 0] = np.clip(image[:, :, 0] * (avg_gray / (avg_b + 1e-5)), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * (avg_gray / (avg_g + 1e-5)), 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * (avg_gray / (avg_r + 1e-5)), 0, 255)
    return image.astype(np.uint8)

def capture_industrial_camera():
    global zoom_factor, center_x, center_y

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("无法打开相机")
        return

    # 设置全屏窗口
    cv2.namedWindow('White Balanced', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('White Balanced', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        return

    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # balanced_frame = auto_white_balance_gray_world(frame)
            balanced_frame = frame
            # 计算缩放后裁剪区域
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            x1 = center_x - new_w // 2
            y1 = center_y - new_h // 2
            x2 = x1 + new_w
            y2 = y1 + new_h

            # 边界限制
            if x1 < 0:
                x1 = 0
                x2 = new_w
            if y1 < 0:
                y1 = 0
                y2 = new_h
            if x2 > w:
                x2 = w
                x1 = w - new_w
            if y2 > h:
                y2 = h
                y1 = h - new_h

            roi = balanced_frame[y1:y2, x1:x2]
            if roi.size > 0:
                balanced_frame = cv2.resize(roi, (w, h))

            cv2.imshow('White Balanced', balanced_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('w'):  # 上
                center_y -= move_step
            elif key == ord('s'):  # 下
                center_y += move_step
            elif key == ord('a'):  # 左
                center_x -= move_step
            elif key == ord('d'):  # 右
                center_x += move_step
            elif key == ord('h'):  # 放大
                zoom_factor = min(zoom_factor + zoom_speed, ZOOM_MAX)
            elif key == ord('j'):  # 缩小
                zoom_factor = max(zoom_factor - zoom_speed, ZOOM_MIN)
            elif key == ord('r'):  # 恢复原图
                zoom_factor = 1.0
                center_x = w // 2
                center_y = h // 2

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_industrial_camera()
