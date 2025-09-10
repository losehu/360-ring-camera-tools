#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, time, argparse

BACKEND = cv2.CAP_AVFOUNDATION  # macOS 推荐
COMMON_RES = [
    (640, 480), (960, 540), (1280, 720),
    (1600, 900), (1920, 1080), (2560, 1440),
    (3840, 2160)
]
FOURCC_TRY = ["MJPG", "YUY2", "AVC1"]  # 依次尝试更稳

def open_cam(index):
    cap = cv2.VideoCapture(index, BACKEND)
    return cap if cap.isOpened() else None

def warmup(cap, timeout=2.0):
    """开流预热，直到连续读取到帧或超时"""
    t0 = time.time()
    ok_cnt = 0
    while time.time() - t0 < timeout:
        ok, _ = cap.read()
        if ok:
            ok_cnt += 1
            if ok_cnt >= 3:  # 连续几帧才算稳定
                return True
        else:
            ok_cnt = 0
        time.sleep(0.01)
    return False

def try_fourcc(cap):
    """尝试不同像素格式以提升稳定性/帧率"""
    for cc in FOURCC_TRY:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # 有些机型需要给一点反应时间
        if warmup(cap, 0.8):
            return cc
    return None

def set_props_robust(cap, w=None, h=None, fps=None):
    """更稳的设置顺序：先不设 FPS → 设分辨率 → 预热 → 再尝试 FPS（可选）"""
    # 1) 尝试合适的 FourCC
    chosen = try_fourcc(cap)
    # 2) 仅设置分辨率
    if w: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
    if h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))

    # 分辨率变更后要清缓存并预热
    if not warmup(cap, 1.2):
        return False, chosen

    # 3) FPS 很多设备会忽略；若用户真的指定，尝试但失败也不视为致命
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        warmup(cap, 0.8)

    return True, chosen

def get_dev_props(cap):
    rw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rfp = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
    return rw, rh, rfp, cc

def measure_fps(cap, seconds=2.0):
    start = time.time()
    frames = 0
    while time.time() - start < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    dur = max(1e-6, time.time() - start)
    return frames / dur

def probe_modes(index, fps_req=None, secs=0.7):
    cap = open_cam(index)
    if cap is None:
        print("无法打开摄像头进行探测"); return
    print("开始探测常见分辨率（≈表示设备回报的实际值）:")
    try_fourcc(cap)
    for w, h in COMMON_RES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        if fps_req: cap.set(cv2.CAP_PROP_FPS, float(fps_req))
        if not warmup(cap, 0.8):
            print(f"- 请求 {w}x{h}: 打开失败")
            continue
        rw, rh, rfp, cc = get_dev_props(cap)
        mfps = measure_fps(cap, secs)
        print(f"- 请求 {w}x{h}"
              f"{f', FPS={fps_req}' if fps_req else ''}"
              f" -> 设备≈{rw}x{rh}, DevFPS≈{rfp:.1f}, 测得FPS≈{mfps:.1f}, FourCC={cc}")
    cap.release()

def main():
    ap = argparse.ArgumentParser(description="macOS OpenCV 外置相机：分辨率+帧率设置与实测（稳健版）")
    ap.add_argument("--index", type=int, default=0, help="摄像头索引")
    ap.add_argument("-W", "--width",  type=int, default=1280, help="请求宽度")
    ap.add_argument("-H", "--height", type=int, default=720,  help="请求高度")
    ap.add_argument("--fps", type=int, default=0, help="请求FPS（默认0表示不强制）")
    ap.add_argument("--report_every", type=float, default=2.0, help="多少秒打印一次实测FPS")
    ap.add_argument("--probe", action="store_true", help="只探测常见分辨率并退出")
    args = ap.parse_args()

    if args.probe:
        probe_modes(args.index, args.fps if args.fps>0 else None)
        return

    cap = open_cam(args.index)
    if cap is None:
        print("无法打开摄像头（检查权限/索引）"); return

    # 初次预热，避免“首帧读取失败”
    if not warmup(cap, 1.0):
        print("设备预热失败"); cap.release(); return

    ok, chosen_cc = set_props_robust(cap, args.width, args.height, args.fps)
    if not ok:
        print("设置后首帧读取失败（已包含重试）"); cap.release(); return

    rw, rh, rfp, cc = get_dev_props(cap)
    print(f"请求: {args.width}x{args.height}, 请求FPS: {args.fps or '不强制'}")
    print(f"设备回报: {rw}x{rh}, DevFPS: {rfp:.2f}, FourCC: {cc}（尝试得到: {chosen_cc}）")

    win_start = time.time()
    win_frames = 0
    measured = 0.0

    res_cycle = COMMON_RES
    cur_res_i = max(0, res_cycle.index((args.width, args.height))
                       if (args.width, args.height) in res_cycle else 0)

    print("按键：q 退出 | r 切换分辨率（在常见列表中轮换）")
    while True:
        ok, frame = cap.read()
        if not ok:
            # 尝试一次轻量恢复：短暂停顿 + 预热
            time.sleep(0.1)
            if not warmup(cap, 0.8):
                print("读帧失败（恢复失败）"); break
            continue

        win_frames += 1
        now = time.time()
        if now - win_start >= args.report_every:
            measured = win_frames / (now - win_start)
            rw, rh, rfp, cc = get_dev_props(cap)
            print(f"[{time.strftime('%H:%M:%S')}] 回报: {rw}x{rh}, DevFPS: {rfp:.1f}, 实测FPS: {measured:.2f}, FourCC: {cc}")
            win_start = now; win_frames = 0

        text = f"{rw}x{rh} | ReqFPS {args.fps or 0} | Dev {rfp:.1f} | Meas {measured:.1f} | {cc}"
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Camera (AVFoundation)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            cur_res_i = (cur_res_i + 1) % len(res_cycle)
            nw, nh = res_cycle[cur_res_i]
            print(f"切换分辨率 -> 请求 {nw}x{nh}")
            # 切换分辨率后记得预热
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(nw))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(nh))
            warmup(cap, 0.8)
            rw, rh, rfp, cc = get_dev_props(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
