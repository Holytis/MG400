import cv2
import numpy as np
import json
import os
import time

# ====== CONFIG ======
DEFAULT_HSV_RANGES = [
    {"name": "Yellow", "lower": (3,   137, 131), "upper": (47,  226, 220)},
    {"name": "Red",    "lower": (150, 147, 137), "upper": (226, 255, 196)},
    {"name": "Blue",   "lower": (87,   77,  63), "upper": (150, 250, 255)},
    {"name": "Green",  "lower": (36,   68, 114), "upper": (74,  219, 184)},
]
DRAW_COLORS = [(0,255,255), (0,0,255), (255,0,0), (0,255,0)]
MIN_AREA  = 200
BOX_TYPE  = 0
THICKNESS = 2

CONFIG_PATH = "hsv_config.json"

# ====== Perspective Calibration (Placeholder) ======
camera_points = np.float32([[585, 314], [243, 321], [563, 98], [239, 131]])
world_points = np.float32([[368.39, 93.74], [371.16, -29.02], [292.13, 83.77], [302.78, -29.97]])
matrix = cv2.getPerspectiveTransform(camera_points, world_points)

def to_pos_robot(box1):
    camera_x, camera_y, r = box1[0], box1[1], box1[2]
    camera_coord = np.float32([[camera_x, camera_y]])
    robot_coord = cv2.perspectiveTransform(camera_coord.reshape(-1, 1, 2), matrix)
    robotx, roboty = robot_coord[0][0][0], robot_coord[0][0][1]
    return robotx, roboty, 90 - r

def compute_distortion_map(w, h, k):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    max_rad = np.sqrt(cx**2 + cy**2)
    x = np.linspace(0, w - 1, w, dtype=np.float32)
    y = np.linspace(0, h - 1, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Xc, Yc = X - cx, Y - cy
    xn, yn = Xc / max_rad, Yc / max_rad
    r2 = xn*xn + yn*yn
    scale = 1.0 + k * r2
    Xd = (xn * scale) * max_rad + cx
    Yd = (yn * scale) * max_rad + cy
    return Xd.astype(np.float32), Yd.astype(np.float32)

def ensure_odd(n): return max(1, n | 1)
def nothing(_): pass

def load_config_or_default():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "ranges": data.get("ranges", DEFAULT_HSV_RANGES),
            "distortion_kx100": int(data.get("distortion_kx100", 100)),
            "morph_op": int(data.get("morph_op", 0)),
            "kernel_size": int(data.get("kernel_size", 3)),
            "iterations": int(data.get("iterations", 1)),
        }
    else:
        return {
            "ranges": DEFAULT_HSV_RANGES,
            "distortion_kx100": 100,
            "morph_op": 0,
            "kernel_size": 3,
            "iterations": 1,
        }

def save_config(ranges, kx100, morph_op, ksize, iters):
    data = {
        "ranges": ranges,
        "distortion_kx100": int(kx100),
        "morph_op": int(morph_op),
        "kernel_size": int(ksize),
        "iterations": int(iters),
    }
    for r in data["ranges"]:
        r["lower"] = list(r["lower"])
        r["upper"] = list(r["upper"])
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved settings to {CONFIG_PATH}")

def create_controls_windows(cfg):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 420) # เพิ่มความสูงเล็กน้อย
    cv2.createTrackbar("Distortion k x100", "Controls", cfg["distortion_kx100"], 200, nothing)
    cv2.createTrackbar("Morph Op (0-4)", "Controls", cfg["morph_op"], 4, nothing)
    cv2.createTrackbar("Kernel Size", "Controls", cfg["kernel_size"], 31, nothing)
    cv2.createTrackbar("Iterations", "Controls", cfg["iterations"], 10, nothing)
    
    ranges = cfg.get("ranges", DEFAULT_HSV_RANGES)
    for i, spec in enumerate(ranges):
        name = spec.get("name", f"Color{i}")
        track_name = f"Enable {name}"
        cv2.createTrackbar(track_name, "Controls", 1, 1, nothing)
        
    # --- STEP 1: เพิ่ม Trackbar 'Start' ---
    cv2.createTrackbar("Start Processing", "Controls", 0, 1, nothing)

def read_controls(cfg):
    kx100 = cv2.getTrackbarPos("Distortion k x100", "Controls")
    morph = cv2.getTrackbarPos("Morph Op (0-4)", "Controls")
    ksize = ensure_odd(cv2.getTrackbarPos("Kernel Size", "Controls"))
    iters = cv2.getTrackbarPos("Iterations", "Controls")
    
    enabled = []
    ranges = cfg.get("ranges", DEFAULT_HSV_RANGES)
    for i, spec in enumerate(ranges):
        name = spec.get("name", f"Color{i}")
        track_name = f"Enable {name}"
        try:
            val = cv2.getTrackbarPos(track_name, "Controls")
        except cv2.error:
            val = 1
        enabled.append(bool(val))
    return kx100, morph, ksize, iters, enabled

# ====== Vision Processing Loop ======
def vision_processing_loop(cfg):
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    last_w, last_h, last_k = -1, -1, None
    map_x, map_y = None, None
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # --- STEP 2: เพิ่มเงื่อนไข Start/Stop ---
        start_val = cv2.getTrackbarPos("Start Processing", "Controls")

        # ถ้า Trackbar อยู่ที่ 0 (Stop)
        if start_val == 0:
            # สร้างหน้าจอสีดำเพื่อแสดงสถานะรอ
            waiting_screen = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(waiting_screen, "Move 'Start Processing' to 1 to begin", 
                        (50, FRAME_HEIGHT // 2), font, 0.7, (255, 255, 255), 2)
            cv2.imshow("Annotated Output", waiting_screen)
            cv2.imshow("Combined Mask", np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8))
        
        # ถ้า Trackbar อยู่ที่ 1 (Start)
        else:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            kx100, morph, ksize, iters, enabled = read_controls(cfg)
            k_val = (kx100 - 100) / 100.0

            if (w != last_w) or (h != last_h) or (k_val != last_k) or (map_x is None):
                map_x, map_y = compute_distortion_map(w, h, k_val)
                last_w, last_h, last_k = w, h, k_val

            distorted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV)
            annotated = distorted.copy()
            all_masks = []
            
            ranges = cfg.get("ranges", DEFAULT_HSV_RANGES)
            for i, spec in enumerate(ranges):
                is_enabled = enabled[i] if i < len(enabled) else True
                lower = np.array(spec["lower"], dtype=np.uint8)
                upper = np.array(spec["upper"], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                all_masks.append(mask)

                if not is_enabled:
                    continue

            if all_masks:
                combined_mask = np.zeros_like(all_masks[0])
                for mask in all_masks:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                cv2.imshow("Combined Mask", combined_mask)
            
            cv2.imshow("Annotated Output", annotated)
        
        # ส่วนควบคุมการกดปุ่ม (ต้องอยู่นอก if/else เพื่อให้ทำงานตลอดเวลา)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            ranges, kx100, morph, ksize, iters = read_controls(cfg)
            save_config(cfg['ranges'], kx100, morph, ksize, iters)
            break
        elif key in (ord('s'), ord('S')):
            ranges, kx100, morph, ksize, iters = read_controls(cfg)
            save_config(cfg['ranges'], kx100, morph, ksize, iters)
        elif key in (ord('r'), ord('R')):
            print("Reset to defaults by reloading config file.")
            new_cfg = load_config_or_default()
            cfg.update(new_cfg) 
            cv2.destroyWindow("Controls")
            create_controls_windows(cfg)

    cv2.destroyAllWindows()

def main():
    cfg = load_config_or_default()
    create_controls_windows(cfg)
    print("Controls: S = save, R = reset, Q/ESC = quit")
    
    vision_processing_loop(cfg)

if __name__ == "__main__":
    main()