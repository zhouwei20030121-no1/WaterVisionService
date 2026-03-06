import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================================
# 本地视频视觉效果直观测试脚本
# 运行此脚本会弹出一个播放窗口，实时展示 YOLO 画框和动态去雾效果
# =========================================================================

# 1. 加载模型
print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')

# 2. 填入你的本地视频文件名称
video_path = "屏幕录制 2026-03-06 155545.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 错误：无法打开视频 {video_path}，请检查文件名是否正确，或者是否和代码在同一目录下。")
    exit()

print("✅ 视频加载成功！正在播放... (点击视频窗口后，按键盘 'q' 键退出)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频播放结束。")
        break

    # --- 1. 能见度检测 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_foggy = variance < 50.0

    # 复制一帧用于画面显示
    display_frame = frame.copy()

    # --- 2. 模拟触发去雾增强 ---
    if is_foggy:
        # 简单快速的 CLAHE 去雾增强
        lab = cv2.cvtColor(display_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        display_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 在画面左上角打上红色提示
        cv2.putText(display_frame, "DEFOG ACTIVATED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 3. YOLOv8 目标识别并自动画框 ---
    # YOLO 提供了一个非常方便的 plot() 方法，直接把识别框画在图片上
    results = model(display_frame, verbose=False)

    # 提取画好框的图像
    annotated_frame = results[0].plot()

    # 显示方差数值，方便你调整阈值
    cv2.putText(annotated_frame, f"Variance: {variance:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- 4. 渲染显示画面 ---
    cv2.imshow("Water Agent V1.0 - Local Video Test", annotated_frame)

    # 控制播放速度 (延迟 30 毫秒，大概 30 帧/秒)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()