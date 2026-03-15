import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================================
# 本地视频视觉效果直观测试脚本
# 运行此脚本会弹出一个播放窗口，实时展示 YOLO 画框和动态去雾效果
# =========================================================================

def fast_defog(frame):
    """
    核心算法：基于 Dehaze-RetinexGAN 物理模型的简易去雾算法
    理论依据：Dehazing(I) = 1 - Retinex(1 - I)。[cite: 140]
    使用矩阵运算近似替代深度学习网络，保证本地测试的视频实时性。
    """
    # 将图像转换为 0-1 的浮点数，防止计算溢出
    I = frame.astype(np.float32) / 255.0

    # 1. 图像反转: S(x) = 1 - I(x) [cite: 140]
    S = 1.0 - I

    # 2. 估算反转图像的光照图 L(x)
    L = cv2.GaussianBlur(S, (0, 0), 15)
    L = np.clip(L, 0.05, 1.0)

    # 3. 计算反射率 R(x): R(x) = S(x) / L(x) [cite: 140]
    R = S / L

    # 4. 还原初步去雾图像: J(x) = 1 - R(x) [cite: 140]
    # 引入 0.85 的阻尼系数，防止画面局部过暗
    J = 1.0 - (0.85 * R)

    # 将数值安全地限制在 0-1 并转回标准图像格式
    J = np.clip(J, 0.0, 1.0)
    J_uint8 = (J * 255.0).astype(np.uint8)

    # 5. 快速色彩/对比度恢复
    lab = cv2.cvtColor(J_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


# 1. 加载模型
print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')

# 2. 填入你的本地视频文件名称
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 错误：无法打开视频 {video_path}，请检查文件名是否正确，或者是否和代码在同一目录下。")
    exit()

print("✅ 视频加载成功！正在播放... (点击视频窗口后，按键盘 'q' 键退出)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # 如果你想让视频循环播放，可以取消下面两行的注释
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # continue
        print("视频播放结束。")
        break

    # --- 1. 能见度检测 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_foggy = variance < 50.0

    # --- 2. 模拟触发去雾增强 ---
    if is_foggy:
        # 调用基于论文模型的新去雾算法
        display_frame = fast_defog(frame)

        # 在画面左上角打上红色提示
        cv2.putText(display_frame, "DEFOG ACTIVATED (Retinex)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # 如果没有起雾，直接使用原图
        display_frame = frame.copy()

    # --- 3. YOLOv8 目标识别并自动画框 ---
    # 过滤掉不需要的目标，仅保留：0(人), 8(船只), 39(瓶子/水面垃圾)
    results = model(display_frame, classes=[0, 8, 39], verbose=False)

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