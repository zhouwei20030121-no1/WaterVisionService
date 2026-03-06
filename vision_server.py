import cv2
import uvicorn
from fastapi import FastAPI
from ultralytics import YOLO

# =========================================================================
# 水域监管智能体 - AI视觉微服务 (Python端)
# 描述：通过 FastAPI 提供 HTTP 接口，真实调用 OpenCV 和 YOLOv8
# =========================================================================

# 1. 初始化 FastAPI 服务实例
app = FastAPI(title="Water AI Vision API")

# 2. 加载真实的 YOLOv8 轻量级模型 (首次运行会自动下载 yolov8n.pt 到当前目录)
print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')


@app.get("/api/v1/analyze/{camera_id}")
async def analyze_camera(camera_id: str):
    """
    视觉分析接口：接收摄像头ID，截取当前帧，进行去雾检测和目标识别
    """
    print(f"\n[微服务] 收到仓颉 Agent 请求，正在分析摄像头: {camera_id}")

    # 获取视频流：为了本地测试方便，这里填 0 调用你的电脑摄像头。
    # 如果你想用本地视频测试，请把 0 改成同目录下的视频文件名，如 "test_river.mp4"
    cap = cv2.VideoCapture("屏幕录制 2026-03-06 155545.mp4")

    if not cap.isOpened():
        return {"isFoggy": False, "detectedTargets": [], "errorMessage": f"无法连接摄像头/视频源: {camera_id}"}

    ret, frame = cap.read()
    cap.release()  # 读完一帧立刻释放摄像头资源，防止占用报错

    if not ret:
        return {"isFoggy": False, "detectedTargets": [], "errorMessage": "读取视频帧失败"}

    # --- 核心算法 1：能见度检测 (基于拉普拉斯方差) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_foggy = bool(variance < 50.0)  # 方差低于阈值，判定为有雾/模糊

    # --- 核心算法 2：YOLOv8 目标识别 ---
    results = model(frame, verbose=False)
    raw_targets = []

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5:  # 只保留置信度大于 50% 的结果
                class_id = int(box.cls[0])
                raw_targets.append(model.names[class_id])

    # 对识别结果进行去重（比如画面里有 3 个人，只向仓颉返回一个 "person" 即可）
    unique_targets = list(set(raw_targets))

    print(f"[微服务] 分析完成 -> 是否有雾: {is_foggy}, 识别目标: {unique_targets}")

    # --- 组装并返回标准 JSON 数据 ---
    return {
        "isFoggy": is_foggy,
        "detectedTargets": unique_targets,
        "errorMessage": ""
    }


if __name__ == "__main__":
    print("\n=================================================")
    print("🚀 AI 视觉微服务已启动！正在监听端口 8000...")
    print("=================================================")
    # 启动命令：运行在本地的 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)