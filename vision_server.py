import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import threading

# =========================================================================
# 水域监管智能体 - AI视觉实时流媒体微服务
# 架构：FastAPI + YOLOv8 + OpenCV 实时去雾流媒体分发
# =========================================================================

app = FastAPI(title="Water AI Vision API")

print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')

# 全局变量，用于存储最新的识别状态，供仓颉通过 JSON 接口拉取预警信息
global_status = {
    "isFoggy": False,
    "detectedTargets": [],
    "errorMessage": ""
}


def fast_defog(frame):
    """
    核心算法 1：快速自适应直方图均衡化 (CLAHE) 去雾
    相比暗通道先验算法，CLAHE 在 Python 中能保证 30fps 以上的实时处理速度
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def process_video_stream(video_source):
    """
    核心视频处理生成器：读取视频 -> 能见度检测 -> 去雾 -> YOLO 识别 -> 编码推流
    """
    global global_status
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 视频播放完毕则循环播放 (适用于本地视频测试)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # --- 1. 能见度检测与动态去雾 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_foggy = bool(variance < 50.0)

        if is_foggy:
            frame = fast_defog(frame)
            cv2.putText(frame, "Defogging ACTIVATED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # --- 2. YOLOv8 目标识别 (水域场景过滤) ---
        # classes: 0(person/涉水人员), 8(boat/违规船只), 39(bottle/水面垃圾)
        results = model(frame, classes=[0, 8, 39], verbose=False)

        raw_targets = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    class_id = int(box.cls[0])
                    raw_targets.append(model.names[class_id])

        unique_targets = list(set(raw_targets))

        # 更新全局状态，供 JSON 接口使用
        global_status["isFoggy"] = is_foggy
        global_status["detectedTargets"] = unique_targets
        global_status["errorMessage"] = ""

        # --- 3. 画面渲染引擎 ---
        # YOLO 提供了极其方便的 plot() 方法，直接在画面上画出彩色识别框
        annotated_frame = results[0].plot()

        # 添加自定义的违规行为警告文字
        if "person" in unique_targets:
            cv2.putText(annotated_frame, "WARNING: Illegal Water Entry", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- 4. 编码为 MJPEG 视频流格式 ---
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # 产出 multipart 视频流流数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==================== API 路由配置 ====================

@app.get("/api/v1/stream/{camera_id}")
async def video_stream(camera_id: str):
    """
    📺 视频流接口：仓颉 UI 组件直接加载这个 URL 即可看到实时画面
    """
    # 此处替换为你的测试视频源
    video_source = "屏幕录制 2026-03-06 155545.mp4"
    return StreamingResponse(process_video_stream(video_source),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/v1/status/{camera_id}")
async def get_status(camera_id: str):
    """
    📊 数据接口：仓颉后台每秒轮询此接口，获取最新预警数据来触发业务逻辑
    """
    return global_status


if __name__ == "__main__":
    print("\n=================================================")
    print("🚀 视觉微服务已启动！")
    print("📺 视频流请在浏览器预览: http://127.0.0.1:8000/api/v1/stream/test")
    print("=================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)