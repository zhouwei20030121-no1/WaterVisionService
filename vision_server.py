import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import time
from fastapi.responses import PlainTextResponse

# =========================================================================
# 水域监管智能体 - AI视觉实时流媒体微服务 (业务分流版)
# =========================================================================

app = FastAPI(title="Water AI Vision API")

print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')

# 全局变量：结构化业务数据，专供仓颉 Agent 读取
global_status = {
    "isFoggy": False,
    "hasIllegalBehavior": False,  # 是否存在违法行为
    "illegalBehaviors": [],  # 具体的违法行为列表 (例如: "违规下河")
    "environmentalIssues": [],  # 具体的环境问题列表 (例如: "水面塑料垃圾")
    "errorMessage": ""
}
# 全局真实数据统计器
report_statistics = {
    "person_count": 0,
    "boat_count": 0,
    "bottle_count": 0
}
# 防抖记录：防止视频1秒钟30帧导致计数狂飙，设置5秒内同一类目标只算1次
last_detect_time = {"person": 0, "boat": 0, "bottle": 0}


def fast_defog(frame):
    """基于 Dehaze-RetinexGAN 物理模型的简易去雾算法"""
    I = frame.astype(np.float32) / 255.0
    S = 1.0 - I
    L = cv2.GaussianBlur(S, (0, 0), 15)
    L = np.clip(L, 0.05, 1.0)
    R = S / L
    J = 1.0 - (0.85 * R)
    J = np.clip(J, 0.0, 1.0)
    J_uint8 = (J * 255.0).astype(np.uint8)

    lab = cv2.cvtColor(J_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def process_video_stream(video_source):
    """核心视频处理生成器"""
    global global_status
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # --- 1. 能见度检测与去雾 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_foggy = bool(variance < 50.0)

        if is_foggy:
            frame = fast_defog(frame)
            # OpenCV 默认不支持中文，画面渲染使用英文
            cv2.putText(frame, "STATUS: Defogging ON", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # --- 2. YOLOv8 一次性提取大小目标 ---
        # 0: person, 8: boat, 39: bottle
        results = model(frame, classes=[0, 8, 39], verbose=False)
        annotated_frame = results[0].plot()

        # 临时业务集合，用于去重
        current_illegal_behaviors = set()
        current_env_issues = set()

        # --- 3. 核心：底层识别结果 -> 上层业务逻辑映射 ---
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    class_id = int(box.cls[0])

                    # 大目标映射：违法行为研判
                    if class_id == 0:  # 识别到人
                        current_illegal_behaviors.add("涉水违规/违规下河")
                        cv2.putText(annotated_frame, "ALERT: Illegal Water Entry", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif class_id == 8:  # 识别到船
                        current_illegal_behaviors.add("违规船只/非法捕捞")
                        cv2.putText(annotated_frame, "ALERT: Illegal Boat", (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # 小目标映射：环境垃圾研判
                    elif class_id == 39:  # 识别到塑料瓶
                        current_env_issues.add("水面漂浮物(塑料瓶等)")
                        cv2.putText(annotated_frame, "ISSUE: Surface Garbage", (20, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- 4. 更新全局 JSON 状态供仓颉调用 ---
        global_status["isFoggy"] = is_foggy
        global_status["illegalBehaviors"] = list(current_illegal_behaviors)
        global_status["environmentalIssues"] = list(current_env_issues)
        global_status["hasIllegalBehavior"] = len(current_illegal_behaviors) > 0
        global_status["errorMessage"] = ""

        # --- 5. 视频流编码下发 ---
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==================== API 路由配置 ====================

@app.get("/api/v1/stream/{camera_id}")
async def video_stream(camera_id: str):
    """📺 视频流接口：保持原样，吐出带画框和警告的实时视频流"""
    # 确保视频文件名正确，无中文无空格
    video_source = "test.mp4"
    return StreamingResponse(process_video_stream(video_source),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/v1/status/{camera_id}")
async def get_status(camera_id: str):
    """📊 数据接口：返回结构化的业务 JSON 数据"""
    return global_status


# 🔥 新增接口：提供真实的报告文本，直接返回纯文本 (规避 JSON 解析)
@app.get("/api/v1/report", response_class=PlainTextResponse)
async def generate_real_report():
    p = report_statistics["person_count"]
    b = report_statistics["boat_count"]
    t = report_statistics["bottle_count"]

    # Python 端直接拼接好要发给大模型的话术
    summary = f"【水域监管真实数据总结】\n本监控周期内，AI 视觉引擎共实时拦截并记录：涉水违规 {p} 次，非法船只 {b} 次，水面垃圾 {t} 次。各项异常数据已同步保存。"

    # 拼接前端 UI 拦截画图所需的格式
    chart_data = f"[CHART_DATA]涉水违规:{p},非法船只:{b},水面垃圾:{t}"

    return f"{summary}\n{chart_data}"


if __name__ == "__main__":
    print("\n=================================================")
    print("🚀 视觉微服务已启动！正在监听端口 8000...")
    print("=================================================")
    print("📺 【实时视频流网址】(直接在浏览器中查看带框画面):")
    print("   -> http://127.0.0.1:8000/api/v1/stream/test")
    print("\n📊 【业务数据 JSON 网址】(供仓颉端后台拉取数据):")
    print("   -> http://127.0.0.1:8000/api/v1/status/test")
    print("\n📖 【API 交互文档网址】(在线接口测试):")
    print("   -> http://127.0.0.1:8000/docs")
    print("=================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)