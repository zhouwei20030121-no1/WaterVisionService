# 🌊 Water-Vision-Service (水域视觉感知微服务)

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

本项目是“水域监管智能体”的核心视觉算法节点。采用 Python 独立开发，基于 FastAPI 框架封装，为仓颉 (Cangjie) 业务侧 Agent 提供高实时、低延迟的 RESTful 视觉分析 API。

## ✨ 核心特性

- **🚀 极简微服务架构**：与主业务逻辑完全解耦，支持跨语言（Cangjie / Java / Go）HTTP 调用。
- **👁️ 实时目标检测**：集成 Ultralytics YOLOv8 轻量级大模型，精准识别 `person`（违规下河）、`boat`（非法船只）、`plastic_bottle`（水面垃圾）等关键目标。
- **🌫️ 智能环境感知**：内置基于 OpenCV 拉普拉斯方差算法的能见度评估，实时触发大雾预警与动态去雾增强。
- **⚡ 高并发与自动文档**：借助 FastAPI 与 Uvicorn，提供自带 Swagger UI 的交互式接口文档，方便联调与测试。

## 📁 核心目录结构

```text
Water-Vision-Service/
├── vision_server.py       # FastAPI 微服务主程序
├── test_local_video.py    # 本地视频可视化测试脚本
├── download_videos.py     # 自动化测试素材下载脚本
├── yolov8n.pt             # YOLOv8 预训练权重 (首次运行自动下载)
└── README.md              # 项目说明文档

## 🛠️ 快速启动

### 1. 克隆项目与创建环境
建议使用 Python 虚拟环境 (Virtualenv) 来隔离项目依赖：

```bash
git clone [https://github.com/zhouwei20030121-no1/Water-Vision-Service.git](https://github.com/zhouwei20030121-no1/Water-Vision-Service.git)
cd Water-Vision-Service
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate

### 2. 安装核心依赖

```bash
pip install fastapi uvicorn opencv-python ultralytics


###3. 启动服务
Bash
uvicorn vision_server:app --host 127.0.0.1 --port 8000
服务启动后，将在终端看到 Uvicorn running on http://127.0.0.1:8000 的成功日志。

###📡 API 接口说明
微服务启动后，可直接在浏览器访问 http://127.0.0.1:8000/docs 查看交互式 API 文档。

核心端点：视觉画面分析
URL: /api/v1/analyze/{camera_id}

Method: GET

说明: 传入指定摄像头 ID（或本地视频文件名），返回当前帧的结构化分析结果。

成功响应示例 (HTTP 200):

JSON
{
  "isFoggy": true,
  "detectedTargets": [
    "person",
    "boat"
  ],
  "errorMessage": ""
}
###🧪 本地算法测试
为了方便算法调优与效果展示，本项目附带了本地视频直观测试工具。

运行 python download_videos.py 获取测试视频。

运行 python test_local_video.py，将会弹出图形界面，实时渲染 YOLO 识别框与去雾增强效果。

###🤝 协作指南
本服务目前专为“水域监管 Agent”定制。调用端请配合 AgentVisionConnector (仓颉侧的三方库) 进行集成，以实现最佳的系统健壮性与异常兜底逻辑。


