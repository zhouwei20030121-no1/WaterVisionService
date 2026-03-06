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
