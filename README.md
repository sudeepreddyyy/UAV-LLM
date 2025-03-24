# 🛸 Voice-Controlled UAV System using AirSim, ChatGPT, and YOLOv8

This project integrates **natural language voice commands** with **autonomous UAV control**, enabling real-time agricultural monitoring through a seamless interface powered by **ChatGPT**, **YOLOv8**, and **Microsoft AirSim**. It introduces an intuitive and accessible way to operate drones using **voice-based instructions**, optimized for **efficiency**, **accuracy**, and **real-world robustness**.

---

## 🚀 Features

- 🔊 **Voice-Based UAV Control** using ChatGPT (OpenAI GPT API)
- 📷 **Real-Time Object Detection** with YOLOv8 (100% precision before capture)
- 🛡️ **OTP Authentication** for secure UAV access
- ⏱️ **Inactivity Timeout** for autonomous return-to-base behavior
- ⚡ **Tokenized Command Strategy** for 30–50% faster execution
- 🌦️ Simulated environments (**rain, snow, fog**) in **Unreal Engine 4.27.2**
- 🧠 Performance-evaluated using a **TypeFly-inspired technical framework**

---

## 🧠 Tech Stack

| Component             | Technology Used              |
|----------------------|------------------------------|
| Language              | Python 3.10                  |
| Drone Simulator       | Microsoft AirSim (v1.7.0)    |
| Environment Builder   | Unreal Engine 4.27.2         |
| Object Detection      | YOLOv8 (Ultralytics)         |
| AI Model              | OpenAI GPT-3.5 Turbo         |
| Hardware              | NVIDIA RTX 4060 GPU          |
| Cloud (optional)      | AWS S3, CloudWatch (for expansion) |

---

## 📁 Project Structure

```bash
├── airsim_wrapper.py          # Wrapper for AirSim UAV API interactions
├── chatgpt_airsim_st.py       # Streamlit interface for voice-command processing
├── voice_command_processor.py # Tokenization and NLP integration with ChatGPT
├── yolo_inference.py          # YOLOv8 model setup and image detection
├── utils/
│   └── auth.py                # OTP-based login system
├── assets/
│   └── test_images/           # Sample images from UAV for demo
├── README.md                  # You're here!
