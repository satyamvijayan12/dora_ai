# DORA AI 🤖

## Overview

DORA AI is a Python-based emotion-aware AI interaction system.

It combines real-time facial expression detection using MediaPipe
with intelligent response generation via OpenAI API to create
a context-aware conversational experience.

The system demonstrates integration of Computer Vision with
Large Language Models in a modular AI architecture.

## Screenshots
<img width="1157" height="768" alt="dora_ai33" src="https://github.com/user-attachments/assets/d13168a1-3a9c-46dc-b64e-60163697b823" />
<img width="1249" height="767" alt="dora_ai332" src="https://github.com/user-attachments/assets/01e7ee27-adf0-40bd-9a32-ad216dfe6409" />
![dora_ai_icon](https://github.com/user-attachments/assets/9b2b66d6-bd75-4cdd-8446-0aab9244d156)


## System Architecture

Camera Input
    →
Face & Landmark Detection (MediaPipe)
    →
Expression Interpretation (Python Logic Layer)
    →
Emotion Context Mapping
    →
Response Generation (OpenAI API)
    →
Output Display

## Key Features

- Real-time facial landmark tracking
- Expression-based emotion inference
- LLM-driven contextual response generation
- Modular Python architecture
- API-based scalable design

## Tech Stack

- Python
- MediaPipe
- OpenAI API
- OpenCV
- NumPy


## Project Structure
dora_ai/ │ ├── main.py ├── expression_detector.py ├── response_engine.py ├── requirements.txt ├── assets/ └── README.md


## Current Status

🚧 Under Active Development

Planned Improvements:
- Enhanced emotion classification logic
- Response tone adaptation
- GUI/Web interface
- Performance optimization

## Installation

```bash
pip install -r requirements.txt
python main.py



