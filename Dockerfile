# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=whisperx-medical-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY handler.py /app/handler.py
COPY paddle_ocr_factory.py /app/paddle_ocr_factory.py

CMD ["python", "-u", "/app/handler.py"]
