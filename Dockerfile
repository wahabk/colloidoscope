FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y \
    && apt-get install wget -y \
    && apt-get install git -y \
    && apt-get install libopenmpi-dev -y \
    && apt-get install python3 python3-dev -y

WORKDIR /app/
COPY colloidoscope /app/colloidoscope/
COPY examples /app/examples/
COPY setup.py /app/setup.py
COPY scripts /app/scripts/
COPY tests /app/tests/

RUN python3 -m pip install .
