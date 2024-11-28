FROM --platform=linux/amd64 nvidia/cuda:12.6.2-base-ubuntu22.04

# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh

# Shared python package cache
ENV PIP_CACHE_DIR="/runpod-volume/.cache/pip/"

RUN apt -y update && \
    apt -y install python3 python3-pip libgl1-mesa-glx libglu1-mesa libglib2.0-0

COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade -r /requirements.txt

COPY src .

CMD python3 -u /handler.py
