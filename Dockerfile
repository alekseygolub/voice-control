FROM ubuntu:20.04

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install --yes \
    git \
    python3 \
    python3-pip \
    portaudio19-dev \
    libsndfile1 \
    alsa-base \
    alsa-utils
RUN pip3 install \
    numpy \
    torch \
    torchaudio
RUN pip3 install \
    python-dateutil \
    requests \
    librosa \
    grpcio-tools \
    paho-mqtt \
    sklearn \
    PyAudio

RUN git clone https://github.com/alekseygolub/voice-control.git \
    && cd voice-control \
    && git clone https://github.com/yandex-cloud/cloudapi \
    && cd cloudapi \
    && mkdir output \
    && python3 -m grpc_tools.protoc -I . -I third_party/googleapis --python_out=output --grpc_python_out=output google/api/http.proto google/api/annotations.proto yandex/cloud/api/operation.proto google/rpc/status.proto yandex/cloud/operation/operation.proto yandex/cloud/ai/stt/v2/stt_service.proto

COPY data /voice-control/data