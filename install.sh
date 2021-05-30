#!/bin/bash
sudo apt-get install portaudio19-dev
pip3 install -r requirements.txt
git clone https://github.com/yandex-cloud/cloudapi
cd cloudapi
mkdir output
python3 -m grpc_tools.protoc -I . -I third_party/googleapis --python_out=output --grpc_python_out=output google/api/http.proto google/api/annotations.proto yandex/cloud/api/operation.proto google/rpc/status.proto yandex/cloud/operation/operation.proto yandex/cloud/ai/stt/v2/stt_service.proto
