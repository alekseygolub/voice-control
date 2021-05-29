import requests
import time
import json
import sys

from collections import deque

sys.path.append("cloudapi/output")
import yandex.cloud.ai.stt.v2.stt_service_pb2 as stt_service_pb2
import yandex.cloud.ai.stt.v2.stt_service_pb2_grpc as stt_service_pb2_grpc
import grpc

import bin.config as config
from tools.helpers import convertBinFloat32ToBinInt16

from dateutil import parser
import datetime


token = None
token_expiration_time = None

def request_cloud_api_token(oauth_token):
    global token_expiration_time
    global token
    data = "{{\"yandexPassportOauthToken\":\"{}\"}}".format(oauth_token)
    for i in range(10):
        print("Trying to get cloud_api_token")
        ret = requests.post(config.CLOUD_API_URL, data=data)
        if ret.ok:
            j = json.loads(ret.text)
            token_expiration_time = parser.parse(j['expiresAt'])
            token = j['iamToken']
            print("Success! Token expired at", token_expiration_time)
            return
        else:
            print("Request failed with code: {}. Try again in 5 second".format(ret.status_code))
            time.sleep(5)
    raise Exception('All request failed')


specification = stt_service_pb2.RecognitionSpec(
    language_code=config.LANGUAGE_CODE,
    model='general',
    partial_results=True,
    audio_encoding='LINEAR16_PCM',
    sample_rate_hertz=config.SAMPLE_RATE,
    single_utterance=True
)

def setupDataStream(folder_id, stream, mutex, record=deque()):
    global specification
    # Set the recognition settings
    streaming_config = stt_service_pb2.RecognitionConfig(specification=specification, folder_id=folder_id) 

    # Send a message with the recognition settings.
    yield stt_service_pb2.StreamingRecognitionRequest(config=streaming_config)

    # Creating a listening microphone generator
    data = b''
    if len(record) == 0:
        mutex.acquire()
        data = stream.read(config.CHUNK_SIZE, exception_on_overflow = False)
        mutex.release()
    else:
        data = record.popleft()
    data = convertBinFloat32ToBinInt16(data)
    
    while data != b'':
        yield stt_service_pb2.StreamingRecognitionRequest(audio_content=data)
        if (len(record) > 0):
            data = record.popleft()
        else:
            mutex.acquire()
            data = stream.read(config.CHUNK_SIZE, exception_on_overflow = False)
            mutex.release()
        data = convertBinFloat32ToBinInt16(data)


def recognize_phrase(folder_id, stream, mutex, oauth, record=deque()):
    global token_expiration_time
    global token
    if token_expiration_time is None or datetime.datetime.now(datetime.timezone.utc) >= token_expiration_time:
        request_cloud_api_token(oauth)
    
    # Establishing connection
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', cred)
    stub = stt_service_pb2_grpc.SttServiceStub(channel)

    # Sending data for recognition
    it = stub.StreamingRecognize(setupDataStream(folder_id, stream, mutex, record), metadata=(('authorization', 'Bearer %s' % token),))

    # Processing responses
    text = ''
    flag = 0
    for r in it:
        for alternative in r.chunks[0].alternatives:
            text = alternative.text
            if len(text.split()) > config.MAXWORDINRECORD and config.KEYWORD not in text.lower().split():
                channel.close()
                return ''
            break
        if r.chunks[0].final:
            channel.close()
            if config.KEYWORD not in text.lower().split():
                return ''
            return text

