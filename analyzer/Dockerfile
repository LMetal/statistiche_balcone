FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" scipy prometheus-client tflite-runtime==2.14.0

WORKDIR /app

# Modello YAMNet (TFLite) + elenco delle 521 classi
RUN python -c "import urllib.request as u; \
u.urlretrieve('https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite','yamnet.tflite'); \
u.urlretrieve('https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv','yamnet_class_map.csv')"

COPY app.py .
CMD ["python", "-u", "app.py"]
