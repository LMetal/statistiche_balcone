import collections
import csv
import logging
import os
import subprocess
import time

import numpy as np
from prometheus_client import Counter, Gauge, start_http_server
from scipy import signal
from tflite_runtime.interpreter import Interpreter

STREAM_URL = os.environ["STREAM_URL"]
CAL_OFFSET_DB = float(os.getenv("CAL_OFFSET_DB", "0"))
THRESHOLD = float(os.getenv("DETECT_THRESHOLD", "0.30"))
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "5"))
PORT = int(os.getenv("METRICS_PORT", "8000"))

SR = 16000
FRAME = 15600  # campioni per frame YAMNet (~0.975 s)
WINDOW_FRAMES = max(1, round(WINDOW_SECONDS / (FRAME / SR)))

# Classe YAMNet -> etichetta mostrata in Grafana. Modificala a piacere.
TRACKED = {
    "Siren": "sirena",
    "Civil defense siren": "sirena_protezione_civile",
    "Emergency vehicle": "veicolo_emergenza",
    "Police car (siren)": "polizia",
    "Ambulance (siren)": "ambulanza",
    "Fire engine, fire truck (siren)": "pompieri",
    "Bell": "campana",
    "Church bell": "campana_chiesa",
    "Aircraft": "aereo",
    "Fixed-wing aircraft, airplane": "aeroplano",
    "Helicopter": "elicottero",
    "Thunder": "tuono",
    "Thunderstorm": "temporale",
    "Rain": "pioggia",
    "Wind": "vento",
    "Dog": "cane",
    "Car": "auto",
    "Truck": "camion",
    "Motorcycle": "moto",
    "Speech": "voce",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("analyzer")

# ---- metriche ----
g_level = Gauge("sound_level_db", "LAeq sulla finestra (dB, ponderazione A)")
g_level_max = Gauge("sound_level_max_db", "Picco di livello (frame ~1s) sulla finestra (dB A)")
g_score = Gauge("yamnet_score", "Punteggio massimo YAMNet sulla finestra", ["label"])
c_det = Counter("yamnet_detections_total", "Eventi rilevati da YAMNet", ["label"])
g_up = Gauge("stream_up", "1 se lo stream audio arriva")

# ---- filtro di ponderazione A (IEC 61672, approssimato a 16 kHz) ----
def make_a_weighting(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    z = [0, 0, 0, 0]
    p = 2 * np.pi * np.array([-f1, -f1, -f2, -f3, -f4, -f4])
    k = (2 * np.pi * f4) ** 2
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    _, h = signal.sosfreqz(sos, worN=[1000], fs=fs)
    sos[0, :3] /= abs(h[0])  # 0 dB a 1 kHz
    return sos

SOS = make_a_weighting(SR)
zi = np.zeros((SOS.shape[0], 2))

# ---- YAMNet ----
labels = [r["display_name"] for r in csv.DictReader(open("yamnet_class_map.csv"))]
idx = {}
for cls, short in TRACKED.items():
    if cls in labels:
        idx[short] = labels.index(cls)
    else:
        log.warning("Classe YAMNet non trovata: %s", cls)

interp = Interpreter(model_path="yamnet.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()

def classify(x):
    interp.set_tensor(inp["index"], x.reshape(inp["shape"]).astype(np.float32))
    interp.invoke()
    for o in outs:
        t = interp.get_tensor(o["index"])
        if t.size == len(labels):
            return t.flatten()
    raise RuntimeError("Output YAMNet non riconosciuto")

for short in idx:
    g_score.labels(short).set(0)
    c_det.labels(short)  # inizializza a 0

# ---- stream ----
def open_stream():
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-rw_timeout", "10000000",
        "-i", STREAM_URL, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)

def main():
    global zi
    start_http_server(PORT)
    log.info("Metriche su :%d  stream=%s  finestra=%d frame", PORT, STREAM_URL, WINDOW_FRAMES)
    active = {s: False for s in idx}
    ms_win = collections.deque(maxlen=WINDOW_FRAMES)
    sc_win = collections.deque(maxlen=WINDOW_FRAMES)

    while True:
        proc = open_stream()
        while True:
            data = proc.stdout.read(FRAME * 2)
            if len(data) < FRAME * 2:
                break
            g_up.set(1)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # livello sonoro (A-weighted)
            xa, zi = signal.sosfilt(SOS, x, zi=zi)
            ms_win.append(float(np.mean(xa ** 2)) + 1e-12)
            laeq = 10 * np.log10(np.mean(ms_win)) + CAL_OFFSET_DB
            peak = 10 * np.log10(max(ms_win)) + CAL_OFFSET_DB
            g_level.set(laeq)
            g_level_max.set(peak)

            # YAMNet
            scores = classify(x)
            sc_win.append(scores)
            wmax = np.max(sc_win, axis=0)
            for short, i in idx.items():
                g_score.labels(short).set(float(wmax[i]))
                s = float(scores[i])
                if s >= THRESHOLD and not active[short]:
                    c_det.labels(short).inc()
                    active[short] = True
                elif s < THRESHOLD * 0.7:
                    active[short] = False

        g_up.set(0)
        log.warning("Stream interrotto, riprovo tra 5 s")
        proc.kill()
        time.sleep(5)

if __name__ == "__main__":
    main()
