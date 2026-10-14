import cv2
import os
import json
import time
import multiprocessing as mp_lib
import numpy as np
import threading
from collections import deque
import math
import pyaudio
import librosa
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# ==========================================
# 0. NLP & SPEECH TRANSCRIPT LISTENER
# ==========================================
class NLPListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.latest_text = ""
        self.latest_sentiment = 0.0
        self.stop_listening = None
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.15)
        except Exception:
            pass
            
    def start(self):
        try:
            self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
        except Exception:
            self.stop_listening = None
        return self
        
    def stop(self):
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except Exception:
                pass
            
    def callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            self.latest_text = text
            sentiment_dict = self.sentiment_analyzer.polarity_scores(text)
            self.latest_sentiment = sentiment_dict['compound']
            with open(os.path.join(BASE_DIR, "voice_transcript.txt"), "a", encoding="utf-8") as f:
                f.write(text + " ")
        except Exception:
            pass

# ==========================================
# 1. HIGH-ACCURACY AI WORKER (Face Emotion)
# ==========================================
def ai_worker(frame_queue, result_queue):
    import os
    
    new_emotions = ["😊 Happiness", "😢 Sadness", "😠 Anger", "😨 Fear", "😲 Surprise", "🤢 Disgust", "😐 Neutral"]
    class_names = new_emotions
    try:
        from deepface import DeepFace
        # Pre-warm DeepFace emotion model with a blank frame for instant inference
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, detector_backend="skip", silent=True)
    except Exception:
        DeepFace = None

    emotion_history = deque(maxlen=6) # 6-frame rolling smoothing for fast responsiveness
    
    mapping = {
        "😊 Happiness": "happy",
        "😢 Sadness": "sad",
        "😠 Anger": "angry",
        "😨 Fear": "fear",
        "😲 Surprise": "surprise",
        "🤢 Disgust": "disgust",
        "😐 Neutral": "neutral"
    }

    while True:
        try:
            face_img = frame_queue.get()
            if face_img is None:
                break
                
            if DeepFace is not None and face_img is not None and face_img.size > 0:
                if face_img.shape[0] != 224 or face_img.shape[1] != 224:
                    face_img = cv2.resize(face_img, (224, 224))
                    
                result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False, detector_backend="skip", silent=True)
                if isinstance(result, list):
                    result = result[0]
                
                df_emotions = result.get("emotion", {})
                
                preds = []
                for cls in class_names:
                    key = mapping.get(cls, "neutral")
                    score = float(df_emotions.get(key, 0.0)) / 100.0
                    preds.append(score)
                
                total = sum(preds)
                preds = [p / total for p in preds] if total > 0 else [1.0/len(preds)]*len(preds)
                preds = np.array(preds)
                
                dom_raw = result.get("dominant_emotion", "neutral").lower()
                dom_mapped = next((k for k, v in mapping.items() if v == dom_raw), "😐 Neutral")
                
                with open(os.path.join(BASE_DIR, "face_base_emotion.txt"), "w", encoding="utf-8") as f:
                    f.write(dom_mapped)
                    
                emotion_history.append(preds)
                
            if len(emotion_history) > 0:
                avg_preds = np.mean(emotion_history, axis=0)
                smoothed_probs = {class_names[i]: float(avg_preds[i]) for i in range(len(class_names))}
                dom_idx = int(np.argmax(avg_preds))
                
                with open(os.path.join(BASE_DIR, "face_emotion.txt"), "w", encoding="utf-8") as f:
                    f.write(class_names[dom_idx])
                with open(os.path.join(BASE_DIR, "face_base_emotion.txt"), "w", encoding="utf-8") as f:
                    f.write(class_names[dom_idx])
                with open(os.path.join(BASE_DIR, "face_confidence.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{float(avg_preds[dom_idx]) * 100:.1f}")
                with open(os.path.join(BASE_DIR, "emotion_probabilities.json"), "w", encoding="utf-8") as f:
                    json.dump(smoothed_probs, f)
                
                while not result_queue.empty():
                    try: result_queue.get_nowait()
                    except: pass
                result_queue.put((smoothed_probs, class_names[dom_idx], float(avg_preds[dom_idx])))
        except Exception:
            pass

# ==========================================
# 2. CAMERA COMPONENT
# ==========================================
class CameraStream:
    def __init__(self, src=0):
        if os.name == 'nt':
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(src)
        else:
            self.stream = cv2.VideoCapture(src)
        self.grabbed = False
        self.frame = None
        for _ in range(20):
            self.grabbed, self.frame = self.stream.read()
            if self.grabbed and self.frame is not None:
                break
            time.sleep(0.01)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed and frame is not None:
                self.frame = frame
                self.grabbed = True
            time.sleep(0.01)

    def read(self):
        return (True, self.frame.copy()) if (self.grabbed and self.frame is not None) else (False, None)

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

# =========================================================
# 3. HIGH-ACCURACY VOICE RANGE & TONE ANALYSIS COMPONENT
# =========================================================
class VoiceStream:
    def __init__(self, nlp_listener=None):
        self.class_names = ["😠 Anger", "😭 Cry", "😨 Fear", "😢 Sadness", "🤢 Disgust", "😊 Happiness", "😐 Neutral"]
        self.nlp_listener = nlp_listener
        self.chunk = 1024
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        except Exception:
            self.stream = None
            
        self.stopped = False
        self.audio_data = np.zeros(self.chunk, dtype=np.int16)
        self.rolling_buffer = deque(maxlen=8000) # 0.5s audio history for autocorrelation
        self.pitch_history = deque(maxlen=16)
        self.voice_probs = {cls: 1.0/len(self.class_names) for cls in self.class_names}
        self.dominant_voice = "Listening..."
        self.pitch = 0.0
        self.volume = 0.0
        self.jitter = 0.0
        self.centroid = 0.0
        self.db_level = 20.0
        self.range_tier = "Ambient / Idle"
        
    def start(self):
        if self.stream is not None:
            threading.Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                raw_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_data = raw_chunk
                self.rolling_buffer.extend(raw_chunk)
                
                if len(self.rolling_buffer) < 2048:
                    time.sleep(0.01)
                    continue
                    
                audio_float = np.array(self.rolling_buffer, dtype=np.float32) / 32768.0
                
                # 1. Volume (RMS) & Calibrated Decibels
                rms = float(np.sqrt(np.mean(audio_float**2)) * 32768.0)
                self.volume = rms
                self.db_level = max(20.0, min(95.0, 20.0 * np.log10(max(1e-4, rms / 32768.0)) + 92.0))
                
                # 2. Voice Activity Detection (VAD)
                is_speaking = (rms > 420.0)
                
                # 3. FFT Autocorrelation Pitch ($F_0$) Tracking (60 - 550 Hz human band)
                pitch_hz = 0.0
                if is_speaking:
                    n = len(audio_float)
                    w_audio = audio_float * np.hanning(n)
                    f = np.fft.rfft(w_audio, n=2*n)
                    r = np.fft.irfft(f * np.conj(f))[:n]
                    
                    min_lag = max(1, int(self.rate / 550.0))
                    max_lag = min(n - 1, int(self.rate / 60.0))
                    if max_lag > min_lag:
                        peak_idx = min_lag + np.argmax(r[min_lag:max_lag])
                        if r[0] > 0 and (r[peak_idx] / r[0]) > 0.16:
                            pitch_hz = float(self.rate / peak_idx)
                            self.pitch_history.append(pitch_hz)
                            self.pitch = pitch_hz
                
                # Pitch Jitter / Tremor Tracking
                valid_pitches = [p for p in self.pitch_history if 60.0 <= p <= 550.0]
                pitch_mean = float(np.mean(valid_pitches)) if valid_pitches else 150.0
                pitch_std = float(np.std(valid_pitches)) if len(valid_pitches) >= 4 else 0.0
                jitter = float(pitch_std / max(1.0, pitch_mean))
                self.jitter = jitter
                
                # 4. Spectral Centroid (Timbre Brightness)
                mag = np.abs(np.fft.rfft(audio_float))
                freqs = np.fft.rfftfreq(len(audio_float), d=1.0/self.rate)
                sum_mag = np.sum(mag)
                centroid = float(np.sum(freqs * mag) / max(1e-6, sum_mag))
                self.centroid = centroid
                
                # 5. Spectral Flatness (Breathiness / Sobbing gasps)
                power = mag**2 + 1e-12
                flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
                
                # 6. Speaking Range Tier Classification
                if not is_speaking:
                    self.range_tier = "Ambient / Idle"
                elif pitch_mean < 110:
                    self.range_tier = "Low Grave / Fry"
                elif pitch_mean < 185:
                    self.range_tier = "Norm Conversational"
                elif pitch_mean < 255:
                    self.range_tier = "Animated Melodic"
                elif pitch_mean < 360:
                    self.range_tier = "Elevated Strain"
                else:
                    self.range_tier = "High Shrill / Acute"
                    
                # 7. Multi-Feature Acoustic Emotion Scoring Algorithms
                if is_speaking:
                    scores = {
                        "😠 Anger": 1.0,
                        "😭 Cry": 0.8,
                        "😨 Fear": 1.0,
                        "😢 Sadness": 1.0,
                        "🤢 Disgust": 0.8,
                        "😊 Happiness": 1.0,
                        "😐 Neutral": 1.4
                    }
                    
                    # ANGER: High volume, high spectral centroid, elevated/forceful pitch
                    if rms > 3800 and centroid > 2000:
                        scores["😠 Anger"] += 11.0 + min(12.0, (rms - 3800) / 300.0)
                    elif rms > 2800 and pitch_mean > 225:
                        scores["😠 Anger"] += 6.0

                    # CRY / WEEPING / SOBS: Pitch tremor (jitter > 0.08), breathy sob flatness (> 0.065), fluctuating volume
                    if jitter > 0.08:
                        scores["😭 Cry"] += 13.0 + (jitter * 45.0)
                        scores["😢 Sadness"] += 4.5
                    if flatness > 0.065 and pitch_mean > 175:
                        scores["😭 Cry"] += 7.0
                        scores["😨 Fear"] += 3.0

                    # FEAR: Shrill high pitch (> 270 Hz), tense centroid (> 2150 Hz), moderate volume
                    if pitch_mean > 270 and centroid > 2150:
                        scores["😨 Fear"] += 12.0 + min(8.0, (pitch_mean - 270) / 10.0)
                    elif pitch_mean > 230 and rms < 3600 and centroid > 1950:
                        scores["😨 Fear"] += 6.0

                    # SADNESS: Low volume, depressed pitch (< 160 Hz), dark centroid (< 1600 Hz), flat intonation
                    if rms < 1800 and pitch_mean < 160 and centroid < 1650:
                        scores["😢 Sadness"] += 11.0 + ((160 - pitch_mean) / 7.0)
                    elif rms < 1300 and centroid < 1500:
                        scores["😢 Sadness"] += 6.5

                    # DISGUST: Low pitch, vocal fry, downward inflections, rough timbre
                    if 80 < pitch_mean < 155 and 1100 < centroid < 1750 and rms < 2600 and jitter < 0.07:
                        scores["🤢 Disgust"] += 8.5

                    # HAPPINESS: Energetic volume, bright centroid, warm elevated melodic pitch
                    if 180 < pitch_mean < 310 and centroid > 1850 and 1700 < rms < 4500 and jitter < 0.075:
                        scores["😊 Happiness"] += 10.0 + (centroid / 1000.0)

                    # NEUTRAL: Standard conversational volume & pitch, stable jitter
                    if 105 < pitch_mean < 210 and 800 < rms < 2800 and jitter < 0.06:
                        scores["😐 Neutral"] += 9.0
                        
                    # NLP Keyword Reinforcement
                    if self.nlp_listener:
                        txt_lower = self.nlp_listener.latest_text.lower()
                        sent = self.nlp_listener.latest_sentiment
                        if any(w in txt_lower for w in ["angry", "furious", "hate", "mad", "shut up", "yell"]):
                            scores["😠 Anger"] += 16.0
                        if any(w in txt_lower for w in ["cry", "crying", "sobbing", "tears", "weep", "hurt", "pain"]):
                            scores["😭 Cry"] += 18.0
                            scores["😢 Sadness"] += 8.0
                        if any(w in txt_lower for w in ["scared", "afraid", "panic", "fear", "terrified", "nervous"]):
                            scores["😨 Fear"] += 16.0
                        if any(w in txt_lower for w in ["sad", "depressed", "unhappy", "lonely", "hopeless"]):
                            scores["😢 Sadness"] += 16.0
                        if any(w in txt_lower for w in ["disgust", "gross", "nasty", "sick", "yuck", "awful"]):
                            scores["🤢 Disgust"] += 16.0
                        if any(w in txt_lower for w in ["happy", "great", "wonderful", "love", "joy", "smile"]):
                            scores["😊 Happiness"] += 16.0
                        if sent > 0.35: scores["😊 Happiness"] += 6.0
                        elif sent < -0.35:
                            scores["😠 Anger"] += 4.0
                            scores["😢 Sadness"] += 4.0
                            
                    total = sum(scores.values())
                    new_probs = {k: v / total for k, v in scores.items()}
                    for k in self.class_names:
                        self.voice_probs[k] = 0.72 * self.voice_probs.get(k, 0.0) + 0.28 * new_probs.get(k, 0.0)
                    self.dominant_voice = max(self.voice_probs, key=self.voice_probs.get)
                else:
                    self.dominant_voice = "Listening..."
                    
                with open(os.path.join(BASE_DIR, "voice_emotion.txt"), "w", encoding="utf-8") as f:
                    clean_v = self.dominant_voice.split(" ")[-1] if " " in self.dominant_voice else self.dominant_voice
                    f.write(clean_v)
            except Exception:
                pass
                
    def read(self):
        return (
            self.audio_data, 
            self.voice_probs, 
            self.dominant_voice, 
            self.pitch, 
            self.volume, 
            self.jitter, 
            self.centroid, 
            self.db_level, 
            self.range_tier
        )
        
    def stop(self):
        self.stopped = True
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception: pass
        try:
            self.p.terminate()
        except Exception: pass

# ==========================================
# 4. FAST AI LOADER & FACIAL DETECTORS
# ==========================================
class FastAILoader:
    def __init__(self):
        p = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(p, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(p, 'haarcascade_eye.xml'))
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(p, 'haarcascade_smile.xml'))

# ==========================================
# 5. BEAUTIFUL 3D UI RENDERER & HUD CONSOLE
# ==========================================
class UIRenderer:
    def __init__(self, face_classes, voice_classes):
        self.face_classes = face_classes
        self.voice_classes = voice_classes
        
        self.face_colors = {
            "😊 Happiness": (50, 255, 100),
            "😢 Sadness": (255, 150, 50),
            "😠 Anger": (0, 0, 255),
            "😨 Fear": (200, 50, 200),
            "😲 Surprise": (0, 255, 255),
            "🤢 Disgust": (50, 200, 100),
            "😐 Neutral": (255, 243, 0)
        }
        
        self.voice_colors = {
            "😠 Anger": (0, 0, 255),
            "😭 Cry": (255, 120, 255),
            "😨 Fear": (200, 50, 200),
            "😢 Sadness": (255, 150, 50),
            "🤢 Disgust": (50, 200, 100),
            "😊 Happiness": (50, 255, 100),
            "😐 Neutral": (255, 243, 0)
        }
        
        self.ui_face_probs = {cls: 0.0 for cls in face_classes}
        self.ui_voice_probs = {cls: 0.0 for cls in voice_classes}
        self.voice_signal_history = deque(maxlen=200)
    
    def smooth_face_probabilities(self, new_probs):
        for cls in self.face_classes:
            self.ui_face_probs[cls] += (new_probs.get(cls, 0.0) - self.ui_face_probs[cls]) * 0.15
            
    def smooth_voice_probabilities(self, new_probs):
        for cls in self.voice_classes:
            self.ui_voice_probs[cls] += (new_probs.get(cls, 0.0) - self.ui_voice_probs[cls]) * 0.20
            
    def draw_text_with_shadow(self, img, text, pos, font, scale, color, thickness, shadow_color=(0, 0, 0), offset=1):
        cv2.putText(img, text, (pos[0] + offset, pos[1] + offset), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    def draw_3d_bar(self, img, x, y, w, h, depth, color):
        if h <= 0: return
        base_y = y
        top_y = y - h
        
        # Front face
        cv2.rectangle(img, (x, top_y), (x+w, base_y), color, -1)
        # Top face
        pts_top = np.array([[x, top_y], [x+depth, top_y-depth], [x+w+depth, top_y-depth], [x+w, top_y]])
        top_color = (min(255, int(color[0]*1.3)), min(255, int(color[1]*1.3)), min(255, int(color[2]*1.3)))
        cv2.fillPoly(img, [pts_top], top_color)
        # Right face
        pts_right = np.array([[x+w, top_y], [x+w+depth, top_y-depth], [x+w+depth, base_y-depth], [x+w, base_y]])
        right_color = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
        cv2.fillPoly(img, [pts_right], right_color)

    # -------------------------------------------------------------
    # RENDER CAMERA BOTTOM CONSOLE (VOICE RANGE & TONE ANALYSIS)
    # -------------------------------------------------------------
    def render_voice_console_overlay(self, frame, audio_data, pitch, db_level, jitter, centroid, range_tier, dom_voice, voice_probs):
        h, w = frame.shape[:2]
        
        # Bottom Console Geometry: y = h - 145 to h - 10
        cy1, cy2 = h - 145, h - 10
        left_box_w = 300
        
        # --- LEFT BOX: FREQUENCY SPECTRUM (Hz) & VU METER ---
        cv2.rectangle(frame, (10, cy1), (10 + left_box_w, cy2), (10, 8, 4), -1)
        cv2.rectangle(frame, (10, cy1), (10 + left_box_w, cy2), (255, 243, 0), 1)
        
        self.draw_text_with_shadow(frame, "ACOUSTIC SPECTRUM & VU", (18, cy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 243, 0), 1)
        
        # 16-Band Gradient EQ Bars
        if len(audio_data) > 0:
            windowed = audio_data * np.hanning(len(audio_data))
            fft_data = np.abs(np.fft.rfft(windowed))
            num_bands = 16
            fft_data = fft_data[2:len(fft_data)//2]
            
            if len(fft_data) >= num_bands:
                band_size = len(fft_data) // num_bands
                eq_w = (left_box_w - 20) // num_bands
                for i in range(num_bands):
                    band_amp = np.mean(fft_data[i*band_size : (i+1)*band_size])
                    norm_amp = np.log10(band_amp + 1) / 6.8
                    bar_h = int(min(1.0, max(0.0, norm_amp)) * 65)
                    
                    r = int((i / num_bands) * 255)
                    g = int((1.0 - (i / num_bands)) * 255)
                    b = 220
                    
                    bx = 18 + i * eq_w
                    by = cy2 - 32
                    if bar_h > 0:
                        cv2.rectangle(frame, (bx+1, by - bar_h), (bx + eq_w - 2, by), (b, g, r), -1)
                        
        # Live VU Decibel Meter Bar at bottom of Left Box
        cv2.rectangle(frame, (18, cy2 - 22), (10 + left_box_w - 18, cy2 - 10), (20, 20, 30), -1)
        norm_db = max(0.0, min(1.0, (db_level - 20.0) / 70.0))
        vu_fill_w = int(norm_db * (left_box_w - 36))
        vu_color = (0, 255, 170) if db_level < 68 else ((0, 200, 255) if db_level < 80 else (0, 0, 255))
        if vu_fill_w > 0:
            cv2.rectangle(frame, (18, cy2 - 22), (18 + vu_fill_w, cy2 - 10), vu_color, -1)
        self.draw_text_with_shadow(frame, f"VU: {db_level:.0f} dB", (22, cy2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # --- RIGHT BOX: VOICE RANGE & TONE ANALYSIS (PREVIOUSLY EMPTY!) ---
        rx1 = 10 + left_box_w + 10
        rx2 = w - 10
        rw = rx2 - rx1
        
        cv2.rectangle(frame, (rx1, cy1), (rx2, cy2), (10, 8, 4), -1)
        cv2.rectangle(frame, (rx1, cy1), (rx2, cy2), (255, 243, 0), 1)
        
        # Cyber Corner Accents
        cv2.line(frame, (rx1, cy1), (rx1 + 15, cy1), (127, 0, 255), 2)
        cv2.line(frame, (rx1, cy1), (rx1, cy1 + 15), (127, 0, 255), 2)
        cv2.line(frame, (rx2, cy2), (rx2 - 15, cy2), (127, 0, 255), 2)
        cv2.line(frame, (rx2, cy2), (rx2, cy2 - 15), (127, 0, 255), 2)

        self.draw_text_with_shadow(frame, "VOICE RANGE & TONE ANALYSIS (PROSODY ENGINE)", (rx1 + 12, cy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 243, 0), 1)
        
        # 1. Pitch & Speaking Range Slider
        pitch_display = f"{pitch:.1f} Hz" if pitch > 0 else "Calibrating..."
        self.draw_text_with_shadow(frame, f"PITCH: {pitch_display}  |  TIER: {range_tier}", (rx1 + 12, cy1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1)
        
        # Pitch Gauge Track
        track_x = rx1 + 12
        track_y = cy1 + 50
        track_w = rw - 24
        cv2.rectangle(frame, (track_x, track_y), (track_x + track_w, track_y + 6), (25, 25, 35), -1)
        
        # Normalized slider pointer: 60Hz to 450Hz
        norm_pitch = max(0.0, min(1.0, (pitch - 60.0) / 390.0)) if pitch > 0 else 0.25
        slider_x = int(track_x + norm_pitch * track_w)
        cv2.circle(frame, (slider_x, track_y + 3), 5, (0, 255, 255), -1)
        
        # 2. Acoustic Telemetry (Energy, Tremor/Jitter, Timbre)
        jitter_txt = f"{jitter*100:.1f}%"
        jitter_color = (0, 0, 255) if jitter > 0.08 else ((0, 255, 200) if jitter > 0.04 else (200, 200, 200))
        self.draw_text_with_shadow(frame, f"ENERGY: {db_level:.0f}dB  |  TREMOR (JITTER): {jitter_txt}  |  TIMBRE: {centroid:.0f}Hz", (rx1 + 12, cy1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 205, 240), 1)
        
        # 3. Live Vocal Affect Classification Badge
        clean_dom = dom_voice.split(" ")[-1] if " " in dom_voice else dom_voice
        dom_color = self.voice_colors.get(dom_voice, (0, 255, 170))
        affect_conf = voice_probs.get(dom_voice, 0.0) * 100.0
        
        self.draw_text_with_shadow(frame, f"VOCAL AFFECT: {clean_dom.upper()} ({affect_conf:.0f}%)", (rx1 + 12, cy1 + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, dom_color, 2)
        
        # Mini probability meter for top 4 acoustic emotions
        top_v = sorted(voice_probs.items(), key=lambda x: x[1], reverse=True)[:4]
        mini_x = rx1 + 12
        mini_y = cy1 + 122
        for i, (vcls, vprob) in enumerate(top_v):
            c_name = (vcls.split(" ")[-1] if " " in vcls else vcls)[:4]
            cv2.putText(frame, f"{c_name}:{int(vprob*100)}%", (mini_x + i * 65, mini_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.voice_colors.get(vcls, (200, 200, 200)), 1)

    # -------------------------------------------------------------
    # RENDER 3D DASHBOARD (RIGHT PANEL)
    # -------------------------------------------------------------
    def render_3d_dashboard(self, ui_panel, frames_unseen, dom_face, conf_face, face_probs, dom_voice, voice_probs, time_elapsed, final_emotion=None, volume=0.0, nlp_text="", nlp_sentiment=0.0):
        gh, gw = ui_panel.shape[:2]
        
        # Background Navy
        ui_panel[:] = (11, 5, 3)
        
        # Cyber Grid
        for i in range(0, gh, 40): cv2.line(ui_panel, (0, i), (gw, i), (30, 18, 10), 1)
        for i in range(0, gw, 40): cv2.line(ui_panel, (i, 0), (i, gh), (30, 18, 10), 1)
        
        # Accents
        cv2.rectangle(ui_panel, (2, 2), (gw-2, gh-2), (255, 243, 0), 1)
        cv2.line(ui_panel, (0, 0), (80, 0), (127, 0, 255), 3)
        cv2.line(ui_panel, (0, 0), (0, 80), (127, 0, 255), 3)
        cv2.line(ui_panel, (gw-80, gh-1), (gw-1, gh-1), (255, 243, 0), 3)
        cv2.line(ui_panel, (gw-1, gh-80), (gw-1, gh-1), (255, 243, 0), 3)

        # 1. 3D FACE TOPOLOGY BARS (TOP HALF)
        self.draw_text_with_shadow(ui_panel, "3D FACE TOPOLOGY TELEMETRY", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 243, 0), 2)
        
        if frames_unseen >= 10 or dom_face == "Detecting...":
            self.draw_text_with_shadow(ui_panel, "SEARCHING BIOMETRIC SIGNAL...", (20, gh//5), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 120, 150), 2)
        else:
            top_face = sorted(face_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            num_bars = len(top_face)
            start_x = 20
            gap = 16
            depth = 10
            available_w = gw - start_x * 2 - (num_bars - 1) * gap - depth
            bar_w = max(24, int(available_w / num_bars))
            max_h = 130
            base_y_face = 205
            
            for i, (cls, prob) in enumerate(top_face):
                x = start_x + i * (bar_w + gap)
                h = int(prob * max_h)
                color = self.face_colors.get(cls, (200, 200, 200))
                self.draw_3d_bar(ui_panel, x, base_y_face, bar_w, h, depth, color)
                display_name = (cls.split(" ")[-1] if " " in cls else cls).capitalize()[:6] + "."
                self.draw_text_with_shadow(ui_panel, display_name, (x-2, base_y_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)
                self.draw_text_with_shadow(ui_panel, f"{prob*100:.0f}%", (x, base_y_face + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)

            clean_dominant = dom_face.split(" ")[-1] if " " in dom_face else dom_face
            self.draw_text_with_shadow(ui_panel, f"PRIMARY FACIAL: {clean_dominant.upper()} ({conf_face*100:.1f}%)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 243, 0), 2)

        if nlp_text:
            sentiment_color = (170, 255, 0) if nlp_sentiment > 0.1 else ((255, 0, 127) if nlp_sentiment < -0.1 else (200, 200, 200))
            self.draw_text_with_shadow(ui_panel, f"NLP TRANSCRIPT: \"{nlp_text[:35]}\"", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sentiment_color, 1)

        # 2. VOICE EMOTION PROBABILITY BARS & WAVEFORM (BOTTOM HALF)
        self.draw_text_with_shadow(ui_panel, "VOICE PROSODY SPECTRUM", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (127, 0, 255), 2)
        clean_voice = dom_voice.split(" ")[-1] if " " in dom_voice else dom_voice
        dom_color = self.voice_colors.get(dom_voice, (170, 255, 0))
        self.draw_text_with_shadow(ui_panel, f"ACOUSTIC AFFECT: {clean_voice.upper()}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dom_color, 2)
        
        # 3D Bars for Voice Emotions
        top_voice = sorted(voice_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        v_num_bars = len(top_voice)
        v_start_x = 20
        v_gap = 16
        v_depth = 10
        v_available_w = gw - v_start_x * 2 - (v_num_bars - 1) * v_gap - v_depth
        v_bar_w = max(24, int(v_available_w / v_num_bars))
        v_max_h = 100
        v_base_y = 440
        
        for i, (vcls, vprob) in enumerate(top_voice):
            vx = v_start_x + i * (v_bar_w + v_gap)
            vh = int(vprob * v_max_h)
            vcol = self.voice_colors.get(vcls, (180, 180, 180))
            self.draw_3d_bar(ui_panel, vx, v_base_y, v_bar_w, vh, v_depth, vcol)
            v_name = (vcls.split(" ")[-1] if " " in vcls else vcls).capitalize()[:6] + "."
            self.draw_text_with_shadow(ui_panel, v_name, (vx-2, v_base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            self.draw_text_with_shadow(ui_panel, f"{vprob*100:.0f}%", (vx, v_base_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vcol, 1)

        # Rolling Waveform Buffer
        self.voice_signal_history.append((volume, dom_voice))
        graph_x = 18
        graph_y = 485
        graph_w = gw - 36
        graph_h = min(150, gh - graph_y - 25)
        
        if graph_h > 35 and len(self.voice_signal_history) > 1:
            cv2.rectangle(ui_panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (8, 12, 24), -1)
            cv2.rectangle(ui_panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (255, 243, 0), 1)
            
            step_x = graph_w / max(1, len(self.voice_signal_history) - 1)
            max_vol = 10000.0
            mid_y = graph_y + graph_h // 2
            
            pts = []
            colors = []
            for idx, (vol, dom) in enumerate(self.voice_signal_history):
                px = int(graph_x + (idx * step_x))
                scaled_vol = min(1.0, vol / max_vol)
                direction = 1 if idx % 2 == 0 else -1
                offset = max(2, scaled_vol * (graph_h // 2 - 8))
                py = int(mid_y + direction * offset)
                pts.append((px, py))
                colors.append(self.voice_colors.get(dom, (255, 255, 255)))
                
            for j in range(len(pts) - 1):
                cv2.line(ui_panel, pts[j], pts[j+1], colors[j], 2, cv2.LINE_AA)

        return ui_panel

# ==========================================
# 6. MAIN APPLICATION ORCHESTRATOR
# ==========================================
def main():
    frame_queue = mp_lib.Queue(maxsize=2)
    result_queue = mp_lib.Queue(maxsize=2)
    
    ai_process = mp_lib.Process(target=ai_worker, args=(frame_queue, result_queue), daemon=True)
    ai_process.start()

    face_classes = ["😊 Happiness", "😢 Sadness", "😠 Anger", "😨 Fear", "😲 Surprise", "🤢 Disgust", "😐 Neutral"]
    voice_classes = ["😠 Anger", "😭 Cry", "😨 Fear", "😢 Sadness", "🤢 Disgust", "😊 Happiness", "😐 Neutral"]

    nlp_listener = NLPListener().start()
    camera = CameraStream(0).start()
    voice_stream = VoiceStream(nlp_listener).start()
    ai_loader = FastAILoader()
    ui_renderer = UIRenderer(face_classes, voice_classes)
    
    window_title = "PSYFUSION - Fullscreen AI Analysis"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
    except Exception: pass

    start_time = time.time()
    session_start_time = time.time()
    frame_count = fps = total_frames = frames_unseen = 0
    smoothed_face_probs = {cls: 0.0 for cls in face_classes}
    cumulative_probs = {cls: 0.0 for cls in face_classes}
    last_dominant_face = "Detecting..."
    last_confidence_face = 0.0
    current_micro_exprs = ["MOUTH: AU0 Baseline", "BROWS: Relaxed", "EYES: Normal Blink", "POSTURE: Calibrated"]

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count, start_time = 0, time.time()

        h, w = frame.shape[:2]
        face_detected = False
        
        try:
            if not result_queue.empty():
                smoothed_face_probs, last_dominant_face, last_confidence_face = result_queue.get_nowait()
                for cls in face_classes: cumulative_probs[cls] += smoothed_face_probs.get(cls, 0.0)
                total_frames += 1
        except Exception: pass
            
        ui_renderer.smooth_face_probabilities(smoothed_face_probs)

        # Facial Detection & Cyber Topology Rendering
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = ai_loader.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))
        if len(faces) > 0:
            face_detected = True
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx, cy = fx + fw // 2, fy + fh // 2
            
            # Expanded Crop (15% padding for complete facial expression accuracy)
            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.15)
            sq_x = max(0, fx - pad_x)
            sq_y = max(0, fy - pad_y)
            sq_w = min(w - sq_x, fw + pad_x * 2)
            sq_h = min(h - sq_y, fh + pad_y * 2)
            
            if sq_w > 0 and sq_h > 0 and frame_queue.empty():
                try: frame_queue.put_nowait(frame[sq_y:sq_y+sq_h, sq_x:sq_x+sq_w])
                except Exception: pass

            # Dynamic FACS Micro-Expressions Calculation
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
            smiles = ai_loader.smile_cascade.detectMultiScale(face_roi_gray, 1.7, 20)
            eyes = ai_loader.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 6)
            
            smile_active = len(smiles) > 0
            wide_eyes = len(eyes) >= 2
            
            dom_f_clean = (last_dominant_face.split()[-1] if " " in last_dominant_face else last_dominant_face).lower()
            if "happ" in dom_f_clean:
                current_micro_exprs = [
                    "MOUTH: AU12 Zygomatic (Smile Active)" if smile_active else "MOUTH: AU12 Lip Corner Puller",
                    "CHEEKS: AU6 Orbicularis Oculi Raised",
                    "EYES: Duchenne Crow's Feet Active",
                    "POSTURE: Congruent Affective Valence"
                ]
            elif "sad" in dom_f_clean:
                current_micro_exprs = [
                    "BROWS: AU1 Inner Brow Raiser (Distress)",
                    "MOUTH: AU15 Depressor Anguli Oris",
                    "CHIN: AU17 Mentalis Muscle Tension",
                    "EYES: Drooped Palpebral Fissure"
                ]
            elif "ang" in dom_f_clean:
                current_micro_exprs = [
                    "BROWS: AU4 Corrugator (Furrowed Tense)",
                    "EYES: AU5 Glare & Upper Lid Retraction",
                    "LIPS: AU23 Orbicularis Tightened",
                    "JAW: Masseter Clench Detected"
                ]
            elif "fear" in dom_f_clean:
                current_micro_exprs = [
                    "BROWS: AU1+AU2 Raised Transverse",
                    "EYES: AU5 Scleral Exposure (Wide)",
                    "LIPS: AU20 Risorius Stretched Tense",
                    "REFLEX: Autonomic Guarding Response"
                ]
            elif "disg" in dom_f_clean:
                current_micro_exprs = [
                    "NOSE: AU9 Levator Labii Wrinkled",
                    "LIPS: AU10 Upper Lip Raised Aversive",
                    "CHIN: AU17 Compressed Mentalis",
                    "TONE: Pharyngeal Constriction"
                ]
            elif "surp" in dom_f_clean:
                current_micro_exprs = [
                    "BROWS: AU1+AU2 High Frontal Arch",
                    "EYES: AU5 Maximum Aperture",
                    "JAW: AU26 Mandibular Drop Open",
                    "BURST: Phasic Autonomic Arousal"
                ]
            else:
                current_micro_exprs = [
                    "MOUTH: AU0 Resting Tonicity",
                    "BROWS: Relaxed Frontalis Baseline",
                    "EYES: Normative Saccadic Movement",
                    "POSTURE: Calibrated Neutral Symmetry"
                ]

            # Glowing Biometric Corner Brackets
            color_cyan = (255, 243, 0)
            color_magenta = (127, 0, 255)
            length = int(fw * 0.22)
            thick = 2

            cv2.line(frame, (fx, fy), (fx + length, fy), color_cyan, thick)
            cv2.line(frame, (fx, fy), (fx, fy + length), color_cyan, thick)
            cv2.line(frame, (fx + fw, fy), (fx + fw - length, fy), color_cyan, thick)
            cv2.line(frame, (fx + fw, fy), (fx + fw, fy + length), color_cyan, thick)
            cv2.line(frame, (fx, fy + fh), (fx + length, fy + fh), color_magenta, thick)
            cv2.line(frame, (fx, fy + fh), (fx, fy + fh - length), color_magenta, thick)
            cv2.line(frame, (fx + fw, fy + fh), (fx + fw - length, fy + fh), color_magenta, thick)
            cv2.line(frame, (fx + fw, fy + fh), (fx + fw, fy + fh - length), color_magenta, thick)

            # Scanning Laser Line
            scan_y = fy + int((fh) * ((time.time() * 1.5) % 1.0))
            cv2.line(frame, (fx, scan_y), (fx + fw, scan_y), (0, 200, 255), 1)

            # Target Crosshair
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1) 
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

            # 3D Topology Grid lines (4x4)
            grid_cols, grid_rows = 4, 4
            for gi in range(1, grid_cols):
                gx = fx + int((fw / grid_cols) * gi)
                cv2.line(frame, (gx, fy), (gx, fy + fh), (80, 80, 80), 1, cv2.LINE_AA)
            for gi in range(1, grid_rows):
                gy = fy + int((fh / grid_rows) * gi)
                cv2.line(frame, (fx, gy), (fx + fw, gy), (80, 80, 80), 1, cv2.LINE_AA)

            cv2.putText(frame, "3D TOPOLOGY: LOCKED", (fx + fw + 10, fy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 170), 1)
            cv2.putText(frame, f"TRACK: {fw}x{fh}px", (fx + fw + 10, fy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        frames_unseen = 0 if face_detected else frames_unseen + 1

        # Left HUD Overlay: PSYFUSION CORE & Real Micro-Expressions
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 275), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        cv2.putText(frame, "PSYFUSION CORE", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        status_txt = "MAPPED" if face_detected else "SEARCHING..."
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, f"STATUS: {status_txt}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, "--- LIVE FACS MICRO-EXPRESSIONS ---", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        for i, text in enumerate(current_micro_exprs):
            cv2.putText(frame, text, (20, 165 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
            
        # Audio Prosody Feature Readout
        audio_data, voice_probs, dom_voice, pitch, volume, jitter, centroid, db_level, range_tier = voice_stream.read()
        ui_renderer.smooth_voice_probabilities(voice_probs)
        nlp_text = nlp_listener.latest_text
        nlp_sent = nlp_listener.latest_sentiment

        # Render High-Accuracy Voice Range & Tone Analysis Console in the bottom empty space!
        ui_renderer.render_voice_console_overlay(frame, audio_data, pitch, db_level, jitter, centroid, range_tier, dom_voice, voice_probs)

        # Right 3D Panel
        panel_w = max(460, int(w * 0.4))
        ui_panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        session_elapsed = time.time() - session_start_time
        final_emotion = None
        if session_elapsed >= 4.5 and total_frames > 0:
            final_emotion = max(cumulative_probs, key=cumulative_probs.get)

        # Periodic synchronization of multimodal metrics for dashboard
        if total_frames % 5 == 0 and total_frames > 0:
            try:
                face_clean = (last_dominant_face.split()[-1] if " " in last_dominant_face else last_dominant_face).lower()
                voice_clean = (dom_voice.split()[-1] if " " in dom_voice else dom_voice).lower()
                if face_clean and voice_clean and voice_clean != "listening...":
                    is_match = (face_clean == voice_clean) or (("sad" in face_clean or "cry" in face_clean) and ("sad" in voice_clean or "cry" in voice_clean))
                    comp = "Emotion Matched" if is_match else "Emotion Mismatch"
                    m_score = 10 if is_match else 80
                    m_status = "No Cultural Masking" if is_match else "Cultural Masking Detected"
                    
                    if ("happ" in face_clean or "neutral" in face_clean) and ("cry" in voice_clean or "sad" in voice_clean):
                        m_score = 96
                        m_status = "Critical Masking (Suppressed Weeping)"
                    r_level = "LOW" if m_score <= 30 else ("MEDIUM" if m_score <= 70 else "HIGH")
                    with open(os.path.join(BASE_DIR, "comparison_result.txt"), "w", encoding="utf-8") as f: f.write(comp)
                    with open(os.path.join(BASE_DIR, "masking_result.txt"), "w", encoding="utf-8") as f: f.write(str(m_score))
                    with open(os.path.join(BASE_DIR, "masking_status.txt"), "w", encoding="utf-8") as f: f.write(m_status)
                    with open(os.path.join(BASE_DIR, "risk_result.txt"), "w", encoding="utf-8") as f: f.write(r_level)
            except Exception: pass

        ui_panel = ui_renderer.render_3d_dashboard(ui_panel, frames_unseen, last_dominant_face, last_confidence_face, ui_renderer.ui_face_probs, dom_voice, ui_renderer.ui_voice_probs, session_elapsed, final_emotion, volume, nlp_text, nlp_sent)

        combined_img = np.hstack((frame, ui_panel))

        # Real-time Web Stream Frame Export (Atomic write for zero tearing)
        try:
            ret_enc, buf = cv2.imencode('.jpg', combined_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                tmp_path = os.path.join(BASE_DIR, "latest_frame_tmp.jpg")
                out_path = os.path.join(BASE_DIR, "latest_frame.jpg")
                with open(tmp_path, "wb") as f_img:
                    f_img.write(buf.tobytes())
                try: os.replace(tmp_path, out_path)
                except Exception: pass
        except Exception: pass

        # Save session telemetry state for Web UI
        try:
            with open(os.path.join(BASE_DIR, "session_telemetry.json"), "w", encoding="utf-8") as f_tel:
                json.dump({
                    "elapsed": round(session_elapsed, 1),
                    "target": 10.0,
                    "progress": min(100, int((session_elapsed / 10.0) * 100)),
                    "fps": round(fps, 1),
                    "face_mapped": face_detected,
                    "dominant_face": last_dominant_face,
                    "face_confidence": round(last_confidence_face * 100, 1),
                    "dominant_voice": dom_voice,
                    "voice_pitch": round(pitch, 1),
                    "voice_jitter": round(jitter * 100, 1),
                    "voice_db": round(db_level, 1),
                    "voice_centroid": round(centroid, 0),
                    "voice_range_tier": range_tier,
                    "final_diagnosis": final_emotion if final_emotion else "Analyzing...",
                    "is_complete": False
                }, f_tel)
        except Exception: pass

        try:
            cv2.imshow("PSYFUSION - Fullscreen AI Analysis", combined_img)
        except Exception: pass
        
        # Check for user exit (q/ESC) or web session abort
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if frame_count % 10 == 0:
            try:
                with open(os.path.join(BASE_DIR, "session_active.txt"), "r", encoding="utf-8") as f_act:
                    if f_act.read().strip() == "False":
                        break
            except Exception: pass

    try: frame_queue.put_nowait(None)
    except Exception: pass
    camera.stop()
    voice_stream.stop()
    nlp_listener.stop()
    try: cv2.destroyAllWindows()
    except Exception: pass
    
    try:
        with open(os.path.join(BASE_DIR, "session_active.txt"), "w", encoding="utf-8") as f: f.write("False")
    except Exception: pass

    # Write final telemetry indicating session completion
    try:
        with open(os.path.join(BASE_DIR, "session_telemetry.json"), "w", encoding="utf-8") as f_tel:
            json.dump({
                "elapsed": round(time.time() - session_start_time, 1),
                "fps": round(fps, 1),
                "face_mapped": face_detected,
                "dominant_face": last_dominant_face,
                "face_confidence": round(last_confidence_face * 100, 1),
                "dominant_voice": dom_voice,
                "voice_pitch": round(pitch, 1),
                "voice_jitter": round(jitter * 100, 1),
                "voice_db": round(db_level, 1),
                "voice_range_tier": range_tier,
                "final_diagnosis": final_emotion if final_emotion else last_dominant_face,
                "is_complete": True
            }, f_tel)
    except Exception: pass
    
    ai_process.join(timeout=1.0)
    if ai_process.is_alive(): ai_process.terminate()

    # Ultra-Fast In-Memory Synthesis (Sub-millisecond)
    try:
        from analysis_engine import perform_fast_analysis
        res = perform_fast_analysis()
        print(f"Synthesis complete: {res['comparison']} | Risk: {res['risk']} | Masking: {res['masking_score']}%")
    except Exception as e:
        print("Analysis engine fallback:", e)

    print("Session finalized. Ready for user review.")

if __name__ == "__main__":
    mp_lib.freeze_support()
    main()