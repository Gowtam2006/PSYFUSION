from flask import Flask, render_template, redirect, url_for, send_file, request, jsonify, Response # type: ignore
import os
import subprocess
import random
import requests
import json
import threading
import time
import sys

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_file(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read().strip()
    return "Waiting..."

def run_module(module):
    mod_path = os.path.join(BASE_DIR, module)
    if os.path.exists(mod_path):
        venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable
        subprocess.Popen([python_exec, mod_path], cwd=BASE_DIR)

@app.route("/")
def home():
    return render_template(
        "index.html",
        face=read_file("face_emotion.txt"),
        voice=read_file("voice_emotion.txt"),
        comparison=read_file("comparison_result.txt"),
        masking=read_file("masking_result.txt"),
        risk=read_file("risk_result.txt"),
        explanation=read_file("explanation.txt")
    )

session_proc = None
session_lock = threading.Lock()

def is_session_running():
    global session_proc
    return session_proc is not None and session_proc.poll() is None

def clean_emotion_name(val):
    if not val or val == "Waiting..." or val == "N/A":
        return ""
    parts = val.strip().split()
    return parts[-1].lower() if parts else ""

@app.route("/start")
def start():
    global session_proc
    with session_lock:
        if not is_session_running():
            with open(os.path.join(BASE_DIR, "session_active.txt"), "w", encoding="utf-8") as f:
                f.write("True")
            with open(os.path.join(BASE_DIR, "session_start.txt"), "w", encoding="utf-8") as f:
                f.write(str(time.time()))

            for f in ["voice_transcript.txt", "voice_stats.txt"]:
                fp = os.path.join(BASE_DIR, f)
                if os.path.exists(fp):
                    try: os.remove(fp)
                    except: pass

            # Reset telemetry so old session status does not trigger instant modal
            try:
                with open(os.path.join(BASE_DIR, "session_telemetry.json"), "w", encoding="utf-8") as f_tel:
                    json.dump({
                        "elapsed": 0.0,
                        "fps": 0.0,
                        "face_mapped": False,
                        "dominant_face": "Detecting...",
                        "dominant_voice": "Listening...",
                        "final_diagnosis": "Analyzing...",
                        "is_complete": False
                    }, f_tel)
            except Exception:
                pass

            venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            session_proc = subprocess.Popen([python_exec, os.path.join(BASE_DIR, "emotion.py")], cwd=BASE_DIR)

    return redirect(url_for("session"))

@app.route("/session")
def session():
    global session_proc
    # Auto-start if not running
    if not is_session_running():
        start()
    return render_template(
        "session.html",
        face=read_file("face_emotion.txt"),
        voice=read_file("voice_emotion.txt")
    )

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        frame_path = os.path.join(BASE_DIR, "latest_frame.jpg")
        last_mtime = 0
        while is_session_running():
            if os.path.exists(frame_path):
                try:
                    mtime = os.path.getmtime(frame_path)
                    if mtime != last_mtime:
                        last_mtime = mtime
                        with open(frame_path, "rb") as f:
                            frame_bytes = f.read()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception:
                    pass
            time.sleep(0.03) # ~30 FPS streaming
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/session_telemetry")
def api_session_telemetry():
    running = is_session_running()
    telemetry_path = os.path.join(BASE_DIR, "session_telemetry.json")
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["session_active"] = running
                if running:
                    data["is_complete"] = False
                return jsonify(data)
        except Exception:
            pass
    return jsonify({
        "elapsed": 0.0,
        "target": 10.0,
        "progress": 0,
        "fps": 0.0,
        "face_mapped": False,
        "dominant_face": read_file("face_emotion.txt"),
        "dominant_voice": read_file("voice_emotion.txt"),
        "voice_pitch": 0.0,
        "voice_jitter": 0.0,
        "voice_db": 20.0,
        "voice_range_tier": "Idle",
        "final_diagnosis": "Analyzing...",
        "is_complete": False,
        "session_active": running
    })

@app.route("/stop")
def stop():
    global session_proc
    with session_lock:
        if is_session_running():
            try:
                session_proc.terminate()
                session_proc.wait(timeout=1.0)
            except Exception:
                try:
                    session_proc.kill()
                except: pass
            session_proc = None

        with open(os.path.join(BASE_DIR, "session_active.txt"), "w", encoding="utf-8") as f:
            f.write("False")

        # Instant Sub-millisecond Analysis
        try:
            from analysis_engine import perform_fast_analysis
            perform_fast_analysis()
        except Exception:
            pass

        # Update telemetry to complete
        try:
            face_val = read_file("face_emotion.txt")
            voice_val = read_file("voice_emotion.txt")
            with open(os.path.join(BASE_DIR, "session_telemetry.json"), "w", encoding="utf-8") as f_tel:
                json.dump({
                    "elapsed": 5.0,
                    "fps": 30.0,
                    "face_mapped": True,
                    "dominant_face": face_val,
                    "dominant_voice": voice_val,
                    "voice_pitch": 0.0,
                    "voice_jitter": 0.0,
                    "voice_db": 20.0,
                    "voice_range_tier": "Idle",
                    "final_diagnosis": face_val,
                    "is_complete": True
                }, f_tel)
        except Exception:
            pass

    return redirect(url_for("home"))

@app.route("/emotion")
def emotion():
    return redirect(url_for("session"))

@app.route("/voice")
def voice():
    return redirect(url_for("session"))

@app.route("/api/data")
def api_data():
    face_raw = read_file("face_emotion.txt")
    face_base_raw = read_file("face_base_emotion.txt")
    voice_raw = read_file("voice_emotion.txt")
    micro_raw = read_file("micro_expressions.txt")

    face_clean = clean_emotion_name(face_raw)
    face_base_clean = clean_emotion_name(face_base_raw)
    voice_clean = clean_emotion_name(voice_raw)

    target_face = face_clean if (face_clean == voice_clean) else (face_base_clean or face_clean)

    if target_face and voice_clean and voice_clean != "listening...":
        # 1. Live Comparison (Congruence)
        is_matched = (target_face == voice_clean) or (("sad" in target_face or "cry" in target_face) and ("sad" in voice_clean or "cry" in voice_clean))
        if is_matched:
            comparison = "Emotion Matched"
            masking_score = 10
            masking_status = "No Cultural Masking"
        else:
            comparison = "Emotion Mismatch"
            masking_score = 80
            masking_status = "Cultural Masking Detected"

        # 2. FACS Action Unit Analysis & Suppressed Weeping Check
        micro_lower = micro_raw.lower() if micro_raw != "Waiting..." else ""
        if ("happ" in face_clean or "neutral" in face_clean) and ("cry" in voice_clean or "sad" in voice_clean):
            masking_score = 96
            masking_status = "Critical Masking (Suppressed Weeping)"
        elif "happ" in face_clean and ("au12" in micro_lower and "au6" not in micro_lower):
            masking_score = 92
            masking_status = "Severe Masking (Pan Am Fake Smile)"
        elif "neutral" in face_clean and ("au4" in micro_lower or "au5" in micro_lower or "au20" in micro_lower):
            masking_score = 88
            masking_status = "Masking Detected (Suppressed Tension)"

        # 3. Clinical Risk Level
        if masking_score <= 30:
            risk = "LOW"
        elif masking_score <= 70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # 4. Clinical Rationale / Explanation
        if comparison == "Emotion Matched":
            explanation = (
                f"The patient's facial expression ({face_raw}) and vocal prosody ({voice_raw}) "
                f"are well-aligned. Congruent affective markers indicate minimal emotional concealment "
                f"and low autonomic suppression."
            )
        else:
            explanation = (
                f"Biometric divergence detected: facial affect exhibits {face_raw} while vocal acoustics reflect {voice_raw}. "
                f"Calculated masking score is {masking_score}% ({masking_status}), indicating potential cultural suppression or concealed distress."
            )

        # Synchronize local state files
        try:
            with open(os.path.join(BASE_DIR, "comparison_result.txt"), "w", encoding="utf-8") as f: f.write(comparison)
            with open(os.path.join(BASE_DIR, "masking_result.txt"), "w", encoding="utf-8") as f: f.write(str(masking_score))
            with open(os.path.join(BASE_DIR, "masking_status.txt"), "w", encoding="utf-8") as f: f.write(masking_status)
            with open(os.path.join(BASE_DIR, "risk_result.txt"), "w", encoding="utf-8") as f: f.write(risk)
            with open(os.path.join(BASE_DIR, "explanation.txt"), "w", encoding="utf-8") as f: f.write(explanation)
        except Exception:
            pass
    else:
        comparison = read_file("comparison_result.txt")
        masking_score = read_file("masking_result.txt")
        risk = read_file("risk_result.txt")
        explanation = read_file("explanation.txt")

    return jsonify({
        "face": face_raw,
        "voice": voice_raw,
        "masking": masking_score,
        "risk": risk,
        "comparison": comparison,
        "explanation": explanation,
        "session_active": is_session_running()
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_msg:
        return jsonify({"response": "I am actively listening. Please share what you are experiencing."})
    
    try:
        from chat_engine import generate_gpt_clinical_response
        reply = generate_gpt_clinical_response(user_msg, history)
        return jsonify({"response": reply})
    except Exception as e:
        face = read_file("face_emotion.txt")
        voice = read_file("voice_emotion.txt")
        risk = read_file("risk_result.txt")
        masking = read_file("masking_result.txt")
        fallback_reply = (
            f"I am actively tracking your biometric state (Face: {face}, Voice: {voice}, Masking: {masking}%, Risk: {risk}). "
            f"Regarding your question about '{user_msg}': giving yourself space to explore difficult sensations without judgment reduces autonomic stress. "
            f"Would you like to practice a 4-7-8 calming breath or review your masking index breakdown?"
        )
        return jsonify({"response": fallback_reply})


@app.route("/masking")
def masking():
    venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    subprocess.run([python_exec, os.path.join(BASE_DIR, "masking.py")], cwd=BASE_DIR)
    return render_template("masking_animation.html", masking=read_file("masking_result.txt"), status=read_file("masking_status.txt"))


@app.route("/risk")
def risk():
    venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    subprocess.run([python_exec, os.path.join(BASE_DIR, "risk.py")], cwd=BASE_DIR)
    return render_template("risk_animation.html", risk=read_file("risk_result.txt"))


@app.route("/explain")
def explain():

    run_module("explain.py")

    return redirect(url_for("home"))


@app.route("/synthesis")
def synthesis():
    return render_template("synthesis_animation.html")


@app.route("/report")
def report():
    # Run the report script synchronously
    venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    # Run instant fast synthesis
    try:
        from analysis_engine import perform_fast_analysis
        perform_fast_analysis()
    except Exception:
        pass
    
    subprocess.run([python_exec, os.path.join(BASE_DIR, "report.py")], cwd=BASE_DIR)
    
    report_pdf = os.path.join(BASE_DIR, "report.pdf")
    return send_file(report_pdf, as_attachment=True, download_name="Clinical_Diagnostic_Report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)