import os
import requests

def read_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "N/A"

face = read_file("face_emotion.txt").upper()
voice = read_file("voice_emotion.txt").upper()
risk = read_file("risk_result.txt").upper()
masking_str = read_file("masking_result.txt")
transcript = read_file("voice_transcript.txt")
try:
    masking = float(masking_str)
except:
    masking = 0.0

summary = ""
recommendations = ""
next_session = ""

# AI Generation via Ollama
prompt = f"""You are a senior clinical psychiatrist analyzing a patient's end-of-session multimodal biometric data.
Based on the patient's condition, generate a highly professional, detailed medical counseling report based on established psychiatric medical resources.

Patient Data:
- Facial Emotion: {face}
- Vocal Emotion: {voice}
- Cultural Masking Index: {masking}% (Indicates likelihood of concealing emotions)
- Clinical Risk Level: {risk}
- Voice Transcript during session: {transcript if transcript else 'No vocal transcript recorded.'}

You MUST structure your response EXACTLY as follows, using these exact three headers (no markdown for headers). Do not add any extra text outside of these headers.

SUMMARY
[Provide a detailed, professional clinical analysis of their emotional state and condition. CRITICAL: You must cite standard psychiatric frameworks (e.g., DSM-5 criteria) and refer to potential neurobiological mechanisms (e.g., autonomic nervous system arousal, HPA axis dysfunction, amygdala reactivity) that align with their biometric data.]

RECOMMENDATIONS
[Provide detailed therapeutic recommendations based on established clinical guidelines. Include specific cognitive or somatic therapies, potential medicinal/pharmacological pathways if severity warrants it, and direct patient feedback on their masking or risk levels.]

NEXT_SESSION
[State the recommended timeframe for the next meeting based on the severity of their condition, e.g. "Immediate", "1 Week", "4 Weeks".]
"""

ollama_url = "http://localhost:11434/api/generate"
payload = {
    "model": "phi3",
    "prompt": prompt,
    "stream": False
}

try:
    response = requests.post(ollama_url, json=payload, timeout=60)
    if response.status_code == 200:
        ai_text = response.json().get("response", "").strip()
        
        # Validate format roughly
        if "SUMMARY" in ai_text and "RECOMMENDATIONS" in ai_text:
            output = ai_text
        else:
            raise ValueError("AI output format invalid.")
    else:
        raise ValueError(f"HTTP {response.status_code}")
except Exception as e:
    # Fallback Analysis logic if AI is offline
    if "SAD" in face or "SAD" in voice or "FEAR" in face or "FEAR" in voice:
        summary = "The patient exhibits strong indicators of negative affect, primarily leaning towards sadness or fear. This points towards an underlying distress or anxiety."
        recommendations = "Consider Cognitive Behavioral Therapy (CBT) focusing on anxiety and mood regulation. If distress is severely impacting daily life, a psychiatric consultation for potential pharmacological intervention (e.g., SSRIs or anxiolytics) may be warranted."
        next_session = "1 Week (High Priority)"
    elif "ANGER" in face or "ANGER" in voice or "DISGUST" in face:
        summary = "The patient exhibits signs of frustration, anger, or disgust. This may indicate environmental stressors or unresolved conflicts."
        recommendations = "Recommend anger management strategies and stress-reduction techniques. Dialectical Behavior Therapy (DBT) could be beneficial for emotional regulation."
        next_session = "2 Weeks (Moderate Priority)"
    else:
        summary = "The patient predominantly displays neutral or positive affect. No severe immediate emotional distress detected."
        recommendations = "Continue current supportive counseling. Encourage maintenance of healthy lifestyle habits and mindfulness routines."
        next_session = "4 Weeks (Routine)"

    if masking > 50:
        summary += f" Furthermore, a high Cultural Masking Index ({masking}%) suggests the patient may be attempting to conceal their true emotional state."
        recommendations += " Focus on building therapeutic alliance and creating a safe space to reduce emotional masking."
        if "4 Weeks" in next_session:
            next_session = "2 Weeks (Elevated Priority due to Masking)"

    if "HIGH" in risk or "CRITICAL" in risk:
        summary += " WARNING: The overall clinical risk level is elevated. Immediate attention required."
        recommendations = "URGENT: Immediate psychiatric evaluation recommended. Ensure patient safety protocols are in place. " + recommendations
        next_session = "Immediate / 48 Hours"

    output = f"SUMMARY\n{summary}\n\nRECOMMENDATIONS\n{recommendations}\n\nNEXT_SESSION\n{next_session}"

with open("consultant_report.txt", "w", encoding="utf-8") as f:
    f.write(output)
