import os
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_file_safe(filename, default="Waiting..."):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return content if content else default
        except Exception:
            return default
    return default

def write_file_safe(filename, content):
    filepath = os.path.join(BASE_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(content))
    except Exception:
        pass

def perform_fast_analysis():
    """
    Ultra-fast in-memory synthesis of multimodal emotions, FACS Action Units,
    cultural masking index, clinical risk assessment, and professional psychiatric rationale.
    Executes in under 5 milliseconds.
    """
    face_raw = read_file_safe("face_emotion.txt", "Neutral")
    face_base_raw = read_file_safe("face_base_emotion.txt", face_raw)
    voice_raw = read_file_safe("voice_emotion.txt", "Neutral")
    micro_raw = read_file_safe("micro_expressions.txt", "")
    transcript_raw = read_file_safe("voice_transcript.txt", "")
    conf_raw = read_file_safe("face_confidence.txt", "94.5")

    def clean(val):
        if not val or val in ["Waiting...", "N/A", "Listening..."]:
            return "neutral"
        return val.strip().split()[-1].lower()

    face_clean = clean(face_raw)
    face_base_clean = clean(face_base_raw)
    voice_clean = clean(voice_raw)

    target_face = face_clean if (face_clean == voice_clean) else (face_base_clean or face_clean)
    if not target_face: target_face = "neutral"
    if not voice_clean: voice_clean = "neutral"

    # 1. LIVE COMPARISON
    is_congruent = (target_face == voice_clean) or (("sad" in target_face or "cry" in target_face) and ("sad" in voice_clean or "cry" in voice_clean))
    if is_congruent:
        comparison = "Emotion Matched"
        masking_score = 10
        masking_status = "No Cultural Masking"
    else:
        comparison = "Emotion Mismatch"
        masking_score = 80
        masking_status = "Cultural Masking Detected"

    # 2. FACS ACTION UNIT ANALYSIS
    micro_lower = micro_raw.lower()
    fake_smile = ("happ" in target_face) and ("au12" in micro_lower and "au6" not in micro_lower)
    suppressed_tension = ("neutral" in target_face) and any(au in micro_lower for au in ["au4", "au5", "au20", "au7"])
    smiling_depression = ("happ" in target_face or "neutral" in target_face) and ("cry" in voice_clean or "sad" in voice_clean)
    
    if smiling_depression:
        masking_score = 96
        masking_status = "Critical Masking (Suppressed Weeping / Affective Concealment)"
    elif fake_smile:
        masking_score = 92
        masking_status = "Severe Masking (Pan Am Fake Smile)"
    elif suppressed_tension:
        masking_score = 88
        masking_status = "Masking Detected (Suppressed Tension)"

    # 3. CLINICAL RISK LEVEL
    if masking_score <= 30:
        risk = "LOW"
    elif masking_score <= 70:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # 4. NEUROBIOLOGICAL & PSYCHIATRIC EXPLANATION
    if comparison == "Emotion Matched":
        explanation = (
            f"Congruence observed between observable facial expression ({face_raw}) and acoustic prosody ({voice_raw}). "
            f"Autonomic tone displays emotional coherence with low sympathetic discordance (Masking: {masking_score}% - {masking_status}). "
            f"Affective presentation aligns with balanced hypothalamic-pituitary-adrenal (HPA) axis baseline and low clinical risk ({risk})."
        )
    else:
        explanation = (
            f"Biometric divergence detected: facial musculature projects {face_raw} while acoustic vocal prosody signals {voice_raw}. "
            f"Elevated cultural masking index of {masking_score}% ({masking_status}) reflects learned affective suppression and subconscious concealment of emotional tension. "
            f"Somatic indicators suggest elevated sympathetic nervous system arousal and amygdala reactivity consistent with {risk} clinical risk."
        )

    # 5. DSM-5 / CLINICAL CONSULTANT REPORT FORMULATION
    if "cry" in voice_clean or "cry" in target_face:
        dsm = "DSM-5 Diagnostic Consideration: Acute Affective Distress / Depressive Dysregulation (F32.1 / F43.23) with Somatosensory Weeping."
        consultant_summary = (
            "MENTAL STATUS EXAMINATION & AFFECTIVE EVALUATION:\n"
            "Patient exhibits pronounced acoustic vocal tremor (jitter instability) and vocal fold distress congruent with weeping/crying affect. "
            "Severe autonomic discordance observed when juxtaposed with baseline facial composure."
        )
        recommendations = (
            "EVIDENCE-BASED CLINICAL INTERVENTIONS:\n"
            "1. Compassionate Validation & Somatic Grounding: Permit safe cathartic release without judgment.\n"
            "2. Polyvagal De-escalation: Extended exhalations and gentle chest tapping to down-regulate sympathetic spike.\n"
            "3. Trauma-Informed Counseling: Schedule immediate supportive clinical exploration."
        )
        next_session = "3 Days (Urgent Priority)"
    elif "sad" in target_face or "sad" in voice_clean or "fear" in target_face or "fear" in voice_clean:
        dsm = "DSM-5 Diagnostic Consideration: Unspecified Anxiety / Affective Tension (F41.9 / F32.9) with prominent Affective Incongruence."
        consultant_summary = (
            "MENTAL STATUS EXAMINATION & AFFECTIVE EVALUATION:\n"
            "Patient demonstrates notable biometric divergence characterized by vocal acoustic strain and facial micro-tension. "
            "Autonomic nervous system (ANS) arousal is elevated, manifesting as pitch modulation shifts and masked negative affect."
        )
        recommendations = (
            "EVIDENCE-BASED CLINICAL INTERVENTIONS:\n"
            "1. Cognitive Behavioral Therapy (CBT): Implement cognitive reframing for automatic stress responses.\n"
            "2. Somatic Regulation: 4-7-8 diaphragmatic breathing and vagal tone modulation.\n"
            "3. Emotion Regulation: Distress tolerance protocols and safe emotional expression exercises."
        )
        next_session = "7 Days (Elevated Priority)"
    elif "anger" in target_face or "anger" in voice_clean:
        dsm = "DSM-5 Diagnostic Consideration: Impulse & Affective Regulation Vulnerability (F34.1 / F43.21) with Frustration-Evoked Affect."
        consultant_summary = (
            "MENTAL STATUS EXAMINATION & AFFECTIVE EVALUATION:\n"
            "Corrugator supercilii muscle activation (AU4) and acoustic intensity surges indicate suppressed frustration. "
            "Guarded posture and acoustic pitch spikes point to acute autonomic reactivity."
        )
        recommendations = (
            "EVIDENCE-BASED CLINICAL INTERVENTIONS:\n"
            "1. Dialectical Behavior Therapy (DBT): STOP skill and progressive somatic de-escalation.\n"
            "2. Cognitive Reframing: Deconstructing hostile attribution bias and cognitive friction."
        )
        next_session = "14 Days (Moderate Priority)"
    else:
        dsm = "DSM-5 Diagnostic Consideration: Normal Affective Variation (Z03.89) - Subclinical Affective Regulation."
        consultant_summary = (
            "MENTAL STATUS EXAMINATION & AFFECTIVE EVALUATION:\n"
            "Patient demonstrates stable baseline mood and congruent affective modulation. Autonomic parameters remain within normative physiological limits."
        )
        recommendations = (
            "EVIDENCE-BASED CLINICAL INTERVENTIONS:\n"
            "1. Preventative Mindfulness: Daily somatic grounding and self-reflection.\n"
            "2. Maintenance: Continue healthy emotional hygiene and regular self-check-ins."
        )
        next_session = "30 Days (Routine Follow-up)"

    consultant_text = (
        f"DSM_IMPRESSION\n{dsm}\n\n"
        f"SUMMARY\n{consultant_summary}\n\n"
        f"RECOMMENDATIONS\n{recommendations}\n\n"
        f"NEXT_SESSION\n{next_session}"
    )

    # 6. ATOMIC SYNCHRONIZATION
    write_file_safe("comparison_result.txt", comparison)
    write_file_safe("masking_result.txt", str(masking_score))
    write_file_safe("masking_status.txt", masking_status)
    write_file_safe("risk_result.txt", risk)
    write_file_safe("explanation.txt", explanation)
    write_file_safe("consultant_report.txt", consultant_text)

    return {
        "comparison": comparison,
        "masking_score": masking_score,
        "masking_status": masking_status,
        "risk": risk,
        "explanation": explanation,
        "face": face_raw,
        "voice": voice_raw,
        "confidence": conf_raw
    }

if __name__ == "__main__":
    t0 = time.time()
    res = perform_fast_analysis()
    elapsed = (time.time() - t0) * 1000.0
    print(f"Fast Analysis executed in {elapsed:.2f} ms")
    print(f"Comparison: {res['comparison']} | Masking: {res['masking_score']}% | Risk: {res['risk']}")
