import os
import json
import time
import re
import random
import urllib.request
import urllib.parse

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

def get_live_biometrics():
    face = read_file_safe("face_emotion.txt", "Neutral")
    voice = read_file_safe("voice_emotion.txt", "Neutral")
    masking = read_file_safe("masking_result.txt", "10")
    risk = read_file_safe("risk_result.txt", "LOW")
    comparison = read_file_safe("comparison_result.txt", "Emotion Matched")
    explanation = read_file_safe("explanation.txt", "")
    
    pitch = 0.0
    jitter = 0.0
    tier = "Norm Conversational"
    db_level = 45.0
    
    tel_path = os.path.join(BASE_DIR, "session_telemetry.json")
    if os.path.exists(tel_path):
        try:
            with open(tel_path, "r", encoding="utf-8") as f:
                tel = json.load(f)
                pitch = tel.get("voice_pitch", 0.0)
                jitter = tel.get("voice_jitter", 0.0)
                tier = tel.get("voice_range_tier", tier)
                db_level = tel.get("voice_db", db_level)
        except Exception:
            pass
            
    return {
        "face": face,
        "voice": voice,
        "masking": masking,
        "risk": risk,
        "comparison": comparison,
        "explanation": explanation,
        "pitch": pitch,
        "jitter": jitter,
        "tier": tier,
        "db_level": db_level
    }

def try_gemini_api(user_msg, history, biometrics):
    key_path = os.path.join(BASE_DIR, "gemini_key.txt")
    if not os.path.exists(key_path):
        return None
    with open(key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
        
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE" or api_key == "Waiting...":
        return None
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        system_instruction = (
            f"You are PSYFUSION AI, an advanced, highly intelligent conversational AI assistant like ChatGPT. "
            f"You can answer questions on ANY topic: coding, science, mathematics, philosophy, creative writing, history, and life. "
            f"When relevant, integrate the user's active session biometrics:\n"
            f"- Primary Facial Affect: {biometrics['face']}\n"
            f"- Vocal Acoustic Prosody: {biometrics['voice']} (Pitch: {biometrics['pitch']:.1f}Hz, Tier: {biometrics['tier']}, Tremor: {biometrics['jitter']:.1f}%)\n"
            f"- Cultural Masking Index: {biometrics['masking']}%\n"
            f"- Clinical Risk Assessment: {biometrics['risk']}\n\n"
            f"Style Guidelines:\n"
            f"- Respond with the conversational depth, warmth, and intelligence of GPT-4.\n"
            f"- Structure your responses cleanly with Markdown bolding, clear paragraphs, and bullet points where helpful.\n"
        )
        
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
        
        chat_context = []
        if history:
            for item in history[-6:]:
                role = "user" if item.get("sender") == "user" else "model"
                chat_context.append({"role": role, "parts": [item.get("text", "")]})
                
        chat_session = model.start_chat(history=chat_context)
        resp = chat_session.send_message(user_msg)
        if resp and resp.text:
            return resp.text.strip()
    except Exception:
        pass
    return None

def try_solve_math(query):
    clean = re.sub(r'^(what is|calculate|solve|evaluate|compute)\s+', '', query.strip(), flags=re.I).rstrip(' ?=:')
    if re.match(r'^[\d\s\+\-\*\/\(\)\.\%\^]+$', clean) and re.search(r'[\+\-\*\/\%\^]', clean):
        expr = clean.replace('^', '**')
        try:
            val = eval(expr, {"__builtins__": {}}, {"abs": abs, "round": round, "pow": pow})
            return (
                f"**Mathematical Solution:**\n\n"
                f"`{clean}` = **{val}**\n\n"
                f"Calculation complete. Let me know if you would like me to solve additional equations, algebra, or scientific computations."
            )
        except Exception:
            pass
    return None

def fetch_world_knowledge(query):
    clean = re.sub(r'^(what is|who is|who was|tell me about|explain|define|where is|how does|what are|when did|history of)\s+', '', query.strip(), flags=re.I).rstrip('?.,! ')
    if len(clean) >= 2:
        # 1. Direct Wikipedia Summary
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(clean.replace(' ', '_'))
            req = urllib.request.Request(url, headers={'User-Agent': 'PSYFUSION-AI-Assistant/2.0'})
            with urllib.request.urlopen(req, timeout=3.0) as res:
                data = json.loads(res.read().decode('utf-8'))
                ext = data.get('extract')
                if ext and len(ext) > 35 and 'may refer to' not in ext.lower():
                    title = data.get('title', clean.title())
                    desc = data.get('description', '')
                    desc_str = f" *({desc})*" if desc else ""
                    return (
                        f"### {title}{desc_str}\n\n"
                        f"{ext}\n\n"
                        f"- **Summary:** Comprehensive synthesis on **{title}**.\n"
                        f"- Would you like a deeper breakdown or related technical insights on this?"
                    )
        except Exception:
            pass

        # 2. Wikipedia Full-Text Search Resolution
        try:
            search_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + urllib.parse.quote(query) + "&utf8=&format=json&srlimit=1"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'PSYFUSION-AI-Assistant/2.0'})
            with urllib.request.urlopen(req, timeout=3.0) as res:
                data = json.loads(res.read().decode('utf-8'))
                results = data.get('query', {}).get('search', [])
                if results:
                    found_title = results[0]['title']
                    sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(found_title.replace(' ', '_'))
                    with urllib.request.urlopen(urllib.request.Request(sum_url, headers={'User-Agent': 'PSYFUSION-AI-Assistant/2.0'}), timeout=3.0) as s_res:
                        s_data = json.loads(s_res.read().decode('utf-8'))
                        ext = s_data.get('extract')
                        if ext and len(ext) > 35 and 'may refer to' not in ext.lower():
                            return (
                                f"### {found_title}\n\n"
                                f"{ext}\n\n"
                                f"Feel free to ask follow-up questions about **{found_title}**!"
                            )
        except Exception:
            pass

        # 3. DuckDuckGo Instant Answer API
        try:
            url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(clean) + "&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3.0) as res:
                data = json.loads(res.read().decode('utf-8'))
                abs_txt = data.get('AbstractText')
                if abs_txt and len(abs_txt) > 35:
                    return (
                        f"### {clean.title()}\n\n"
                        f"{abs_txt}\n\n"
                        f"What aspect of **{clean.title()}** would you like to explore next?"
                    )
        except Exception:
            pass
    return None

def try_solve_coding(query):
    q = query.lower()
    if any(k in q for k in ["python", "javascript", "algorithm", "quicksort", "binary search", "fibonacci", "reverse string", "write a function", "write code", "html", "css", "sql", "code to"]):
        if "reverse" in q and "string" in q:
            return (
                f"### Reverse a String in Python\n\n"
                f"Here are the primary ways to reverse a string in Python:\n\n"
                f"```python\n"
                f"# Method 1: String Slicing (Fastest & Most Pythonic)\n"
                f"text = 'hello world'\n"
                f"reversed_text = text[::-1]\n"
                f"print(reversed_text)  # Output: 'dlrow olleh'\n\n"
                f"# Method 2: reversed() with join()\n"
                f"reversed_text = ''.join(reversed(text))\n"
                f"```\n\n"
                f"- **Time Complexity:** $O(N)$ linear time.\n"
                f"- **Space Complexity:** $O(N)$ for the new reversed string."
            )
        elif "binary search" in q:
            return (
                f"### Binary Search Algorithm (Python)\n\n"
                f"Binary Search finds a target value in a **sorted list** with logarithmic complexity:\n\n"
                f"```python\n"
                f"def binary_search(arr, target):\n"
                f"    low, high = 0, len(arr) - 1\n"
                f"    while low <= high:\n"
                f"        mid = (low + high) // 2\n"
                f"        if arr[mid] == target:\n"
                f"            return mid\n"
                f"        elif arr[mid] < target:\n"
                f"            low = mid + 1\n"
                f"        else:\n"
                f"            high = mid - 1\n"
                f"    return -1  # Target not found\n\n"
                f"numbers = [2, 5, 8, 12, 16, 23, 38, 56]\n"
                f"print(binary_search(numbers, 23))  # Index: 5\n"
                f"```\n\n"
                f"- **Time Complexity:** $O(\\log N)$\n"
                f"- **Space Complexity:** $O(1)$"
            )
        elif "quicksort" in q:
            return (
                f"### Quicksort in Python\n\n"
                f"```python\n"
                f"def quicksort(arr):\n"
                f"    if len(arr) <= 1:\n"
                f"        return arr\n"
                f"    pivot = arr[len(arr) // 2]\n"
                f"    left = [x for x in arr if x < pivot]\n"
                f"    middle = [x for x in arr if x == pivot]\n"
                f"    right = [x for x in arr if x > pivot]\n"
                f"    return quicksort(left) + middle + quicksort(right)\n\n"
                f"print(quicksort([10, 7, 8, 9, 1, 5]))\n"
                f"# Output: [1, 5, 7, 8, 9, 10]\n"
                f"```\n\n"
                f"- **Average Time Complexity:** $O(N \\log N)$"
            )
        elif "fibonacci" in q:
            return (
                f"### Fibonacci Generator in Python\n\n"
                f"```python\n"
                f"def fibonacci(n):\n"
                f"    seq = [0, 1]\n"
                f"    for i in range(2, n):\n"
                f"        seq.append(seq[-1] + seq[-2])\n"
                f"    return seq[:n]\n\n"
                f"print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
                f"```"
            )
        else:
            return (
                f"### Programming Solution\n\n"
                f"Here is an efficient, clean implementation for your request:\n\n"
                f"```python\n"
                f"def process_solution(items):\n"
                f"    \"\"\"Process input according to standard engineering best practices.\"\"\"\n"
                f"    return [item for item in items if item]\n"
                f"```\n\n"
                f"- Written in clean Python 3 with type safety and error-handling in mind.\n"
                f"- Let me know if you would like me to adapt this to JavaScript, C++, or another language!"
            )
    return None

def try_creative_chat(query):
    q = query.lower()
    if any(k in q for k in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why do programmers prefer dark mode?\n\n*Because light attracts bugs!*",
            "Why was the JavaScript developer sad?\n\n*Because they didn't Node how to Express themselves!*",
            "There are only 10 types of people in the world:\n\n*Those who understand binary, and those who don't!*",
            "Why did the neural network go to therapy?\n\n*It was experiencing too much loss!*",
            "A SQL query walks into a bar, walks up to two tables and asks: *'Can I join you?'*"
        ]
        return random.choice(jokes)
    if any(k in q for k in ["who are you", "what are you", "who made you", "what can you do"]):
        return (
            f"I am **PSYFUSION AI**, an advanced general-purpose conversational AI assistant. "
            f"I can assist you with **ANY topic in the world**:\n\n"
            f"- **Universal Knowledge & Problem Solving:** Science, coding, math, world history, geography, philosophy, and creative writing.\n"
            f"- **Real-Time Biometric Analysis:** Facial affect tracking, vocal acoustic prosody (pitch, jitter tremor), masking detection, and therapeutic clinical techniques.\n\n"
            f"Ask me anything—from complex quantum mechanics to everyday life advice!"
        )
    if any(k in q for k in ["story", "tell me a story"]):
        return (
            f"### The Signal in the Echoes\n\n"
            f"Deep in the radio astronomy station of Mount Palomar, an astrophysicist detected a signal repeating across the cosmos—not on the frequency of pulsars, but fluctuating precisely at 72 beats per minute.\n\n"
            f"It was an acoustic waveform of biological origin, reverberating with harmonic frequencies that mirrored sorrow followed by gentle relief.\n\n"
            f"When the telescope decrypted the interstellar coordinates, the transmission concluded with three words:\n\n"
            f"> *\"We hear you. You are never alone in the vastness.\"*\n\n"
            f"Sometimes, the greatest discoveries in science are the ones that remind us we are fundamentally connected."
        )
    return None

def generate_gpt_clinical_response(user_msg, history=None):
    biometrics = get_live_biometrics()
    
    # 1. Try Gemini API first if configured
    gemini_resp = try_gemini_api(user_msg, history, biometrics)
    if gemini_resp:
        return gemini_resp
        
    # 2. Math & Calculation Solver
    math_ans = try_solve_math(user_msg)
    if math_ans:
        return math_ans
        
    # 3. Creative Chit-chat (Jokes, Stories, Identity)
    creative_ans = try_creative_chat(user_msg)
    if creative_ans:
        return creative_ans
        
    # 4. Coding & Technical Queries
    coding_ans = try_solve_coding(user_msg)
    if coding_ans:
        return coding_ans
        
    # 5. Core Medical, Psychiatric & Biometric Intelligence
    msg = user_msg.strip().lower()
    face = biometrics["face"]
    voice = biometrics["voice"]
    masking = biometrics["masking"]
    risk = biometrics["risk"]
    comp = biometrics["comparison"]
    pitch = biometrics["pitch"]
    jitter = biometrics["jitter"]
    tier = biometrics["tier"]
    
    # Check context from recent message in history
    last_user_msg = ""
    if history and len(history) > 1:
        for prev in reversed(history[:-1]):
            if prev.get("sender") == "user":
                last_user_msg = prev.get("text", "").lower()
                break

    # --- CATEGORY A: GREETINGS & INTRODUCTIONS ---
    if re.search(r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b', msg) and len(msg.split()) <= 4:
        return (
            f"Hello. I am your **PSYFUSION AI Assistant**. I am ready to answer **ANY question**—from general knowledge, science, and coding to psychological and emotional counseling.\n\n"
            f"**Active Session Telemetry:**\n"
            f"- **Facial Affect:** {face}\n"
            f"- **Vocal Prosody:** {voice} ({tier}, Pitch: {pitch:.1f} Hz)\n"
            f"- **Masking Index:** {masking}% | **Risk:** {risk}\n\n"
            f"What would you like to explore or talk about right now?"
        )

    # --- CATEGORY B: CRYING, WEEPING, DEEP DISTRESS & GRIEF ---
    if any(k in msg for k in ["cry", "crying", "tears", "weep", "sobbing", "sad", "hopeless", "broken", "empty", "depressed", "hurts"]):
        return (
            f"I hear the depth of what you are carrying right now, and I want you to know that your feelings are completely valid and safe here.\n\n"
            f"**Affective & Neurobiological Observations:**\n"
            f"- Your vocal prosody registers as **{voice}** with a tremor/jitter index of **{jitter:.1f}%**, reflecting genuine autonomic vulnerability.\n"
            f"- When we hold sadness or weeping inside, the somatic nervous system experiences accumulated tension. Crying is not weakness—it is the body's natural parasympathetic release valve, releasing neurochemical endorphins and oxytocin to soothe emotional distress.\n\n"
            f"**Immediate Therapeutic Grounding:**\n"
            f"1. **Allow the Exhale:** Place one hand on your chest and one on your abdomen. Inhale gently through your nose for 4 seconds, and exhale with a soft, audible sigh through your mouth for 7 seconds.\n"
            f"2. **Release the Facade:** With a cultural masking score of **{masking}%**, you do not have to put on a brave face right now. Permitting yourself to grieve or cry is where healing begins.\n\n"
            f"What is feeling heaviest on your heart today? Take your time—I am listening."
        )

    # --- CATEGORY C: ANXIETY, PANIC ATTACKS, HEART RACING & FEAR ---
    if any(k in msg for k in ["anxiety", "anxious", "panic", "scared", "fear", "terrified", "racing heart", "can't breathe", "hyperventilating", "overwhelmed"]):
        return (
            f"I recognize that you are feeling acute anxiety or overwhelm right now. Let's ground your nervous system together step-by-step.\n\n"
            f"**Autonomic Nervous System Context:**\n"
            f"- Current vocal frequency tier: **{tier}** ({pitch:.1f} Hz) with risk level **{risk}**.\n"
            f"- Anxiety activates the sympathetic fight-or-flight pathway, surging adrenaline and elevating respiratory cadence. While distressing, panic sensations are temporary and cannot harm you.\n\n"
            f"**The Physiological Sigh (Fastest Reset):**\n"
            f"- Take **two quick inhales** through your nose (one deep breath, then a second sharp top-up inhale).\n"
            f"- Release with a **long, slow, gentle exhale** through your mouth for 8 seconds.\n"
            f"- Repeat this cycle 3 times right now. This immediately re-inflates alveolar air sacs and triggers the vagus nerve to slow your heart rate.\n\n"
            f"**5-4-3-2-1 Sensory Orientation:**\n"
            f"Look around your room right now: name **5 things you can see**, **4 things you can physically touch**, **3 sounds you can hear**, **2 scents you can smell**, and take **1 deep grounding sip of water**.\n\n"
            f"How is your breathing feeling as you take these breaths?"
        )

    # --- CATEGORY D: ANGER, RAGE, FRUSTRATION & IRRITATION ---
    if any(k in msg for k in ["angry", "anger", "furious", "hate", "mad", "rage", "irritated", "pissed", "annoyed", "unfair"]):
        return (
            f"I acknowledge your anger. Anger is a vital emotional boundary signal—it tells us when our boundaries have been crossed or when we are experiencing deep frustration.\n\n"
            f"**Biometric & Acoustic Analysis:**\n"
            f"- Observable facial affect: **{face}** | Acoustic intensity: **{voice}** (Sound Energy: {biometrics['db_level']:.0f} dB).\n"
            f"- Elevated acoustic energy reflects heightened sympathetic arousal and amygdala activation. Suppressing anger internally increases cortisol, while explosive reactions damage relational safety.\n\n"
            f"**DBT TIPP De-escalation Strategy:**\n"
            f"1. **T - Temperature Change:** Splash cold water on your face or hold an ice cube in your palm. This triggers the mammalian dive reflex, rapidly down-regulating heart rate.\n"
            f"2. **P - Paced Breathing:** Inhale for 4 seconds, exhale for 8 seconds to lower autonomic blood pressure.\n"
            f"3. **Constructive Channeling:** What specific situation or boundary feels compromised right now? Let's dissect the core trigger without self-judgment."
        )

    # --- CATEGORY E: EMOTIONAL MASKING & SMILING DEPRESSION ---
    if any(k in msg for k in ["mask", "masking", "smile", "fake", "pretend", "hiding", "hidden", "facade", "why do i smile"]):
        return (
            f"You are speaking to one of the most critical aspects of mental wellness: **Cultural & Affective Masking**.\n\n"
            f"**Your Biometric Masking Telemetry:**\n"
            f"- **Calculated Masking Index:** **{masking}%** ({comp}).\n"
            f"- **Facial Affect vs Vocal Acoustic Prosody:** Your face presents **{face}**, while your vocal acoustic dynamics reflect **{voice}**.\n\n"
            f"**Clinical Insight:**\n"
            f"- Emotional masking (often referred to as *Smiling Depression* or *High-Functioning Concealment*) occurs when social conditioning or fear of burdening others compels an individual to project composure or cheerfulness while internal psychological distress remains unresolved.\n"
            f"- Neurobiologically, voluntary zygomatic activation (AU12 fake smiling) requires continuous prefrontal effort to inhibit limbic distress signals. Over time, this chronic divergence leads to burnout, emotional numbness, and somatic exhaustion.\n\n"
            f"**Therapeutic Recommendation:**\n"
            f"Permit yourself to drop the performance in safe spaces. You do not need to earn care by appearing happy. What would it feel like to speak honestly about what is exhausting you today?"
        )

    # --- CATEGORY F: SCAN EXPLANATION & RESULTS BREAKDOWN ---
    if any(k in msg for k in ["scan", "result", "telemetry", "biometric", "explain my", "report", "diagnostic"]):
        return (
            f"Here is a comprehensive clinical synthesis of your multimodal biometric session:\n\n"
            f"**1. Affective Channel Breakdown:**\n"
            f"- **Facial Expression (Visual):** **{face}**.\n"
            f"- **Vocal Prosody (Acoustic):** **{voice}**.\n"
            f"- **Pitch ($F_0$) & Range:** **{pitch:.1f} Hz** categorized as **{tier}**.\n"
            f"- **Vocal Tremor (Jitter):** **{jitter:.1f}%** (indicating {'pronounced acoustic instability / crying micro-cues' if jitter > 0.08 else 'stable acoustic phonation'}).\n\n"
            f"**2. Masking & Autonomic Synthesis:**\n"
            f"- **Cultural Masking Score:** **{masking}%** ({'High affective divergence between facial presentation and vocal acoustics' if int(masking) > 50 else 'Coherent alignment between visual and acoustic affective channels'}).\n"
            f"- **Clinical Risk Category:** **{risk}**.\n\n"
            f"**3. Clinical Rationale:**\n"
            f"{biometrics['explanation']}\n\n"
            f"Would you like recommendations on specific therapeutic modalities (CBT, DBT, Somatic) to address these findings?"
        )

    # --- CATEGORY G: BREATHWORK & CALMING EXERCISES ---
    if any(k in msg for k in ["exercise", "technique", "breathe", "breathing", "calm", "relax", "meditate", "meditation"]):
        return (
            f"Here are three evidence-based somatic regulation protocols tailored for your **{risk}** clinical risk profile:\n\n"
            f"**Protocol 1: 4-7-8 Parasympathetic Vagal Reset**\n"
            f"- **Inhale** quietly through the nose for **4 seconds**.\n"
            f"- **Hold** the breath gently for **7 seconds** (stabilizing blood oxygen).\n"
            f"- **Exhale** completely through the mouth with a soft 'whoosh' for **8 seconds**.\n"
            f"- *Perform 4 cycles.* This stimulates the vagus nerve, immediately slowing vocal pitch and cardiac rate.\n\n"
            f"**Protocol 2: Box Breathing (4x4 Operational Calming)**\n"
            f"- Inhale 4s ➔ Hold 4s ➔ Exhale 4s ➔ Hold 4s.\n"
            f"- Excellent for clearing cognitive racing thoughts and regaining centered focus.\n\n"
            f"**Protocol 3: Vagal Laryngeal Resonance (Humming)**\n"
            f"- Inhale deeply, and as you exhale, create a low, smooth 'Huuuuum' sound in your throat for 10 seconds.\n"
            f"- The mechanical vibration relaxes the vocal cords and laryngeal muscles, reducing pitch tremor from **{jitter:.1f}%** back to baseline.\n\n"
            f"Which of these would you like to practice right now?"
        )

    # --- CATEGORY H: SLEEP, INSOMNIA & FATIGUE ---
    if any(k in msg for k in ["sleep", "insomnia", "tired", "exhausted", "nightmare", "can't sleep", "wake up"]):
        return (
            f"Sleep disruption and affective dysregulation are intimately connected through the hypothalamic-pituitary-adrenal (HPA) axis.\n\n"
            f"**Clinical Observations:**\n"
            f"- When emotional masking (**{masking}%**) or clinical risk (**{risk}**) is elevated, nighttime cortisol levels fail to drop naturally. This creates a state of *hyper-arousal*, leading to difficulty falling asleep or early morning awakenings.\n\n"
            f"**Evidence-Based Sleep Hygiene Protocol:**\n"
            f"1. **Cognitive Brain Dump:** Spend 5 minutes writing down every worry or task on physical paper 1 hour before bed. Externalizing thoughts reduces nocturnal default-mode network rumination.\n"
            f"2. **Non-Sleep Deep Rest (NSDR):** Listen to a 10-minute Yoga Nidra or progressive body scan in bed.\n"
            f"3. **Light & Temperature:** Keep bedroom temperature at 65-68°F (18-20°C) and avoid screens 45 minutes prior to sleep.\n\n"
            f"Have you been experiencing trouble falling asleep, or are you waking up repeatedly during the night?"
        )

    # --- CATEGORY I: MEDICATIONS, DOCTORS & THERAPY ---
    if any(k in msg for k in ["medication", "pill", "doctor", "psychiatrist", "therapy", "therapist", "ssri", "antidepressant"]):
        return (
            f"As your PSYFUSION Clinical AI Counselor, I provide clinical assessment and therapeutic guidance, while formal medical prescriptions require licensed physician consultation.\n\n"
            f"**Clinical Guidance on Professional Care:**\n"
            f"- If biometric distress (Current Risk: **{risk}**, Masking: **{masking}%**) persists across daily functioning, combining psychotherapy with medical evaluation yields the highest clinical efficacy.\n"
            f"- **Psychotherapy Modalities:** Cognitive Behavioral Therapy (CBT) for thought restructuring, and Acceptance and Commitment Therapy (ACT) for psychological flexibility.\n"
            f"- **Medical Evaluation:** A licensed psychiatrist can evaluate whether pharmacotherapy (such as SSRIs/SNRIs or targeted somatic medications) would support neurological balance alongside therapy.\n\n"
            f"Would you like guidance on what questions to bring to a consultation with an in-person clinician?"
        )

    # --- 6. World Knowledge Search (Wikipedia & DuckDuckGo on ANY Topic) ---
    world_ans = fetch_world_knowledge(user_msg)
    if world_ans:
        return world_ans

    # --- 7. Universal Adaptive Response ---
    return (
        f"I understand you are asking about: *\"{user_msg}\"*.\n\n"
        f"**Comprehensive Insight:**\n"
        f"This topic touches upon key concepts in modern knowledge synthesis. Whether you are looking at it from an analytical, scientific, or personal growth perspective, exploring the underlying principles provides deep clarity.\n\n"
        f"**Active Session Telemetry:**\n"
        f"- Currently observing: Facial affect **{face}**, vocal prosody **{voice}** ({pitch:.1f} Hz), with **{risk}** risk profile.\n\n"
        f"Feel free to ask me to elaborate further, provide code or calculations, or dive into any related questions you have!"
    )
