print("=" * 50)
print("CULTURAL MASKING DETECTION")
print("=" * 50)

try:

    # Read face emotion for display
    with open("face_emotion.txt", "r", encoding="utf-8") as file:
        face = file.read().strip().lower().split(" ")[-1]
        
    # Read base face emotion for logic
    try:
        with open("face_base_emotion.txt", "r", encoding="utf-8") as file:
            face_base = file.read().strip().lower().split(" ")[-1]
    except FileNotFoundError:
        face_base = face

    # Read voice emotion
    with open("voice_emotion.txt", "r", encoding="utf-8") as file:
        voice = file.read().strip().lower().split(" ")[-1]

    # Read micro-expressions
    try:
        with open("micro_expressions.txt", "r", encoding="utf-8") as file:
            micro_data = file.read().lower()
    except FileNotFoundError:
        micro_data = ""

    print("\n------------------------------")
    print("Face Emotion  :", face.title())
    print("Voice Emotion :", voice.title())
    if micro_data:
        print("Micro-Exprs   : Active")

    # Check for masking
    score = 10
    status = "No Cultural Masking"

    if face_base != voice:
        score = 80
        status = "Cultural Masking Detected"
        
    # Deep FACS Action Unit Analysis
    fake_smile = ("happ" in face_base) and ("au12" in micro_data and "au6" not in micro_data)
    forced_calm = ("neutral" in face_base) and ("au4" in micro_data or "au5" in micro_data or "au20" in micro_data)
    
    if fake_smile:
        score = max(score, 95)
        status = "Severe Masking (Pan Am Fake Smile)"
    elif forced_calm:
        score = max(score, 88)
        status = "Masking Detected (Suppressed Tension)"

    print("Masking Score :", score, "%")
    print("Status        :", status)

    # Save masking score
    with open("masking_result.txt", "w", encoding="utf-8") as file:

        file.write(str(score))

    # Save masking status
    with open("masking_status.txt", "w", encoding="utf-8") as file:

        file.write(status)

    print("\nMasking result saved successfully!")

except FileNotFoundError:

    print("Emotion files not found.")

except Exception as e:

    print("Error:", e)

print("Program Closed")