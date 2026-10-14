print("=" * 45)
print("EMOTION COMPARISON")
print("=" * 45)

try:

    # Read face emotion for display
    with open("face_emotion.txt", "r", encoding="utf-8") as file:
        face = file.read().strip().split(" ")[-1]
        
    # Read base face emotion for logic
    try:
        with open("face_base_emotion.txt", "r", encoding="utf-8") as file:
            face_base = file.read().strip().split(" ")[-1]
    except FileNotFoundError:
        face_base = face

    # Read voice emotion
    with open("voice_emotion.txt", "r", encoding="utf-8") as file:
        voice = file.read().strip().split(" ")[-1]

    print("\n----------------------------")
    print("Face Emotion  :", face)
    print("Voice Emotion :", voice)

    if face_base.lower() == voice.lower():

        result = "Emotion Matched"

    else:

        result = "Emotion Mismatch"

    print("Result :", result)

    # Save the result
    with open("comparison_result.txt", "w", encoding="utf-8") as file:
        file.write(result)

    print("Comparison saved successfully!")

except FileNotFoundError:

    print("Emotion files not found.")

except Exception as e:

    print("Error:", e)

print("Program Closed")