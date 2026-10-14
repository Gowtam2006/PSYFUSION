import os
import tkinter as tk
from tkinter import messagebox
import random

print("=" * 50)
print("AI EXPLANATION MODULE")
print("=" * 50)

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

    # Read risk level
    with open("risk_result.txt", "r", encoding="utf-8") as file:
        risk = file.read().strip().split(" ")[-1]

    import requests
    
    # Generate explanation via Ollama
    ollama_url = "http://localhost:11434/api/generate"
    prompt = f"""You are a senior clinical psychiatrist and neurobiologist analyzing a patient. 
Patient Data:
- Facial Emotion: {face} (Base: {face_base})
- Voice Emotion: {voice}
- Risk Level: {risk}

Provide a DETAILED, professional explanation of the patient's current psychological state. 
CRITICAL REQUIREMENT: You MUST base your analysis on established medical resources. Reference specific neurobiological mechanisms (e.g., amygdala activation, autonomic nervous system arousal, dopamine/serotonin pathways) and standard psychiatric frameworks (like DSM-5 criteria for mood or anxiety disorders). 
Discuss how their facial emotion and voice emotion interact, what this means for their overall risk, and provide a deep psychological rationale. Keep the output to 2-3 paragraphs. Do not use asterisks or markdown."""

    try:
        response = requests.post(ollama_url, json={"model": "phi3", "prompt": prompt, "stream": False}, timeout=30)
        if response.status_code == 200:
            explanation = response.json().get("response", "").strip()
        else:
            raise ValueError("Ollama HTTP Error")
    except Exception as e:
        # Fallback to static text
        if face_base.lower() == voice.lower():
            explanation = (
                "The face emotion and voice emotion are similar. "
                "No significant emotional mismatch was detected."
            )
        else:
            explanation = (
                "The face emotion and voice emotion are different. "
                "This may indicate possible cultural masking or hidden emotions."
            )

    print("\n" + "=" * 50)
    print("AI EXPLANATION")
    print("=" * 50)

    print("Face Emotion  :", face)
    print("Voice Emotion :", voice)
    print("Risk Level    :", risk)

    print("\nExplanation:")
    print(explanation)

    # Save explanation
    with open("explanation.txt", "w", encoding="utf-8") as file:
        file.write(explanation)

    print("\nExplanation saved successfully!")
    
    # -----------------------------
    # TKINTER GAME GUI (5-Question Quiz)
    # -----------------------------
    
    psychology_words = [
        ("EMPATHY", "Understanding another's feelings"),
        ("COGNITION", "Mental action of acquiring knowledge"),
        ("SYNAPSE", "Junction between two nerve cells"),
        ("NEUROSIS", "Mild mental illness involving stress"),
        ("SCHEMA", "Cognitive framework or concept"),
        ("PLACEBO", "Harmless pill used for psychological benefit"),
        ("EGO", "Sense of self-esteem or self-importance"),
        ("TRAUMA", "Deeply distressing experience"),
        ("PHOBIA", "Extreme or irrational fear"),
        ("STIMULUS", "A thing that rouses activity"),
        ("AFFECT", "The experience of feeling or emotion"),
        ("DOPAMINE", "Neurotransmitter linked to pleasure")
    ]
    
    quiz_words = random.sample(psychology_words, 5)
    current_q = 0
    score = 0

    root = tk.Tk()
    root.title("PSYFUSION - Explanation & Therapeutic Game")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int(screen_width/2 - 350)
    y = int(screen_height/2 - 275)
    root.geometry(f"700x550+{x}+{y}")
    root.configure(bg="#F0F4F8")

    # Explanation Section (Always visible)
    tk.Label(root, text="🧠 AI EXPLANATION", font=("Times New Roman", 18, "bold"), bg="#F0F4F8", fg="#5C7A99").pack(pady=10)
    tk.Label(root, text=explanation, font=("Times New Roman", 14), bg="#F0F4F8", fg="#333333", wraplength=600, justify="center").pack(pady=5)
    tk.Frame(root, height=2, bd=1, relief="sunken", bg="#D0D8E0").pack(fill="x", padx=40, pady=10)

    # Game Section (Dynamic)
    game_frame = tk.Frame(root, bg="#F0F4F8")
    game_frame.pack(fill="both", expand=True)

    q_label = tk.Label(game_frame, text="Question 1/5", font=("Times New Roman", 16, "bold"), bg="#F0F4F8", fg="#9060FF")
    q_label.pack(pady=5)
    tk.Label(game_frame, text="Select the correct psychology word from the jumbled letters:", font=("Times New Roman", 12), bg="#F0F4F8", fg="#5C7A99").pack(pady=5)
    
    jumbled_label = tk.Label(game_frame, text="", font=("Courier", 24, "bold"), bg="#F0F4F8", fg="#333333")
    jumbled_label.pack(pady=5)
    
    hint_label = tk.Label(game_frame, text="", font=("Times New Roman", 12, "italic"), bg="#F0F4F8", fg="#888888")
    hint_label.pack(pady=5)

    options_frame = tk.Frame(game_frame, bg="#F0F4F8")
    options_frame.pack(pady=10)
    
    option_buttons = []
    for i in range(4):
        btn = tk.Button(options_frame, text="", font=("Times New Roman", 12, "bold"), bg="#9060FF", fg="white", cursor="hand2", width=14)
        btn.grid(row=i//2, column=i%2, padx=15, pady=10)
        option_buttons.append(btn)
        
    result_lbl = tk.Label(game_frame, text="", font=("Times New Roman", 14, "bold"), bg="#F0F4F8")
    result_lbl.pack(pady=5)

    def load_question():
        global current_q
        if current_q >= 5:
            show_final_score()
            return
            
        word, hint = quiz_words[current_q]
        
        jumbled = list(word)
        random.shuffle(jumbled)
        while "".join(jumbled) == word and len(word) > 1:
            random.shuffle(jumbled)
        jumbled_word = "".join(jumbled)
        
        options = [word]
        other_words = [w[0] for w in psychology_words if w[0] != word]
        options.extend(random.sample(other_words, 3))
        random.shuffle(options)
        
        q_label.config(text=f"Question {current_q+1}/5")
        jumbled_label.config(text=jumbled_word)
        hint_label.config(text=f"Hint: {hint}")
        
        for i, btn in enumerate(option_buttons):
            btn.config(text=options[i], state="normal", bg="#9060FF", command=lambda o=options[i]: check_word(o))
            
        result_lbl.config(text="")

    def check_word(selected_word):
        global current_q, score
        correct_word = quiz_words[current_q][0]
        
        for btn in option_buttons:
            btn.config(state="disabled")
            if btn.cget("text") == correct_word:
                btn.config(bg="#4CAF50")
            elif btn.cget("text") == selected_word:
                btn.config(bg="#F44336")
                
        if selected_word == correct_word:
            result_lbl.config(text="✨ Correct! ✨", fg="#4CAF50")
            score += 1
        else:
            result_lbl.config(text=f"Incorrect! It was {correct_word}", fg="#F44336")
            
        current_q += 1
        root.after(1500, load_question)
        
    def show_final_score():
        game_frame.destroy()
        
        res_frame = tk.Frame(root, bg="#F0F4F8")
        res_frame.pack(pady=20, fill="both", expand=True)
        
        tk.Label(res_frame, text="Quiz Complete!", font=("Times New Roman", 24, "bold"), bg="#F0F4F8", fg="#9060FF").pack(pady=20)
        
        score_lbl = tk.Label(res_frame, text="0/5", font=("Courier", 48, "bold"), bg="#F0F4F8", fg="#333333")
        score_lbl.pack(pady=15)
        
        def animate_score(current):
            score_lbl.config(text=f"{current}/5")
            if current < score:
                root.after(300, animate_score, current + 1)
            else:
                if score == 5:
                    msg = "Perfect! Amazing psychology knowledge! 🌟"
                elif score >= 3:
                    msg = "Great job! Keep learning! 👍"
                else:
                    msg = "Good effort! Practice makes perfect! 💪"
                tk.Label(res_frame, text=msg, font=("Times New Roman", 16, "italic"), bg="#F0F4F8", fg="#5C7A99").pack(pady=10)
                tk.Button(res_frame, text="Close", command=root.destroy, font=("Times New Roman", 14, "bold"), bg="#9060FF", fg="white", padx=20).pack(pady=20)
                
        root.after(500, animate_score, 0)

    import sys
    if "--no-gui" not in sys.argv:
        # Start the quiz
        load_question()
        root.mainloop()

except FileNotFoundError:
    print("Required files not found.")
except Exception as e:
    print("Error:", e)

print("Program Closed")