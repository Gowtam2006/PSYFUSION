import os
import sys
import subprocess
from datetime import datetime

# Auto-install fpdf2 if missing
try:
    from fpdf import FPDF
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "fpdf2"], check=True)
    from fpdf import FPDF

print("=" * 50)
print("COUNSELING REPORT GENERATOR (PDF)")
print("=" * 50)

def read_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip().split(" ")[-1]
    return "N/A"

try:
    face = read_file("face_emotion.txt")
    voice = read_file("voice_emotion.txt")
    masking = read_file("masking_result.txt")
    risk = read_file("risk_result.txt")
    explanation = read_file("explanation.txt")

    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%H:%M:%S")
    
    session_start_time_str = "N/A"
    session_duration_str = "N/A"
    
    if os.path.exists("session_start.txt"):
        with open("session_start.txt", "r", encoding="utf-8") as f:
            start_ts = float(f.read().strip())
        start_dt = datetime.fromtimestamp(start_ts)
        session_start_time_str = start_dt.strftime("%H:%M:%S")
        duration = now - start_dt
        session_duration_str = str(duration).split('.')[0]
        
    micro_expr_text = "N/A"
    if os.path.exists("micro_expressions.txt"):
        with open("micro_expressions.txt", "r", encoding="utf-8") as f:
            micro_expr_text = f.read().strip()
            
    transcript_text = "None recorded."
    if os.path.exists("voice_transcript.txt"):
        with open("voice_transcript.txt", "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
            if not transcript_text: transcript_text = "None recorded."
            
    min_pitch, max_pitch, peak_vol = 0.0, 0.0, 0.0
    if os.path.exists("voice_stats.txt"):
        pitches, vols = [], []
        with open("voice_stats.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    p, v = float(parts[0]), float(parts[1])
                    if p > 0: pitches.append(p)
                    vols.append(v)
        if pitches:
            min_pitch, max_pitch = min(pitches), max(pitches)
        if vols:
            peak_vol = max(vols)
            
    consultant_summary = "Data unavailable."
    consultant_recs = "Data unavailable."
    consultant_next = "Data unavailable."
    if os.path.exists("consultant_report.txt"):
        with open("consultant_report.txt", "r", encoding="utf-8") as f:
            content = f.read()
            parts = content.split('\n\n')
            for part in parts:
                if part.startswith("SUMMARY\n"):
                    consultant_summary = part.replace("SUMMARY\n", "")
                elif part.startswith("RECOMMENDATIONS\n"):
                    consultant_recs = part.replace("RECOMMENDATIONS\n", "")
                elif part.startswith("NEXT_SESSION\n"):
                    consultant_next = part.replace("NEXT_SESSION\n", "")

    class CyberReport(FPDF):
        def header(self):
            # Dark Background
            self.set_fill_color(4, 6, 13)
            self.rect(0, 0, 210, 297, 'F')
            
            # Cyber Borders
            self.set_draw_color(0, 243, 255)
            self.set_line_width(0.3)
            self.rect(10, 10, 190, 277)
            
            # Watermark
            self.set_font("Courier", "B", 60)
            self.set_text_color(15, 20, 35) 
            with self.rotation(45, x=105, y=148):
                self.text(40, 160, "PSYFUSION")
            
            # Header Title
            self.set_font("Courier", "B", 24)
            self.set_text_color(255, 0, 255) # Magenta
            self.cell(0, 15, "P S Y F U S I O N", ln=True, align="C")
            
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(0, 243, 255) # Cyan
            self.cell(0, 10, "CONFIDENTIAL CLINICAL DIAGNOSTIC REPORT", ln=True, align="C")
            self.ln(5)

        def footer(self):
            self.set_y(-20)
            self.set_font("Courier", "", 9)
            self.set_text_color(0, 243, 255)
            self.cell(0, 5, "================================================================", ln=True, align="C")
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, f"SYSTEM: PSYFUSION_CORE_V4 | PAGE {self.page_no()}", align="C")

    # Create PDF instance
    pdf = CyberReport()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()
    
    # Metadata Section
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(0, 243, 255)
    pdf.cell(50, 6, "Date of Scan:", ln=0)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 6, date_str, ln=1)
    
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(0, 243, 255)
    pdf.cell(50, 6, "Time of Scan:", ln=0)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 6, time_str, ln=1)
    
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(0, 243, 255)
    pdf.cell(50, 6, "Session Start:", ln=0)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 6, session_start_time_str, ln=1)
    
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(0, 243, 255)
    pdf.cell(50, 6, "Session Duration:", ln=0)
    pdf.set_font("Courier", "", 11)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 6, session_duration_str, ln=1)
    
    pdf.ln(8)

    # Helper function for sections
    def add_section(title, content_dict=None, text=None):
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(255, 0, 255) # Magenta Title
        pdf.cell(0, 10, title, ln=True)
        pdf.set_draw_color(0, 243, 255)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_text_color(220, 220, 220) 
        
        if content_dict:
            for k, v in content_dict.items():
                pdf.set_font("Courier", "B", 11)
                pdf.set_text_color(0, 243, 255) # Cyan Key
                pdf.cell(70, 7, k, ln=0)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(255, 255, 255) # White Value
                pdf.cell(0, 7, v, ln=1)
        if text:
            pdf.ln(2)
            pdf.set_font("Courier", "", 11)
            pdf.set_text_color(200, 210, 230)
            pdf.multi_cell(0, 6, text)
        pdf.ln(5)

    add_section("1. PRIMARY MULTIMODAL OBSERVATIONS", {
        "Dominant Face Emotion:": face.upper(),
        "Dominant Vocal Emotion:": voice.upper()
    })
    
    add_section("2. AI FACE & VOCAL ANALYSIS", {
        "Micro-Expressions & Pose:": "",
    }, text=micro_expr_text + f"\n\nVoice Pitch Range: {min_pitch:.1f}Hz - {max_pitch:.1f}Hz\nPeak Volume: {peak_vol:.1f}")
    
    add_section("3. VOICE TRANSCRIPT", text=transcript_text)
    
    add_section("4. CLINICAL METRICS", {
        "Cultural Masking Index:": f"{masking}%",
        "Overall Clinical Risk Level:": risk.upper()
    })
    
    # Minor explanatory note for masking
    pdf.set_font("Courier", "I", 9)
    pdf.set_text_color(100, 100, 120)
    pdf.multi_cell(0, 5, "[NOTE: Culturally Masked Emotion correlates with incongruence between vocal frequency variance and facial muscle topology.]")
    pdf.ln(5)

    add_section("5. AI DIAGNOSTIC RATIONALE", text=explanation)
    
    add_section("6. END-OF-SESSION ANALYSIS", text=consultant_summary)
    
    add_section("7. CONSULTANT RECOMMENDATIONS", {
        "Next Session Timing:": consultant_next
    }, text=consultant_recs)

    # Save PDF
    pdf.output("report.pdf")
    print("PDF Report saved successfully!")

except Exception as e:
    print("Error:", e)

print("Program Closed")