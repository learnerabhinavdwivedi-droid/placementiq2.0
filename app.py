import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import joblib
import time
import base64
import os
from openai import OpenAI

# ---------------- 1. CONFIG & SESSION STATE ----------------
st.set_page_config(page_title="PlacementIQ Pro2", layout="wide", page_icon="🚀")

# LOCKING VARIABLES SO YOUR SUB-PAGES DON'T GO BLANK
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "target_role" not in st.session_state:
    st.session_state.target_role = "Software Engineer"
if "extracted_skills" not in st.session_state:
    st.session_state.extracted_skills = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- 2. LOAD 3D LOGO ----------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "Gemini_Generated_Image_6dai4k6dai4k6dai-removebg-preview.png")
img_base64 = get_base64_image(image_path)

# ---------------- 3. GLOBAL STYLES ----------------
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    body { background-color: #f6f8fc; }
    div.stButton > button {
        background: linear-gradient(90deg,#2563eb,#1e40af);
        color: white; border-radius: 12px; height: 3em; font-size: 17px;
        border: none; transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        background: linear-gradient(90deg,#1e40af,#1e3a8a);
    }
    .hero {
        background:#f4f7ff; color: #1f2a44; padding: 32px; border-radius: 24px;
        text-align: center; margin-bottom: 25px; box-shadow: 0 12px 28px rgba(0,0,0,0.2);
    }
    .logo-container { display: flex; justify-content: center; align-items: center; perspective: 1000px; margin-bottom: 20px; }
    .logo-3d { width: 250px; transition: transform 1.2s; transform-style: preserve-3d; filter: drop-shadow(0 10px 20px rgba(0,0,0,0.4)); }
    .logo-container:hover .logo-3d { transform: rotateY(360deg) scale(1.1); }
</style>
""", unsafe_allow_html=True)

# ---------------- 4. SIDEBAR LOGO & AI BOT ----------------
if img_base64:
    st.sidebar.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{img_base64}" class="logo-3d"></div>', unsafe_allow_html=True)

try:
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    client = None

with st.sidebar:
    st.markdown("### 🤖 Placement Assistant")
    with st.popover("💬 Ask AI Assistant", use_container_width=True):
        chat_container = st.container(height=350)
        with chat_container:
            for message in st.session_state.messages:
                st.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.chat_input("How can I help you prepare?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                st.chat_message("user").markdown(prompt)
                if client:
                    try:
                        with st.chat_message("assistant", avatar="✨"):
                            stream = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                                stream=True,
                            )
                            response = st.write_stream(stream)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception:
                        st.error("API Error. Please check your connection.")
                else:
                    st.error("Groq API key missing. Please check secrets.")
            st.rerun()
            
    st.markdown("---")
    st.info("👆 Use the menu above to check your GitHub, LinkedIn, and Mock Tests!")

# ---------------- 5. MAIN HEADER ----------------
st.markdown("""
<div class="hero">
    <div style="font-size: 36px; font-weight: 700;">PlacementIQ Pro2</div>
    <div style="font-size: 18px;">AI Profile & Placement Readiness Analyzer</div>
    <div style="font-size: 14px; opacity: 0.85;">Enterprise-grade placement intelligence</div>
</div>
""", unsafe_allow_html=True)

# ---------------- 6. INPUT METRICS ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Academic & Skill Metrics")
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
    internship = st.selectbox("Internship Experience", [0, 1])
    projects = st.slider("Number of Projects", 0, 5, 2)
    communication = st.slider("Communication Skill (1-10)", 1, 10, 6)
    dsa_score = st.slider("DSA / Coding Skill (1-10)", 1, 10, 5)
    hackathons = st.slider("Hackathons / Certifications", 0, 5, 1)

with col2:
    st.markdown("### 2. Target Role & Resume")
    
    role_options = [
        "Software Engineer", "Python Developer", "Data Analyst", "ML Engineer", 
        "Backend Developer", "Frontend Developer", "Full Stack Developer", 
        "Cybersecurity Analyst", "DevOps Engineer", "Cloud Engineer", 
        "Amazon SDE", "Google SWE", "Infosys Graduate Engineer", "Custom"
    ]
    
    role = st.selectbox("Select Target Role", role_options)
    st.session_state.target_role = role
    
    job_description = st.text_area("Job Description Details", "Requires strong programming, problem solving, and communication skills.")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    
    extracted_text = ""
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() + "\n"
            st.success("✅ Resume parsed!")
        except Exception:
            st.error("Could not read this PDF. Please paste text manually.")

    if extracted_text:
        st.session_state.resume_text = extracted_text

    resume_text = st.text_area("Parsed Resume Text", st.session_state.resume_text, height=150)
    st.session_state.resume_text = resume_text

# ---------------- 7. ANALYZE BUTTON & DASHBOARD ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Analyze Profile Readiness", use_container_width=True):

    if not resume_text.strip():
        st.error("Please upload or paste your resume text to proceed.")
        st.stop()
    
    with st.spinner("Analyzing profile and computing insights..."):
        time.sleep(1.5) 
        
        SKILLS_DB = {
            "python": ["python"], "sql": ["sql", "mysql", "postgresql"],
            "machine learning": ["machine learning", "ml", "tensorflow"],
            "data analysis": ["data analysis", "pandas", "numpy"],
            "git": ["git", "github"], "java": ["java", "spring"],
            "react": ["react", "nextjs"], "docker": ["docker", "containers"],
            "c++": ["c++", "cpp"], "javascript": ["javascript", "js", "node"]
        }
        
        resume_clean = resume_text.lower()
        job_clean = job_description.lower()
        
        resume_skills = [s for s, kw in SKILLS_DB.items() if any(k in resume_clean for k in kw)]
        job_skills = [s for s, kw in SKILLS_DB.items() if any(k in job_clean for k in kw)]
        st.session_state.extracted_skills = resume_skills

        word_count = len(resume_text.split())
        numbers_found = sum(c.isdigit() for c in resume_text)
        resume_quality = min(word_count / 200, 1) * 4 + min(len(resume_skills) / 6, 1) * 4 + min(numbers_found / 10, 1) * 2
        resume_quality = round(resume_quality, 2)

        if len(job_skills) > 0:
            match_percentage = round((len(set(resume_skills).intersection(job_skills)) / len(job_skills)) * 100, 2)
        else:
            match_percentage = min(len(resume_skills) * 15, 100) 

        missing_skills = list(set(job_skills) - set(resume_skills))

        try:
            model = joblib.load("placement_model.pkl")
            input_features = np.array([[cgpa, internship, communication, match_percentage]])
            raw_prob = model.predict_proba(input_features)[0][1]
            probability = raw_prob * 60
        except Exception:
            probability = (cgpa * 4) + (internship * 10) + (match_percentage * 0.2) + (projects * 2)
        
        probability += (dsa_score * 0.8) + (hackathons * 0.6)
        probability = round(max(5, min(probability, 98)), 1)

        # UI RESULTS
        st.divider()
        st.markdown(f"""
        <div style="background: linear-gradient(90deg,#2563eb,#1e40af); color: white; padding: 14px 22px; border-radius: 14px; font-size: 28px; font-weight: 600; text-align: center; margin-bottom: 20px;">
            Results: {probability}% Readiness
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(int(probability))

        colA, colB, colC = st.columns(3)
        with colA: st.metric("Skill Match", f"{match_percentage}%")
        with colB: st.metric("Detected Skills", len(resume_skills))
        with colC: st.metric("Missing JD Skills", len(missing_skills))
        
        st.write(f"**Resume Quality Score:** {resume_quality}/10")
        
        st.subheader("🧐 Why this score?")
        reasons = []
        if cgpa >= 8: reasons.append(f"Strong CGPA (+{round(cgpa*1.5,1)}%)")
        elif cgpa < 6.5: reasons.append("Low CGPA (-8%)")
        if internship == 1: reasons.append("Internship experience (+10%)")
        else: reasons.append("No internship (-10%)")
        if match_percentage >= 70: reasons.append(f"Good skill match (+{round(match_percentage/5,1)}%)")
        elif match_percentage < 40: reasons.append("Low skill match (-10%)")
        if communication >= 7: reasons.append("Good communication (+6%)")
        if dsa_score >= 7: reasons.append("Strong DSA skills (+8%)")
        if hackathons >= 2: reasons.append("Certifications/Hackathons (+5%)")
        
        if len(reasons) == 0:
            st.write("Balanced profile with no major strengths or weaknesses.")
        else:
            for r in reasons:
                st.write("•", r)

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Skill Profile Breakdown")
            skills_df = pd.DataFrame({
                "Category": ["CGPA", "Projects", "Skill Match", "Comm."],
                "Score": [cgpa*10, projects*20, match_percentage, communication*10]
            })
            st.bar_chart(skills_df.set_index("Category"))
            
        with right_col:
            st.subheader("Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"• {skill.title()}")
            else:
                st.success("No major skill gaps detected.")

        st.divider()
        st.markdown("<h3 style='text-align:center;'>Placement Insights</h3>", unsafe_allow_html=True)
        
        col_str, col_weak = st.columns(2)
        box_style = "padding:18px; border-radius:14px; border:1px solid #e6eaf2;"
        
        with col_str:
            strengths = []
            if cgpa >= 8: strengths.append("Strong academic performance")
            if internship == 1: strengths.append("Internship experience adds real-world exposure")
            if len(resume_skills) >= 4: strengths.append("Good skill coverage for the role")
            if dsa_score >= 7: strengths.append("Strong problem-solving ability")
            if not strengths: strengths.append("Keep pushing forward 🚀")
            
            html_str = "<br>".join([f"• {s}" for s in strengths])
            st.markdown(f"<div style='{box_style}'><h3>Strengths</h3>{html_str}</div>", unsafe_allow_html=True)
            
        with col_weak:
            weak = []
            if cgpa < 7: weak.append("Low CGPA may affect shortlist chances")
            if internship == 0: weak.append("No internship experience")
            if len(resume_skills) <= 2: weak.append("Limited technical skills detected")
            if match_percentage < 50: weak.append("Resume not aligned with job role")
            if not weak: weak.append("No major weak areas — keep growing 🔥")
            
            html_weak = "<br>".join([f"• {w}" for w in weak])
            st.markdown(f"<div style='{box_style}'><h3>Weak Areas</h3>{html_weak}</div>", unsafe_allow_html=True)

        st.subheader("🎯 Exact Next Steps")
        if match_percentage < 60: st.write("• Add missing skills from job description")
        if internship == 0: st.write("• Try 1–2 internships or open-source projects")
        if dsa_score < 7: st.write("• Solve 150+ DSA problems on LeetCode")
        if resume_quality < 6: st.write("• Add quantified achievements (numbers, impact)")
        st.write("• Do mock interviews weekly\n• Build 1 real-world project")

        st.subheader("🏢 Company Readiness")
        company_targets = {
            "Amazon SDE": 1.2, "Google SWE": 1.25,
            "Infosys": 0.9, "Startup Intern": 0.8
        }
        for company, difficulty in company_targets.items():
            company_score = round(max(5, min(probability / difficulty, 95)), 1)
            if company_score >= 85: st.success(f"{company} → {company_score}% Ready")
            elif company_score >= 65: st.info(f"{company} → {company_score}% Almost Ready")
            else: st.warning(f"{company} → {company_score}% Needs Improvement")

st.markdown("---")

st.markdown("<p style='text-align:center; font-size:14px;'>PlacementIQ Pro2 • HackWave 2026</p>", unsafe_allow_html=True)
