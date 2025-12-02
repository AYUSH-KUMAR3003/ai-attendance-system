import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector
import pandas as pd
import plotly.express as px
from PIL import Image
from io import BytesIO
import edge_tts
import asyncio

# Page site configuration for site portal name
st.set_page_config(
    page_title="QUANTUM AI ATTENDANCE SYSTEM",
    layout="wide",
    page_icon="logo.png",
    initial_sidebar_state="expanded"
)

# global variables or the   ---
IMAGE_FOLDER = 'student_images'
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "attendance_db"
LATE_THRESHOLD = "09:15:00"

#  these are the funactions  which provides (Database, AI & Voice)

def get_db_connection():
    """Creates a connection to the MySQL database."""
    try:
        return mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
        )
    except Exception as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

@st.cache_resource
def load_encodings():
    """Loads student images and trains the AI model once at startup."""
    st.info("⏳ Loading AI Model & Student Data... Please wait.")
    images = []
    classNames = []
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        return [], []
    
    myList = os.listdir(IMAGE_FOLDER)
    for cl in myList:
        curImg = cv2.imread(f'{IMAGE_FOLDER}/{cl}')
        if curImg is None: continue
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
        
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img_rgb)[0]
            encodeList.append(encode)
        except IndexError: pass
            
    st.success(f"✅ AI Ready! Loaded {len(encodeList)} faces.")
    return encodeList, classNames

# funcations using for voice 
async def generate_voice_file(text, voice, filename):
    """Helper function to run async TTS generation"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def speak_welcome(text):
    """Generates human-like audio using Edge TTS directly in Python."""
    output_file = "welcome.mp3"
    selected_voice = "en-IN-PrabhatNeural" 
    
    try:
        asyncio.run(generate_voice_file(text, selected_voice, output_file))
        
        # Play the audio in Streamlit
        if os.path.exists(output_file):
            audio_file = open(output_file, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3', start_time=0, autoplay=True)
            audio_file.close()
            os.remove(output_file) # Clean up
    except Exception as e:
        st.warning(f"Audio Error (Voice skipped): {e}")

def mark_attendance_db(name, subject):
    """Inserts attendance record into DB and returns a Message for Voice."""
    conn = get_db_connection()
    if not conn: return False, "Database Error"
    
    cursor = conn.cursor()
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')
    
    status = "LATE 🔴" if current_time > LATE_THRESHOLD else "ON TIME 🟢"
    
    # Check duplicate
    cursor.execute(
        "SELECT * FROM attendance_logs WHERE student_name = %s AND date_marked = %s AND subject = %s", 
        (name, current_date, subject)
    )
    
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO attendance_logs (student_name, time_marked, date_marked, subject, status) VALUES (%s, %s, %s, %s, %s)",
            (name, current_time, current_date, subject, status)
        )
        conn.commit()
        st.toast(f"✅ Marked: {name} ({status})", icon="✅")
        st.balloons()
        
        # Voice Message for Success
        clean_status = status.replace("🔴", "").replace("🟢", "")
        msg = f"Welcome {name}. You are marked {clean_status}."
        marked = True
    else:
        st.toast(f"⚠ {name} already marked for {subject}", icon="⚠")
        # Voice Message for Duplicate
        msg = f"{name}, your attendance is already marked."
        marked = False
        
    cursor.close()
    conn.close()
    return marked, msg

@st.cache_data
def get_dashboard_data():
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM attendance_logs", conn)
    conn.close()
    return df

def save_uploaded_file(uploaded_file, student_name):
    if not os.path.exists(IMAGE_FOLDER): os.makedirs(IMAGE_FOLDER)
    filename = f"{student_name.replace(' ', '_')}.jpg"
    file_path = os.path.join(IMAGE_FOLDER, filename)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return True

# Load AI model immediately
knownEncodings, classNames = load_encodings()

#  MAIN APP LAYOUT
col1, col2 = st.columns([1, 5]) 
with col1:
    if os.path.exists("logo.png"): st.image("logo.png", width=150)
with col2:
    st.title("QUANTUM AI ATTENDANCE SYSTEM")

# Create three main tabs
tab_take_attendance, tab_dashboard, tab_register = st.tabs([
    "📸 Take Attendance", "📊 Dashboard & Reports", "➕ Register Student"
])

#  TAB 1: TAKE ATTENDANCE (Camera)
with tab_take_attendance:
    st.header("Mark Attendance")
    subjects = ["MATHS", "SCIENCE", "PYTHON", "DBMS", "DDS"]
    selected_subject_cam = st.selectbox("Select Subject for this Session:", subjects, key="cam_subj")
    st.divider()
    
    col_cam, col_info = st.columns([2, 1])
    with col_cam:
        img_buffer = st.camera_input("Take Photo to Mark Present", key="camera")
    
    with col_info:
        st.info(f"Class: *{selected_subject_cam}*")
        st.write(f"Late Threshold: {LATE_THRESHOLD}")
        
        # Audio Placeholder (Keeps UI stable while sound plays)
        audio_placeholder = st.empty()
        
        if img_buffer:
            if not knownEncodings:
                st.error("AI not trained. No students registered.")
            else:
                img_array = np.array(Image.open(img_buffer))
                faces_cur = face_recognition.face_locations(img_array)
                encodes_cur = face_recognition.face_encodings(img_array, faces_cur)
                
                if not faces_cur: st.warning("No face detected.")
                
                recognized_any = False
                for encodeFace, faceLoc in zip(encodes_cur, faces_cur):
                    matches = face_recognition.compare_faces(knownEncodings, encodeFace)
                    faceDis = face_recognition.face_distance(knownEncodings, encodeFace)
                    matchIndex = np.argmin(faceDis)
                    
                    if matches[matchIndex] and faceDis[matchIndex] < 0.50:
                        name = classNames[matchIndex].upper()
                        
                        # Mark DB and Get Voice Message
                        is_new, voice_msg = mark_attendance_db(name, selected_subject_cam)
                        
                        st.success(f"Identified: *{name}*")
                        
                        # Play Sound
                        with audio_placeholder:
                            speak_welcome(voice_msg)
                            
                        recognized_any = True
                    else:
                        st.error("Unknown Face Detected.")
                
                if recognized_any:
                    st.cache_data.clear()

#  TAB 2: DASHBOARD & REPORTS
with tab_dashboard:
    st.header("Attendance Analytics")
    df = get_dashboard_data()
    
    if df.empty:
        st.warning("No attendance data available.")
    else:
        df['time_marked'] = df['time_marked'].astype(str).str.replace('0 days ', '')
        df['date_marked'] = pd.to_datetime(df['date_marked']).dt.date
        if 'subject' not in df.columns: df['subject'] = 'General'
        if 'status' not in df.columns: df['status'] = 'Unknown'

        st.sidebar.header("Dashboard Filters")
        all_dates = sorted(df['date_marked'].unique(), reverse=True)
        sel_date = st.sidebar.selectbox("Select Date", all_dates, key="dash_date")
        subjects_on_date = df[df['date_marked'] == sel_date]['subject'].unique()
        
        if len(subjects_on_date) > 0:
            sel_subject = st.sidebar.selectbox("Select Subject", subjects_on_date, key="dash_subj")
            df_daily = df[(df['date_marked'] == sel_date) & (df['subject'] == sel_subject)]
            df_history = df[df['subject'] == sel_subject]

            st.subheader(f"📅 Daily Report: {sel_subject} on {sel_date}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Present", len(df_daily))
            m2.metric("On Time", len(df_daily[df_daily['status'].str.contains('ON')]))
            m3.metric("Late", len(df_daily[df_daily['status'].str.contains('LATE')]), delta_color="inverse")
            
            c1, c2 = st.columns(2)
            with c1:
                if not df_daily.empty:
                    fig = px.bar(df_daily.sort_values('time_marked'), x='student_name', y='time_marked', color='status', 
                                 color_discrete_map={'ON TIME 🟢':'green', 'LATE 🔴':'red'}, title="Arrival Times")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(df_daily[['student_name', 'time_marked', 'status']], use_container_width=True, height=300)
            
            st.divider()

            st.subheader(f"📉 Overall Attendance Status ({sel_subject})")
            total_classes = df_history['date_marked'].nunique()
            
            if total_classes > 0:
                att_counts = df_history.groupby('student_name')['date_marked'].nunique().reset_index()
                att_counts.columns = ['Student Name', 'Days Present']
                att_counts['Percentage'] = (att_counts['Days Present'] / total_classes) * 100
                att_counts['Status'] = att_counts['Percentage'].apply(lambda x: "🔴 CRITICAL" if x < 75 else "🟢 SAFE")
                
                st.dataframe(att_counts.style.format({"Percentage": "{:.1f}%"}).map(
                    lambda v: 'color: red; font-weight: bold' if 'CRITICAL' in v else 'color: green', subset=['Status']),
                    use_container_width=True)
            else:
                st.info("Need more historical data for percentage calculation.")
        else:
            st.warning(f"No classes found on {sel_date}.")
            
        if st.button("🔄 Refresh Dashboard Data"):
            st.cache_data.clear()
            st.rerun()

#  TAB 3: REGISTER STUDENT
with tab_register:
    st.header("👤 New Student Registration")
    with st.form("reg_form", clear_on_submit=True):
        new_name = st.text_input("Full Name")
        up_file = st.file_uploader("Upload Face Image (JPG/PNG)", type=['jpg','png'])
        submitted = st.form_submit_button("Save Profile", type="primary")
        
        if submitted and new_name and up_file:
            save_uploaded_file(up_file, new_name)
            st.success(f"✅ Registered {new_name}!")
            st.info("💡 Please restart the app to apply changes.")
            st.cache_resource.clear()
