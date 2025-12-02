import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector
import pyttsx3  # <--- NEW LIBRARY

# --- CONFIGURATION ---
IMAGE_FOLDER = 'student_images'
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "attendance_db"
LATE_THRESHOLD = "09:15:00"

# --- AUDIO SETUP (THE VOICE) ---
engine = pyttsx3.init()

def speak(text):
    """Helper function to make the computer talk"""
    engine.say(text)
    engine.runAndWait()

# --- STEP 0: SUBJECT MENU SELECTION ---
print("--------------------------------")
print("   CLASS ATTENDANCE SYSTEM      ")
print("--------------------------------")

subjects_list = ["MATHS", "SCIENCE", "PYTHON", "DBMS", "DDS"]
print("Select the Subject for this Class:")
for index, subject in enumerate(subjects_list):
    print(f"{index + 1}. {subject}")

while True:
    try:
        choice = int(input(f"Enter choice (1-{len(subjects_list)}): "))
        if 1 <= choice <= len(subjects_list):
            current_subject = subjects_list[choice - 1]
            break
        else:
            print("❌ Invalid number.")
    except ValueError:
        print("❌ Please enter a valid number.")

print(f"\n✅ SYSTEM LOCKED: Marking [{current_subject}]")
speak(f"System active for {current_subject} class") # Audio confirmation
print("--------------------------------")


# --- STEP 1: DATABASE CONNECTION ---
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )

# --- STEP 2: LOAD IMAGES AND TRAIN AI ---
print("1. Loading Images and Training AI...")
images = []
classNames = []
myList = os.listdir(IMAGE_FOLDER)

for cl in myList:
    curImg = cv2.imread(f'{IMAGE_FOLDER}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) 

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"Warning: No face found in one of the training images.")
    return encodeList

knownEncodings = findEncodings(images)
print(f"2. Training Complete. {len(knownEncodings)} faces learned.")
speak("Database loaded. Starting Camera.")

# --- STEP 3: DATABASE LOGGING FUNCTION ---
def mark_attendance_sql(name, subject):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    # STATUS CHECK
    if current_time > LATE_THRESHOLD:
        status = "LATE 🔴"
        audio_msg = f"Welcome {name}, you are late."
    else:
        status = "ON TIME 🟢"
        audio_msg = f"Welcome {name}."

    # CHECK DUPLICATE
    sql_check = "SELECT * FROM attendance_logs WHERE student_name = %s AND date_marked = %s AND subject = %s"
    cursor.execute(sql_check, (name, current_date, subject))
    result = cursor.fetchall()

    if not result:
        # INSERT
        sql_insert = "INSERT INTO attendance_logs (student_name, time_marked, date_marked, subject, status) VALUES (%s, %s, %s, %s, %s)"
        val = (name, current_time, current_date, subject, status)
        cursor.execute(sql_insert, val)
        conn.commit()
        
        print(f"✅ MARKED: {name} | {status}")
        speak(audio_msg) # <--- COMPUTER SPEAKS HERE
        
    else:
        # OPTIONAL: You can make it speak for duplicates too, but it might get annoying.
        # speak(f"{name}, already marked.") 
        pass 
        
    cursor.close()
    conn.close()

# --- STEP 4: WEBCAM LOOP ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(knownEncodings, encodeFace)
        faceDis = face_recognition.face_distance(knownEncodings, encodeFace)
        
        matchIndex = np.argmin(faceDis)

        # Scale up face location
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if matches[matchIndex] and faceDis[matchIndex] < 0.55:
            name = classNames[matchIndex].upper()
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"{name}", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            
            mark_attendance_sql(name, current_subject)
            
        else:
            # UNKNOWN - RED BOX
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "UNKNOWN", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            
            # Note: We don't put speak("Unknown") here because it will 
            # scream "UNKNOWN UNKNOWN UNKNOWN" 30 times a second!

    cv2.imshow('Class Attendance System', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
