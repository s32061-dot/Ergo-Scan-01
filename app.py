import cv2
import mediapipe as mp
import numpy as np
import math
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# เรียกใช้ MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_vertical_angle(p_top, p_bottom):
    """คำนวณมุมเทียบกับเส้นแนวดิ่ง (Vertical Axis) ตามหลักกายศาสตร์"""
    delta_x = abs(p_top[0] - p_bottom[0])
    delta_y = abs(p_top[1] - p_bottom[1])
    angle_rad = math.atan2(delta_x, delta_y)
    return math.degrees(angle_rad)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"})
    
    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # --- เพิ่มส่วนนี้เพื่อแก้ปัญหาเว็บค้าง ---
    # ย่อขนาดภาพให้ไม่เกิน 800px เพื่อป้องกัน RAM เต็มบน Render (Free Tier)
    max_dimension = 800
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    # ------------------------------------
    
    # ประมวลผลภาพ
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return jsonify({"error": "ตรวจไม่พบร่างกาย กรุณาถ่ายภาพด้านข้างให้เห็น หู ไหล่ และสะโพก"})

    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark
    
    # ดึงพิกัดจุดสำคัญ (ใช้ฝั่งขวาเป็นหลักสำหรับการถ่ายด้านข้าง)
    ear = [lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w, lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h]
    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
    hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h]

    # คำนวณมุม
    neck_angle = calculate_vertical_angle(ear, shoulder)
    back_angle = calculate_vertical_angle(shoulder, hip)
    
    # วาดเส้นไกด์ไลน์ (สีแดงคือแนวคอ, สีเขียวคือแนวหลัง)
    cv2.line(image, (int(shoulder[0]), int(shoulder[1])), (int(ear[0]), int(ear[1])), (0, 0, 255), 4)
    cv2.line(image, (int(hip[0]), int(hip[1])), (int(shoulder[0]), int(shoulder[1])), (0, 255, 0), 4)
    
    # แปลงภาพเป็น Base64 เพื่อส่งไปแสดงบนหน้าเว็บ
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "image": img_str,
        "neck_angle": int(neck_angle),
        "back_angle": int(back_angle)
    })

if __name__ == '__main__':
    app.run(debug=True)