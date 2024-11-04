import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('detect/yolov8s_traffic_sign_detection/weights/best.pt')
classes = model.names

cap = cv2.VideoCapture(0)

signs_detected = []
# Define a dictionary for sign index and their respective warning messages
signs = {
    0: "donel_kavsak!",
    1: "dur_stop!",
    2: "durak!",
    3: "duraklamak_park_yasak!",
    4: "gidis_donus!",
    5: "girisi_olmayan_yol!",
    6: "ileri_mecburi!",
    7: "ileri_sag_mecburi!",
    8: "ileri_sol_mecburi!",
    9: "ileriden_saga!",
    10: "ileriden_sola!",
    11: "lamba_kirmizi!",
    12: "lamba_sari!",
    13: "lamba_yesil!",
    14: "park_edilebilir!",
    15: "park_engelli!",
    16: "park_yasak!",
    17: "park_yasak_mavi!",
    18: "sag_mecburi!",
    19: "saga_donulmez!",
    20: "sagdan_gidin!",
    21: "sol_mecburi!",
    22: "sola_donulmez!",
    23: "soldan_gidin!",
    24: "yaya_gecidi!"
}

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the grayscale frame to a specific size if needed
        target_size = (640, 480)  # Örnek boyutlar
        resized_frame = cv2.resize(gray_frame, target_size, interpolation=cv2.INTER_LINEAR)

        # YOLOv8 modeline giriş olarak RGB görüntü gerekebilir, bu yüzden gri tonlamalı görüntüyü RGB'ye çevirelim
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)

        # Run YOLOv8 inference on the resized RGB frame
        result = model.predict(frame_rgb, conf=0.6)

        # Visualize the results on the frame
        annotated_frame = result[0].plot()

        for r in result:
            if r.boxes:
                box = r.boxes[0]
                class_id = int(box.cls)
                
                # Check if detected class is in the signs dictionary
                if class_id in signs:
                    text = signs[class_id]
                    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                    # Tespit edilen işareti ve mesajı listeye ekle
                    signs_detected.append((class_id, text))
                    print(f"Detected: {text}")
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Testing", annotated_frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Son tespit edilen işaretleri ve uyarı mesajlarını göster
print("Signs Detected:")
for sign in signs_detected:
    print(sign)

