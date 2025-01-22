from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound  # Install this library using `pip install playsound`

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold and frame check values
thresh = 0.25
frame_check = 20

# Dlib face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")  # Update path if needed

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)
flag = 0
alert_played = False  # To avoid continuous sound playback

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # Converting to NumPy array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play alert sound if not already played
                if not alert_played:
                    playsound(r"C:\Users\DELL\Downloads\censorbeep.mp3")  # Replace with the path to your alert sound file
                    alert_played = True
        else:
            flag = 0
            alert_played = False  # Reset alert flag

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
