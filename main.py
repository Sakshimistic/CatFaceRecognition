"""
Online Cat Mediapipe Program - MeowCV


A openCV program that detects faces and displays Tiktok cats.

"""

import cv2 # type: ignore
import mediapipe as mp # type: ignore
import os


face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

cam = cv2.VideoCapture(0)

# Thresholds 
eye_opening_threshold = 0.025
mouth_open_threshold = 0.03
squinting_threshold = 0.018
# Smile threshold
smile_threshold = 0.05  # adjust based on testing


def cat_smirk(face_landmark_points):
    # Check eyes squinted
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    eyes_squinted = eye_squint < squinting_threshold

    # Check mouth wide (smile)
    left_corner = face_landmark_points.landmark[61]
    right_corner = face_landmark_points.landmark[291]
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_height = abs(top_lip.y - bottom_lip.y)
    mouth_wide = mouth_width > 0.06 and mouth_height < 0.05

    # Smirk triggers only if both conditions are met
    return eyes_squinted and mouth_wide

def cat_smile(face_landmark_points):
    left_corner = face_landmark_points.landmark[61]
    right_corner = face_landmark_points.landmark[291]
    
    # Top and bottom lips
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_height = abs(top_lip.y - bottom_lip.y)

    # Only consider as smile if mouth is wide but not too open
    return mouth_width > 0.06 and mouth_height < 0.05
def cat_smirk(face_landmark_points):
    # Check eyes squinted
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    eyes_squinted = eye_squint < squinting_threshold

    # Check mouth wide (smile)
    left_corner = face_landmark_points.landmark[61]
    right_corner = face_landmark_points.landmark[291]
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_height = abs(top_lip.y - bottom_lip.y)
    mouth_wide = mouth_width > 0.06 and mouth_height < 0.05

    # Smirk triggers only if both conditions are met
    return eyes_squinted and mouth_wide

def cat_shock(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0

    return eye_opening > eye_opening_threshold


def cat_tongue(face_landmark_points):
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)

    return mouth_open > mouth_open_threshold

def cat_glare(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_squint = (
        abs(l_top.y - l_bot.y) +
        abs(r_top.y - r_bot.y)
    ) / 2.0

    return eye_squint < squinting_threshold



def main():
    while True:
        ret, image = cam.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        height, width, depth = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh.process(rgb_image)
        face_landmark_points = processed_image.multi_face_landmarks

        cat_image = "assets/cat-shock.jpeg"
        if face_landmark_points:

            face_landmark_points = face_landmark_points[0]
           
            
        
        # Draw small circle at the landmark
                # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # # Optional: draw landmark index number
        #         cv2.putText(image, str(idx), (x, y-2),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            if cat_tongue(face_landmark_points):
                cat_image = "assets/cat-tongue.jpeg"
            elif cat_shock(face_landmark_points):
                cat_image = "assets/cat-shock.jpeg"
            elif cat_smirk(face_landmark_points):
                cat_image = "assets/cat-glare.jpeg"   # <-- your smirk cat image
            elif cat_glare(face_landmark_points):
                cat_image = "assets/cat-glare.jpeg"
            elif cat_smile(face_landmark_points):
                cat_image = "assets/larry.jpeg"
            else:
                cat_image = "assets/cat-smile.png"


        

            height, width = image.shape[:2]
            for idx, lm in enumerate(face_landmark_points.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                        # Draw small circle at the landmark
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # # Optional: draw landmark index number
                # cv2.putText(image, str(idx), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 100), 1)
            
        
        cv2.imshow('Face Detection', image)

        # Cat Display
        cat = cv2.imread(cat_image)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
            cv2.imshow("Cat Image", cat)
        else:
            blank = image * 0
            cv2.putText(blank, f"Missing: {cat_image}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cat Image", blank)


        key = cv2.waitKey(1)
        if key == 27:
            break


    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
