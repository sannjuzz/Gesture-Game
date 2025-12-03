import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game parameters
score = 0
paddle_width = 100
paddle_height = 20
ball_radius = 15
ball_speed = 5

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width, _ = frame.shape

# Ball position
ball_x = random.randint(ball_radius, width-ball_radius)
ball_y = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_x = width//2  # Default paddle position

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            # Use wrist x-coordinate for horizontal paddle movement
            hand_x = int(handLms.landmark[mp_hands.HandLandmark.WRIST].x * width)

    # Draw paddle
    paddle_x = hand_x - paddle_width//2
    paddle_y = height - 50
    cv2.rectangle(frame, (paddle_x, paddle_y), 
                  (paddle_x + paddle_width, paddle_y + paddle_height), 
                  (255, 0, 0), -1)

    # Move ball
    ball_y += ball_speed
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Collision detection
    if paddle_y < ball_y + ball_radius < paddle_y + paddle_height and paddle_x < ball_x < paddle_x + paddle_width:
        score += 1
        ball_y = 0
        ball_x = random.randint(ball_radius, width-ball_radius)

    # Game over
    if ball_y > height:
        cv2.putText(frame, "GAME OVER", (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.imshow("Gesture Game", frame)
        cv2.waitKey(3000)
        break

    # Show score
    cv2.putText(frame, f"Score: {score}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
