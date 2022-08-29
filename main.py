import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

#standard distance formula 
def distance(x1, y1, x2, y2):
    return ((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)

#we do a little bit of math
def determine_finger_closed(mcp_base, tip, wrist):
    if distance(mcp_base.x,mcp_base.y, wrist.x, wrist.y) > distance(tip.x,tip.y, wrist.x, wrist.y):
        return True
    else:
        return False

def determine_rps(hand_landmarks):
    
    thumb = determine_finger_closed(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    index = determine_finger_closed(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    middle = determine_finger_closed(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    ring = determine_finger_closed(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])
    pinky = determine_finger_closed(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])


    if index and middle and ring and pinky:
        # rock 
        print("rock")
    elif not index and not middle and ring and pinky:
        # scissor 
        print("sisor")
    elif not index and not middle and not ring and not pinky:
        #all open
        print("paper")
    #elif index and not middle and ring and pinky:
        #uhhhhh
    #    print("frick off") 
    elif not index and middle and ring and not pinky:
        print("SpoderMann")
    #elif not index and middle and not ring and not pinky:
    #    print("three")
    
    else:
        print("unknown")



# For webcam input:
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        determine_rps(hand_landmarks)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()