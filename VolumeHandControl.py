import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Constants
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
HAND_RANGE = (50, 300)  # Min and max distance for hand gesture
VOL_RANGE = (-65, 0)    # Min and max volume levels
SMOOTHING_FACTOR = 5    # Determines the smoothness of volume transition

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    pTime = 0
    detector = htm.HandDetector(detectionCon=0.7)

    # Initialize volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    minVol, maxVol = VOL_RANGE
    vol, volBar, volPer = 0, 400, 0
    currentVol = volume.GetMasterVolumeLevel()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)  # Flip the image
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # Hand tracking and volume control logic
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point between thumb and index finger

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Convert the hand range to volume range
            vol = np.interp(length, HAND_RANGE, [minVol, maxVol])
            volBar = np.interp(length, HAND_RANGE, [400, 150])
            volPer = np.interp(length, HAND_RANGE, [0, 100])
            
            # Smooth the volume transition
            currentVol = currentVol - (currentVol - vol) / SMOOTHING_FACTOR
            volume.SetMasterVolumeLevel(currentVol, None)

            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Display the image and volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()