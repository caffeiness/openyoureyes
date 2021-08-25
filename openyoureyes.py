import os,sys
import cv2
import time
# ビープ音の再生
import winsound
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play


cap = cv2.VideoCapture(1)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
start_time = time.time()
end_time = time.time()


def upload(find):
    if find:
        winsound.PlaySound("metalgearBGM.mp3", winsound.SND_ALIAS)
    else:
        winsound.PlaySound("cat.wav", winsound.SND_ALIAS)

while True:
    ret, rgb = cap.read()

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

    if len(faces) == 1:
        x, y, w, h = faces[0, :]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 処理高速化のために顔の上半分を検出対象範囲とする
        eyes_gray = gray[y : y + int(h/2), x : x + w]
        eyes = eye_cascade.detectMultiScale(eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(rgb, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 1)

        if len(eyes) == 0:
            end_time = time.time()
            #print(end_time-start_time)
            if end_time-start_time >= 5:
                cv2.putText(rgb,"Sleepy eyes. Wake up!",(10,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)
                #upload(True)
                #winsound.PlaySound("metalgearBGM.mp3")
                winsound.Beep(1000, 100) # 1000Hzのビープを100ms再生
        elif len(eyes) > 0:
            start_time = end_time 

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()