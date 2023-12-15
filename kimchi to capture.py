# 최근접 이웃 알고리즘을 사용한 '김치투캡쳐'

# ---------------- 코드 ------------------

import cv2 as cv # 최근접 이웃 알고리즘 사용, 학습을 위해
import mediapipe as mp # 손이 위치한 곳을 그려주기 위해
import datetime
import numpy as np # csv 파일을 불러오거나 배열 사용을 손쉽게 하기 위해

max_num_hands = 2 # 인식하고 싶은 손의 개수

kimchi_gesture = {1:'kimchi'} # 김치 포즈

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # 손가락의 마디마다 선을 그려줌

hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

file = np.genfromtxt('gesture.csv', delimiter=',') # csv파일을 불러옴

angle = file[:,:-1].astype(np.float32) # 각도
label = file[:,-1].astype(np.float32) # 라벨

knn = cv.ml.KNearest_create() # 최근접 이웃 알고리즘
knn.train(angle, cv.ml.ROW_SAMPLE, label) # 각도와 라벨로 학습시키기

cap = cv.VideoCapture(0) # 웹캠 켜기

while cap.isOpened(): # 카메라가 열려있을 때
    ret, img = cap.read()
    
    if not ret: # 일어 오는 데에 성공하면 아래를 실행, if not 그냥 넘어가기
        continue

    now = datetime.datetime.now()
    nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')

    # 이미지 전처리하기
    img = cv.flip(img, 1) # 좌우반전시키기
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR에서 RGB로 변경
    result = hands.process(img) # 이미지 전처리
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR) # 이미지를 출력하기 위해 RGB를 BGR로

    if result.multi_hand_landmarks is not None: # 손을 인식했다면
        for res in result.multi_hand_landmarks: # 여러개의 손 인식
            joint = np.zeros((21, 3)) # 21개의 점과 3개의 좌표(X,Y,Z)
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] # 각 점에 X,Y,Z를 저장

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1 # V1과 V2를 연결(관절 연결)
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32) # 배열로 바꾸고 float32로 바꾸기
            ret, results, neightbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0]) # 결과

            # 김치 재스쳐 인식 시
            if idx in kimchi_gesture.keys(): 
                cv.imwrite("capture " + nowDatetime_path + ".jpg", img)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 선 그어주기

    cv.imshow('Kimchi', img)

    cv.waitKey(30)
