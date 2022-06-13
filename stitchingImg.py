import cv2
import os
import numpy as np
import imutils
import time
import matplotlib
import matplotlib.pyplot as plt

from optical_flow import OpticalFlow

VIDEO_FOLDER = "./video_data/"
VIDEO = 9
FRAMES = 40
SIZE = (480, 320)
VIDEO_FILE = "{}.mp4".format(VIDEO)
FONT=cv2.FONT_HERSHEY_SIMPLEX

cap = None
def setCamera():
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    
    time.sleep(1) # wait for 1 sec
    return capture

def readVideo(path):
    frames = []
    cap = cv2.VideoCapture(VIDEO_FOLDER + VIDEO_FILE)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, SIZE)
        frame = frame[40:]
        frames.append(frame)
    cap.release()

    frameIds = np.linspace(0, frames.shape[0]-1, n, dtype=np.int32)
    return frames[frameIds]
  
def calibOptPoints(points):
    diff = points[:, -1] - points[:, 0]
    diff_s = diff[:, 0]**2 + diff[:,1]**2
    min_id = np.argmin(diff_s)
    max_id = np.argmax(diff_s)

    diff_s -= diff_s[min_id]
    diff_s /= diff_s[max_id]

    D = diff_s.argsort()
    return points[D], diff_s[D]
    


def trackOptPoints(frames):
    OptFlw = OpticalFlow(100, FRAMES)
    prevImg = frames[0]
    canvas = np.zeros(frames[0].shape)
    
    gray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    ids = 0
    
    OptFlw.appendTrackPoint(points)
    for eachFrame in frames:
        res = OptFlw.trace(prevImg, eachFrame, drawOn=canvas)
        cv2.imwrite("./video_data/{}/{}.jpg".format(VIDEO, ids), canvas)
        prevImg = eachFrame
        ids+=1
    return np.array(OptFlw.getPoints())


def draw_tracks(frame, points, colors, until):
    for i in range(len(points)):
        for each_track in range(1, len(points[i])):
            if each_track > until:
                break
            p1, p2 = np.array(points[i][each_track - 1], dtype=np.int32), np.array(points[i][each_track], dtype=np.int32)

            thickness = int(np.sqrt(each_track)) + 1  
            itsColor = int(colors[i])
            itsColor = tuple([itsColor, itsColor, itsColor])
            
            cv2.line(frame, p1, p2, color=itsColor, thickness=thickness)


if __name__=="__main__":
    # frames = readVideo(VIDEO_FOLDER + VIDEO_FILE)
    capture = setCamera()


    OptFlw = OpticalFlow(30, 10)

    ret, frame = capture.read()
    prevImg = frame
    OptFlw.appendTrackPoint(GoodFeature=(True, frame, 20))
    addedTime = time.time()

    while True: 
        frame = np.zeros((SIZE[1], SIZE[0], 3))
        ret, frame = capture.read()


        curTime = time.time()
        if curTime - addedTime > 1:
            OptFlw.appendTrackPoint(GoodFeature=(True, frame, 2))
            addedTime = curTime
        result = OptFlw.trace(prevImg, frame)
        prevImg = frame

        cv2.imshow('OpticalFlow-LK', result)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    capture.release()