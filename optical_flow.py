import numpy as np
import cv2
import time
from collections import deque

# OpticalFlow.appendTrackPoint(...) -> OpticalFlow.trace(...)
class OpticalFlow():
    def __init__(self, trackPoints, trackingLen, maxTrack=100):
        self.TOTAL_TRACK_POINTS_N = trackPoints
        self.TRACKING_LENGTH = trackingLen
        self.RAND_COLOR = deque(maxlen=self.TOTAL_TRACK_POINTS_N)
        # self.BRIGHT_COLOR = np.linspace([0, 0, 0], [255, 0, 0], self.TOTAL_TRACK_POINTS_N)
        
        self.MAX_TRACK_LEN = maxTrack
        self.termcriteria =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.img_size = None
        self.pts = deque(maxlen=self.TOTAL_TRACK_POINTS_N)

    def setRandomColor(self):
        self.RAND_COLOR = np.random.randint(0, 255, (self.TOTAL_TRACK_POINTS_N, 3))

    # Append tracking points to self.pts
    # points: points to append
    # GoodFeature = Use good Feature. If, (True, image, number of points)
    def appendTrackPoint(self, point=None, GoodFeature=(False, None, 0)):
        useG, prevImg, count = GoodFeature[0], GoodFeature[1], GoodFeature[2]

        if useG:
            prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
            point = cv2.goodFeaturesToTrack(prevImg, count, 0.01, 10)

        point = np.asarray(point, dtype=np.float32)
        point = point.reshape((-1, 1, 2))

        for i in range(len(point)):
            self.pts.append(deque(point[i], maxlen=self.TRACKING_LENGTH))
            self.RAND_COLOR.append(np.random.randint(0, 255, 3))

    def drawTrack(self, frame):
        for i in range(len(self.pts)):
            for each_track in range(1, len(self.pts[i])):
                p1, p2 = np.array(self.pts[i][each_track - 1], dtype=np.int32), np.array(self.pts[i][each_track], dtype=np.int32)
                
                thickness = int(np.sqrt(each_track))  +1  
                cv2.line(frame, p1, p2, color=self.RAND_COLOR[i].tolist(), thickness=thickness)
                # cv2.line(frame, np.array(self.pts[i][each_track - 1], dtype=np.int32), np.array(self.pts[i][each_track], dtype=np.int32), color=(0, 0, 255), thickness=thickness)

    def check(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if p1.any() < 0:
            return False
        if sum((p1-p2)**2) > self.MAX_TRACK_LEN**2:
            return False
        return True

    # Trace points of self.pts
    # prevImg: previous frame of image
    # curImg: current frame of image
    # drawOn: which img you want to draw. If none, draw on curImg
    def trace(self, prevImg, curImg, drawOn=None):
        if prevImg is None:
            prevImg = curImg

        prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)

        prevPt = np.array([self.pts[i][-1] for i in range(len(self.pts))])

        # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
        nextPt, _, _ = cv2.calcOpticalFlowPyrLK(prevImg, gray, prevPt, None, 
        criteria=self.termcriteria, winSize=(15, 15), maxLevel=3)
        
        idx = 0
        for i, n in enumerate(nextPt):
            n_nx,n_ny = n.ravel()
            if self.check(self.pts[idx][-1], [n_nx, n_ny]):
                self.pts[idx].append([n_nx, n_ny])
                idx += 1

            else:
                del self.pts[idx]
                del self.RAND_COLOR[idx]

        if drawOn is None: 
            drawOn = curImg.copy()
        self.drawTrack(drawOn)

        return drawOn
    
    def getPoints(self):
        res = []
        for row in self.pts:
            res.append(list(row))
        return res


def main():
    OptFlw = OpticalFlow(10, 10)
    prevImg = None
    addedTime = time.time()
    while True:
        ret,frame = cap.read()

        # OptFlw.appendTrackPoint(GoodFeature=(True, gray, 5))
        curTime = time.time()
        if prevImg is None:
            prevImg = frame
            OptFlw.appendTrackPoint(GoodFeature=(True, frame, 5))
        curTime = time.time()
        if curTime - addedTime > 1:
            OptFlw.appendTrackPoint(GoodFeature=(True, frame, 1))
            addedTime = curTime

        result = OptFlw.trace(prevImg, frame)

        cv2.imshow('OpticalFlow-LK', result)
        prevImg = frame

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

    time.sleep(1) # wait for 1 sec

    main()
