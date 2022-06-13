import cv2
from PIL import ImageGrab, Image
from cv2 import INPAINT_NS

import torch
import numpy as np
import argparse

import NeuralNets as NN
import util
from optical_flow import OpticalFlow
import time

parser = argparse.ArgumentParser(description="활용법")

parser.add_argument('--mode',  help='camera or screen')
parser.add_argument('--optFlow', type=bool, default=True, help='use optical flow')


FONT=cv2.FONT_HERSHEY_SIMPLEX
YOUTUBE_GRAB_AREA = (0, 250, 1600, 1050)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


def setCamera():
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    
    return capture



if __name__ == "__main__":
    args = parser.parse_args()
    model = NN.FocusNN2()
    model = NN.load_model("./weights/model2.ckpt", model)

    capture = None
    if args.mode == 'camera':
        capture = setCamera()
    if args.optFlow:
        optFlw = OpticalFlow(100 , 10)
        prevImg = None

    while True:
        frame = np.zeros((NN.INPUT_SHAPE[1], NN.INPUT_SHAPE[0],3))
        if args.mode == 'camera':
            ret, frame = capture.read()
        elif args.mode == 'screen':
            screen = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, NN.INPUT_SHAPE)
        result_img = frame
        startTime = time.time()
        out = NN.run_model_img(frame, model, device)
        out = util.normalize(out)
        out = cv2.resize((240, 320))

        # if args.optFlow:
        #     optFlw.appendTrackPoint(point=out)
        #     result_img = optFlw.trace(prevImg, frame, drawOn=result_img)
        #     prevImg = frame
        
        cv2.putText(result_img, "FPS: {:.1f}".format(1/(time.time()-startTime)), (70, 50), FONT, 1, (255, 0, 0), 2)

        # result_img = util.draw_points(frame, out, size=9)
        cv2.imshow("result", result_img)
        cv2.imshow("result2", out)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
