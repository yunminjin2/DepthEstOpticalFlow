from re import U
from unittest import result
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
FONT=cv2.FONT_HERSHEY_SIMPLEX
YOUTUBE_GRAB_AREA = (0, 250, 1600, 1050)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

opticalFLow = None

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

    model = NN.AroundModel()
    model = NN.load_model("./weights/model_1.pt", model)

    capture = None
    if args.mode == 'camera':
        capture = setCamera()


    while True:
        frame = np.zeros((NN.INPUT_SHAPE[1], NN.INPUT_SHAPE[0],3))

        if args.mode == 'camera':
            ret, frame = capture.read()
        elif args.mode == 'screen':
            screen = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, NN.INPUT_SHAPE)
        half_frame = cv2.resize(frame, (NN.INPUT_SHAPE[0]//2, NN.INPUT_SHAPE[1]//2))
        startTime = time.time()

        data = NN.img2Tensor(frame, device)
        out = model(data)[0].cpu().detach().numpy()[0]

        
        heatMap1 = util.cvt2Heatmap(out[0], superimposeOn=half_frame)
        heatMap1 = cv2.cvtColor(heatMap1, cv2.COLOR_RGB2BGR)
        cv2.putText(heatMap1, "FPS: {:.1f}".format(1/(time.time()-startTime)), (70, 50), FONT, 1, (255, 0, 0), 2)
        cv2.imshow("heatmap", heatMap1)

        # heatMap2 = util.cvt2Heatmap(out[1], superimposeOn=half_frame)
        # heatMap2 = cv2.cvtColor(heatMap2, cv2.COLOR_RGB2BGR)
        # cv2.imshow("heatmap2", heatMap2)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
