import cv2
from PIL import Image
import imageio
from tqdm import tqdm

import torch
import torch.nn.init
from torch.utils.data import DataLoader
from torchvision import transforms
import torchsummary

import json
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time
import os

from PIL import ImageGrab
import util
import NeuralNets as NN
from dataset import SegDataset1, SegDataset2
import random

import torchsummary
from loss import CrossEntropy2d


from toolkit.devkit.helpers import labels

LEARNING_RATE =  0.01
EPOCHS = 128
BATCH_SIZE = 2
YOUTUBE_GRAB_AREA = (0, 250, 1600, 1050)
FONT = cv2.FONT_HERSHEY_PLAIN

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_PATH = "./train_data/training/image_2/"
LABEL_EDGE_PATH = "./train_data/training/semantic_rgb/"
LABEL_PATH = "./train_data/training/semantic/"

# pretrain_MySegModel
# train_MySegModel
# pretrain_SegModel35
# train_SegModel35
PRE_MODEL_PATH = "./weights/pretrain_SegModel35.pt"
MODEL_PATH = "./weights/train_SegModel35.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Dataset and Model
trainDataset = SegDataset1(TRAIN_PATH, LABEL_EDGE_PATH, LABEL_PATH)
train_loader =  DataLoader(dataset = trainDataset, batch_size=BATCH_SIZE, shuffle=True)
model = NN.SegModel()

def screenDetect(model, threshold=150):
    while True:
        screen = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, NN.INPUT_SHAPE)

        half_frame = cv2.resize(frame, (NN.INPUT_SHAPE[0]//2, NN.INPUT_SHAPE[1]//2))
        startTime = time.time()

        data = NN.img2Tensor(frame, device)
        out = model(data).cpu().detach().numpy()[0]

        res = util.segCombine(out)
        # ax[1, i].imshow(pred[..., 7], cmap='gray')
        cv2.imshow("segmap", out[...,7])
        cv2.imshow("eachChn", util.convert2RGB(res, labels))

        

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

def train(model, trainDataset, dataLoader):
    # criterion = torch.nn.CrossEntropyLoss().to(device) 
    criterion1 =  torch.nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []

    for singleEpoch in range(1, EPOCHS+1):
        train_loss = 0.0
        total = 0
        model.train()

        for data, target in tqdm(dataLoader, desc="Epoch: {}".format(singleEpoch)):
            data = data.permute(0, 3, 1, 2)
            data = data.cuda().float()
            # for segment1
            target = target.cuda().float()
            # for segment2
            # target_label = target[0].cuda().float()
            # target_edge = target[1].cuda().float()

            # cleaer the gradients all optimized variables
            optimizer.zero_grad()

            # forward process
            out = model(data)

            # for segment1
            loss= criterion1(out, target)


            # for segment2
            # loss1 = criterion1(out[0], target_label)
            # loss2 = criterion1(out[1], target_edge)   
            # loss = loss1 + loss2
            


            loss.backward()

            optimizer.step()
            total += BATCH_SIZE
            train_loss += loss.item()

        # calculate average loss
        train_loss = train_loss/total
        train_losses.append(train_loss)

        print('Epoch: {} \tTraining Loss: {:.5f}'.format(singleEpoch, train_loss))
        if singleEpoch%10==0:
            NN.saveModel(model, MODEL_PATH)
        # trainDataset.showData(model)


if __name__=="__main__":
    # trainDataset.showData()

    model = torch.nn.DataParallel(model)
    model = NN.load_model(MODEL_PATH, model, parallel=False)
    model.to(device)
    # torchsummary.torchsummary.summary(model, batch_size=BATCH_SIZE,device=device,input_size=(3, 480, 640))
    screenDetect(model)

    # train(model, trainDataset, train_loader)
    # NN.saveModel(model, MODEL_PATH)

    # trainDataset.showData(model)