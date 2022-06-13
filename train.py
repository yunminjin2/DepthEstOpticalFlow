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
import numpy as np
import time
import os

import util
import NeuralNets as NN
from dataset import StUnstDataset
import random

LEARNING_RATE =  0.01
EPOCHS = 64
BATCH_SIZE = 2

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_PATH = "./train_data/img/"
LABEL_PATH = "./train_data/label.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)


def showData(Dataset, model1=None):
    fig, ax = plt.subplots(2, 5)

    for i in range(5):
        randId = int(random.random() * len(Dataset))
        trainData = Dataset[randId]
        img = trainData[0]

        if model1== None:
            y = trainData[1]
        
            heatImg = util.cvt2Heatmap(y, img)
            out = (np.clip(y, 0, 1) * 255).astype(np.uint8)

        else:
            model.eval()
            out = NN.run_model_img(img, model, device)
            
            heatImg = util.cvt2Heatmap(out, img)
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)


        ax[0, i].imshow(heatImg)
        ax[1, i].imshow(out, cmap='gray')
        ax[0, i].set_title("train {}".format(i))

    plt.tight_layout()
    plt.show()



# TRAIN MODEL    
def train(model, train_loader, epochs, batch_size, lr=0.01):
    criterion = torch.nn.MSELoss().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for singleEpoch in range(1, epochs+1):
        train_loss = 0.0
        total = 0
        model.train()

        for data, target in tqdm(train_loader, desc="Epoch: {}".format(singleEpoch)):
            data = data.permute(0, 3, 1, 2)
            data = data.cuda().float()
            
            target = target.cuda().float()


            # cleaer the gradients all optimized variables
            optimizer.zero_grad()

            # forward process
            out = model(data)[0]

            loss = criterion(out, target)
            loss.backward()
            
            optimizer.step()
            total += batch_size
            train_loss += loss.item()

        # calculate average loss
        train_loss = train_loss/total
        train_losses.append(train_loss)
        # show
        if singleEpoch % 10 == 0:
            showData(trainDataset, model)
            NN.saveModel(model, "./weights/model_1.pt")

        print('Epoch: {} \tTraining Loss: {:.5f}'.format(singleEpoch, train_loss))
    
    return model


if __name__=="__main__":

    trainDataset = StUnstDataset(TRAIN_PATH, LABEL_PATH)
    train_loader =  DataLoader(dataset = trainDataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NN.AroundModel()
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    model = NN.load_model("./weights/preTrained.pt", model, parallel=False)

    model = train(model, train_loader, EPOCHS, BATCH_SIZE)
    
    NN.saveModel(model, "./weights/model_1.pt")
    showData(trainDataset, model=model)
