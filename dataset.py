import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageGrab


from toolkit.devkit.helpers import labels

from torch.utils.data import DataLoader, Dataset
import json
import cv2
import sys
import random
import NeuralNets as NN
import util

YOUTUBE_GRAB_AREA = (0, 250, 1600, 1050)

class StUnstDataset(Dataset):
    def __init__(self, path, label_path, transform=None, limit=None):
        self.x_datas = []
        self.y_datas = {}
        self.path = path
        self.transform = transform

        fileDir = os.listdir(path)
        for file in fileDir:
            self.x_datas.append(file)
        
        file = open(label_path, 'r')
        c = json.load(file)
        for each in c:
            self.y_datas[each["img_path"]] = {"moveable": each["moveable"], "around":each["around"]}

        if limit:
            self.x_datas = self.x_datas[:limit]
    def __len__(self):
        return len(self.x_datas)

    def __getitem__(self, idx):
        target = self.x_datas[idx]
        
        x = Image.open(self.path + target).convert("RGB")
        if self.transform:
            x = util.transform(x)
        x = x.resize(NN.INPUT_SHAPE)
        x = np.array(x)

        #y =  np.array(self.y_datas[target]["around"])
        y = self.y_datas[target]
        # y1 = util.makeHeatmap((NN.INPUT_SHAPE[1]//2, NN.INPUT_SHAPE[0]//2), y//2)[...,None]
        y1 = util.makeHeatmap((NN.INPUT_SHAPE[1]//2, NN.INPUT_SHAPE[0]//2), np.array(y["moveable"])//2)[...,None]
        y2 = util.makeHeatmap((NN.INPUT_SHAPE[1]//2, NN.INPUT_SHAPE[0]//2), np.array(y["around"])//2)[...,None]
        y = np.concatenate((y1, y2), axis=2)
        y = y.transpose(2, 0, 1)

        return x , y

# spliting into 35 channel
class SegDataset1(Dataset):
    def __init__(self, train_path, label_edge_path, label_path,transform=None, limit=None):
        self.x_datas = []
        self.y_datas = []

        fileDir = os.listdir(train_path)
        for eachFile in fileDir:
            self.x_datas.append(train_path + eachFile)
            self.y_datas.append(label_path + eachFile)
        
        if limit:
            self.x_datas = self.x_datas[:limit]

    def __len__(self):
        return len(self.x_datas)

    def showData(self, model=None, device='cpu'):
        fig, ax = plt.subplots(3, 3)

        for i in range(3):
            randId = int(random.random() * len(self))
            trainData = self[randId]
            x = trainData[0]
            y = trainData[1]

            ax[0, i].imshow(x)
            ax[1, i].imshow(y[..., 7], cmap='gray')
            ax[2, i].imshow(y[..., 23], cmap='gray')

            if model is not None:
                model.eval()
                x = NN.img2Tensor(x, device)
                pred = model(x).cpu().detach().numpy()[0]
                res = util.segCombine(pred)
                ax[1, i].imshow(pred[..., 7], cmap='gray')
                # ax[1, i].imshow(util.convert2RGB(util.segCombine(y), labels))
                ax[2, i].imshow(util.convert2RGB(res, labels))

        plt.tight_layout()
        plt.show()


    def __getitem__(self, idx):
        x = Image.open(self.x_datas[idx]).convert("RGB").resize(NN.INPUT_SHAPE)
        x = np.array(x)

        label_img = Image.open(self.y_datas[idx]).resize(NN.OUTPUT_SHAPE, Image.NEAREST)
        w, h = label_img.size

        # split label to each channel #
        label_img = np.array(label_img)
        y = np.zeros((h, w, 35))
        for channel_n in range(35):
            y[..., channel_n] = (label_img==channel_n)*1
        ######

        return x, y

# with edge detections
class SegDataset2(Dataset):
    def __init__(self, train_path, label_edge_path, label_path,transform=None, limit=None):
        self.x_datas = []
        self.y1_datas = []
        self.y2_datas = []

        fileDir = os.listdir(train_path)
        for eachFile in fileDir:
            self.x_datas.append(train_path + eachFile)
            self.y1_datas.append(label_path + eachFile)
            self.y2_datas.append(label_edge_path + eachFile)
        
        if limit:
            self.x_datas = self.x_datas[:limit]
    def postProcessEdge(self, data, threshold=0.5):
        data = data/np.max(data)
        data[data>threshold] = 1
        data[data<=threshold] = 0
        return data

    def showData(self, model=None, device='cpu'):
        fig, ax = plt.subplots(3, 3)

        for i in range(3):
            randId = int(random.random() * len(self))
            trainData = self[randId]
            x = trainData[0]
            y = trainData[1]

            ax[0, i].imshow(x)
            ax[1, i].imshow(util.convert2RGB(y[0], labels))
            ax[2, i].imshow(y[1], cmap='gray')

            if model is not None:
                model.eval()
                x = NN.img2Tensor(x, device)
                pred = model(x)
                pred1 = pred[0].cpu().detach().numpy()[0]
                pred2 = pred[1].cpu().detach().numpy()[0]
                pred2 = self.postProcessEdge(pred2, threshold=0.1)
                ax[1, i].imshow(pred1)
                ax[2, i].imshow(pred2, cmap='gray')

        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.x_datas)

    def __getitem__(self, idx):
        x = Image.open(self.x_datas[idx]).convert("RGB").resize(NN.INPUT_SHAPE)
        x = np.array(x)

        label_img = Image.open(self.y1_datas[idx]).resize((NN.OUTPUT_SHAPE[0]//4, NN.OUTPUT_SHAPE[1]//4), Image.NEAREST)
        label_img = np.array(label_img)[...,None]

        y_img = cv2.imread(self.y2_datas[idx])
        label_img_edge = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)
        label_img_edge = cv2.resize(label_img_edge, NN.OUTPUT_SHAPE)
        label_img_edge = cv2.Canny(label_img_edge, 5, 10)
        label_img_edge = np.array(label_img_edge)[...,None]//255
        
        
        # split label to each channel #
        # y = np.zeros((h, w, 35))
        # for channel_n in range(35):
        #     y[..., channel_n] = (label_img==channel_n)*1
        ######
        return x, (label_img, label_img_edge)



def make_json(img, img_name):
    class IMG:
        def __init__(self, img_name, moveable, stable):
            self.path = img_name
            self.moveable = moveable
            self.stable = stable
    
        def toDict(self):
            self.res = {}
            self.res["img_path"] = self.path
            self.res["moveable"] = self.moveable
            self.res["around"] = self.stable

            return self.res
    

    coords = []
    fig = plt.figure()

    classImg = IMG(img_name, None, None)
    trial = []

    def onPress(event):
        sys.stdout.flush()
        if event.key == 'e':
            print(len(trial), trial)
            if len(trial) == 0:
                np_coords = np.array(coords).reshape(-1, 4)
                classImg.moveable = np_coords.tolist()
                trial.append(0)
                coords.clear()
            elif len(trial) == 1:
                classImg.stable = np.array(coords).reshape(-1, 8).tolist()
                fig.canvas.mpl_disconnect(pid)
                fig.canvas.mpl_disconnect(cid)
                print("{} - Saved!".format(img_name))
                plt.close(fig)

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix == None or iy == None:
            return
                        
        coords.append(int(ix))
        coords.append(int(iy))

        print("Coords: {}, len: {}".format(coords, len(coords)))

    ax = fig.add_subplot()
    ax.imshow(img)

    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+1500+100")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    pid = fig.canvas.mpl_connect('key_press_event', onPress)

    plt.show()

    return classImg.toDict()
    

def file_make_json(path, res, _from=0):
    IMGS = []

    img_dir = os.listdir(path)
    count = 0
    for file in img_dir[_from:]:
        img = Image.open(path + "/" + file).convert("RGB")
        IMGS.append(make_json(img, file))

        count += 1

        if count % 5 == 0:
            res_f = open(res, 'w')
            res_f.write(json.dumps(IMGS))
            res_f.close()
    res_f = open(res, 'w')
    res_f.write(json.dumps(IMGS))
    res_f.close()

# click event rec ~5 * 2 
def real_time_make_json(path, res, count=500, start=1):
    IMGS = []

    for each in range(1, count+1):
        fileName = util.getFileNum(each) + ".jpg"
        img = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA).resize(NN.INPUT_SHAPE))
        IMGS.append(make_json(img, fileName))

        img = Image.fromarray(img)
        img.save(path + fileName)

    res_f = open(res, 'w')
    res_f.write(json.dumps(IMGS))
    res_f.close()

if __name__=="__main__":
    file_make_json("./train_data/img/", "./train_data/label.json")
    # real_time_make_json("./", "./train_data/label.json", count=3)
    