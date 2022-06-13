import numpy as np
from scipy.fft import dst
import NeuralNets as NN

import random
import torch
import torchvision.transforms.functional as FT

from scipy import ndimage

import cv2

COLORS = [(0,0,255), (0, 233, 233), (0, 233, 233), (255, 0, 0), (255, 0, 0)]


#https://guru.tistory.com/73
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size)==2 
            self.output_size = output_size

    def __call__(self, sample, point):
        image = sample
        new_point = point

        xs = point[:,0]
        ys = point[:,1]




        h, w = image.shape[:2]
        new = self.output_size

        rand_x = [[0, min(xs)/2], [(w+max(xs))/2, w]]
        rand_y = [[0, min(ys)/2], [(h+max(ys))/2, h]]
        top_idx = [np.random.randint(0, 2), np.random.randint(0, 2)]
        left_idx = [np.random.randint(0, 2), np.random.randint(0, 2)]

        top = np.random.randint(rand_x[top_idx[0]], rand_x[top_idx[1]])
        left = np.random.randint(rand_y[left_idx[0]], rand_y[left_idx[0]])

        image = image[top[0]:top[1], left[0]:left[1]]

        return image, new_point



def makeGaussian(size, sigma):
    x, y = np.meshgrid(np.linspace(-1,1,size[0]), np.linspace(-1,1,size[1]))
    dst = np.sqrt(x*x+y*y)
    
    # Initializing sigma and muu
    muu = 0.0
    
    # Calculating Gaussian array
    return np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))

def makeHeatmap(size, bboxs, img=None):
    heatMap = np.zeros(size)
    for k, box in enumerate(bboxs):

        if box.size == 4: # which is 2 point square
            start = [min(box[0], box[2]), min(box[1], box[3])]
            w, h = int(abs(box[0] - box[2])), int(abs(box[1] - box[3]))
            heat = makeGaussian((w, h), 0.5)
            heatMap[start[1]:start[1]+heat.shape[0], start[0]:start[0]+heat.shape[1]] += heat
        elif box.size == 8:
            w, h = size[1], size[0]
            tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
            heat = np.ones((h,  w))
            # heat = makeGaussian((w, h), 1)
            dstArr = np.array([box[:2], box[2:4], box[4:6], box[6:8]], np.float32)

            M = cv2.getPerspectiveTransform(tar, dstArr)
            heat = cv2.warpPerspective(heat, M, (w, h), flags=cv2.INTER_NEAREST)

            heatMap += heat
    
    heatMap = np.clip(heatMap, 0, 1)      

    return heatMap


def getFileNum(n):
    return str(n).zfill(5)  

def point_adjust(points, base, to):
    res = points
    res = res * np.asarray(to)/np.asarray(base)

    return res.astype(np.int)

def pointMap(points, size=(72, 56)):
    base = np.ones((size[1], size[0]))
    base *= -1
    for each in points:
        each[0] *= size[0]/NN.INPUT_SHAPE[0]
        each[1] *= size[1]/NN.INPUT_SHAPE[1]

        base[int(each[1]), int(each[0])] =  255

    return base


def normalize(arr, _min=0, _max=255):
    arr = arr - np.min(arr) + _min
    if np.max(arr) == 0: 
        return arr
    arr = np.asarray(arr, dtype=np.float32) * _max/np.max(arr)
    return arr


def cvt2Heatmap(img, superimposeOn=None, ratio=0.8, threshold=128):
    ori = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ori[ori < threshold] = 0
    img = cv2.applyColorMap(ori, cv2.COLORMAP_JET)

    # color map 0 -> blue(128)
    # cut blue background
    blue_img = img[..., 0]
    blue_img[blue_img <= 128] = 0
    img[..., 0] = blue_img

    # impose on impose image
    if superimposeOn is not None:
        if superimposeOn.shape != img.shape:
            superimposeOn = cv2.resize(superimposeOn, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(superimposeOn, ratio, img, 0.6, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def drawCanvasWithImg(img, size=(600, 800)):
    h, w = img.shape[:2]
    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    start = size[0]//2-h//2, size[1]//2-w//2

    canvas[start[0]:start[0]+h, start[1]:start[1]+w] = img

    return canvas

# out: (h, w, 35)
def convert2LabelMap(out):
    res = np.argmax(out, axis=2)
    return res

def convert2RGB(img, labels):
    img[img>33] = 0
    tmp = img.flatten()
    out = np.zeros((tmp.shape[0], 3), dtype=np.uint8)
    for singlePixel in range(len(tmp)):
        out[singlePixel] = labels.id2label[tmp[singlePixel]].color
    out = out.reshape((NN.OUTPUT_SHAPE[1], NN.OUTPUT_SHAPE[0], 3))
    return out

def segCombine(data):
    ori_h, ori_w = data.shape[:2]
    tmp = data.reshape(-1, 35)
    h, w = tmp.shape
    res = np.zeros((h, 1))
    for each in range(h):
        res[each] = np.argmax(tmp[each])

    res = res.reshape(ori_h, ori_w)
    return res