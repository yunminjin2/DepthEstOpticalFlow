
import torch
import torch.nn.init

import torch.nn.functional as F
import torchsummary.torchsummary

import numpy as np


INPUT_SHAPE = (640, 480)
OUTPUT_SHAPE = (320, 240)

def init_weights(modules):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class Bottleneck(torch.nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.norm = torch.nn.BatchNorm2d
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm(planes)
       
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.norm(planes)

        self.conv3 = torch.nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = self.norm(planes*self.expansion)
        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)

        out = self.relu(out)

        return out

class double_conv(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# [batch_size, 256, w/8, h/8]
# [batch_size, 256, w/4, h/4]
# [batch_size, 256, w/2, h/2]
class FPN(torch.nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        
        self.in_planes = 64

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(64),
            torch.nn.MaxPool2d(2, 2)
        )
        
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1) # 64 -> 256
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)# 128 -> 512
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)# 256 -> 1024

        self.toplayer = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce  channels
        
        self.smooth1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Lateral layers
        self.latlayer1 = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer2(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p4, p3, p2

class FeaturePyramidNet(torch.nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        
        self.in_planes = 64

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(64),
            torch.nn.MaxPool2d(2, 2)
        )
        
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], stride=1) # 64 -> 256
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)# 128 -> 512
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)# 256 -> 1024

        self.toplayer = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce  channels
        
        self.smooth1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Lateral layers
        self.latlayer1 = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        p4 = self.toplayer(c4)
        p3 = self.latlayer1(c3)
        p2 = self.latlayer2(c2)

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p4, p3, p2

# U Net
# only for detecting around part
class AroundModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.size_down_layer = torch.nn.MaxPool2d(2,2)

        self.reduce_features = torch.nn.ModuleList()
        self.reduce_features.append(torch.nn.Conv2d(256, 128, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(128, 64, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(64, 1, kernel_size=1))

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            Bottleneck(64, 16),
            Bottleneck(64, 16),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            Bottleneck(128, 32),
            Bottleneck(128, 32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=6, dilation=6),
            torch.nn.Conv2d(512, 512, kernel_size=1)
        )

        self.upconv1 = double_conv(512, 256, 128)
        self.upconv2 = double_conv(256, 128, 64)
        self.upconv3 = double_conv(128, 64, 32)
        self.upconv4 = double_conv(64, 32, 16)

        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 1, kernel_size=1),
        )

        init_weights(self.layer1.modules())
        init_weights(self.layer2.modules())
        init_weights(self.layer3.modules())
        init_weights(self.layer4.modules())
        

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        # x = self.size_down_layer(x)
        # x = self.size_down_layer(x)

        relu1 = self.layer1(x)
        relu2 = self.layer2(relu1)
        relu3 = self.layer3(relu2)
        relu4 = self.layer4(relu3)
        relu5 = self.layer5(relu4)


        h = torch.cat([relu5, relu4], dim=1)
        h = self.upconv1(h)
        h = F.interpolate(h, size=relu3.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu3], dim=1)
        h = self.upconv2(h)
        h = F.interpolate(h, size=relu2.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu2], dim=1)
        h = self.upconv3(h)
        h = F.interpolate(h, size=relu1.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu1], dim=1)
        feature = self.upconv4(h)

        h = self.conv_out(feature)

        return h, feature

class AroundModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.size_down_layer = torch.nn.MaxPool2d(2,2)

        self.reduce_features = torch.nn.ModuleList()
        self.reduce_features.append(torch.nn.Conv2d(256, 128, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(128, 64, kernel_size=1))
        self.reduce_features.append(torch.nn.Conv2d(64, 1, kernel_size=1))

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            Bottleneck(64, 16),
            Bottleneck(64, 16),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            Bottleneck(128, 32),
            Bottleneck(128, 32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=6, dilation=6),
            torch.nn.Conv2d(512, 512, kernel_size=1)
        )

        self.upconv1 = double_conv(512, 256, 128)
        self.upconv2 = double_conv(256, 128, 64)
        self.upconv3 = double_conv(128, 64, 32)
        self.upconv4 = double_conv(64, 32, 16)

        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 2, kernel_size=1),
        )

        init_weights(self.layer1.modules())
        init_weights(self.layer2.modules())
        init_weights(self.layer3.modules())
        init_weights(self.layer4.modules())
        

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        # x = self.size_down_layer(x)
        # x = self.size_down_layer(x)

        relu1 = self.layer1(x)
        relu2 = self.layer2(relu1)
        relu3 = self.layer3(relu2)
        relu4 = self.layer4(relu3)
        relu5 = self.layer5(relu4)


        h = torch.cat([relu5, relu4], dim=1)
        h = self.upconv1(h)
        h = F.interpolate(h, size=relu3.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu3], dim=1)
        h = self.upconv2(h)
        h = F.interpolate(h, size=relu2.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu2], dim=1)
        h = self.upconv3(h)
        h = F.interpolate(h, size=relu1.size()[2:], mode='bilinear', align_corners=False)

        h = torch.cat([h, relu1], dim=1)
        feature = self.upconv4(h)

        h = self.conv_out(feature)

        return h, feature

class SegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            Bottleneck(64, 16),
            Bottleneck(64, 16),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            Bottleneck(128, 32),
            Bottleneck(128, 32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.Conv2d(1024, 1024, kernel_size=1)
        )

        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 256)
        self.upconv4 = double_conv(256, 128, 128)

        self.maxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_out1 = self._make_layer(192, 5)
        self.conv_out2 = self._make_layer(192, 5)
        self.conv_out3 = self._make_layer(192, 5)
        self.conv_out4 = self._make_layer(192, 5)
        self.conv_out5 = self._make_layer(192, 5)
        self.conv_out6 = self._make_layer(192, 5)
        self.conv_out7 = self._make_layer(192, 5)

        init_weights(self.layer1.modules())
        init_weights(self.layer2.modules())
        init_weights(self.layer3.modules())
        init_weights(self.layer4.modules())
        init_weights(self.layer5.modules())
        
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())

    
    def _make_layer(self, in_planes, out_planes):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1), 
            torch.nn.BatchNorm2d(in_planes),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=1),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """ The input should be of size [batch_size, q4, img_h, img_w] """
        relu1 = self.layer1(x)
        relu2 = self.layer2(relu1)
        relu3 = self.layer3(relu2)
        relu4 = self.layer4(relu3)
        relu5 = self.maxPool(relu4)
        relu6 = self.layer5(relu5)

        h = torch.cat([relu6, relu5], dim=1)
        h = self.upconv1(h)
        h = F.interpolate(h, size=relu4.size()[2:], mode='nearest') 

        h = torch.cat([h, relu4], dim=1)
        h = self.upconv2(h)
        h = F.interpolate(h, size=relu3.size()[2:], mode='nearest')

        h = torch.cat([h, relu3], dim=1)
        h = self.upconv3(h)
        h = F.interpolate(h, size=relu2.size()[2:], mode='nearest')

        h = torch.cat([h, relu2], dim=1)
        h = self.upconv4(h)
        h = F.interpolate(h, size=relu1.size()[2:], mode='nearest')

        feature = torch.cat([h, relu1], dim=1)

        h1 = self.conv_out1(feature)
        h2 = self.conv_out2(feature)
        h3 = self.conv_out3(feature)
        h4 = self.conv_out4(feature)
        h5 = self.conv_out5(feature)
        h6 = self.conv_out6(feature)
        h7 = self.conv_out7(feature)

        res = torch.cat([h1, h2], dim=1)
        res = torch.cat([res, h3], dim=1)
        res = torch.cat([res, h4], dim=1)
        res = torch.cat([res, h5], dim=1)
        res = torch.cat([res, h6], dim=1)
        res = torch.cat([res, h7], dim=1)

        return res.permute(0, 2, 3, 1)

class MySegNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FeaturePyramidNet([2,2,2,2])

        self.conv_out1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 1, kernel_size=1)
        )
        self.conv_out2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 1, kernel_size=1)

        )

        self.conv_out3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=1),
            torch.nn.ReLU(),
        )

       


    def forward(self, x):
        out = self.fpn(x)

        labeling = self.conv_out1(out[0])

        # edge = self.conv_out2(out[1])
        edge = self.conv_out3(out[2])


        





        return labeling.permute(0, 2, 3, 1), edge.permute(0, 2, 3, 1)




def load_model(path, model_class, parallel=True):
    print("Loading... {}".format(path))
    if parallel:
        model_class = torch.nn.DataParallel(model_class)
    model_class.load_state_dict(torch.load(path))
    print("Done!\n")
    return model_class

def img2Tensor(data, device):
    data = torch.tensor(data)
    data = data.permute(2, 0, 1)[None,...]
    return data.to(device, non_blocking=True).long().float()
    
def run_model_img(data, model, device):
    model.eval()

    data = img2Tensor(data, device)

    predictions = model(data).cpu().detach().numpy()[0]
    # predictions = np.transpose(predictions, (1, 2, 0))
    return predictions

def saveModel(model, path):
    torch.save(model.state_dict(), path)

# test model
if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    model = MySegNet()
    model = torch.nn.DataParallel(model)
    model.to(device)

    torchsummary.torchsummary.summary(model, batch_size=1, device=device, input_size=(3, 640, 480))
    



