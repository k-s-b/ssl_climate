import torch
import torch.nn as nn
import math
from PIL import Image
import numpy as np
import torch.nn.functional as F
from cutmix.utils import onehot, rand_bbox
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img
import time
from sklearn.metrics import mean_squared_error

from network.mynn import initialize_weights, Norm2d, Norm1d

from ssl_dataset import random_crop, HRClimateDataset, ZSSRTypeDataset
cnn_channels = 64
num_cnn_layers = 8


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(cnn_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(cnn_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data) #residual here
        return output


class _NetG(nn.Module):
    def __init__(self, num_channels=3):
        
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block(), num_cnn_layers)

        self.do1 = torch.nn.Dropout(p=0.2)
        self.do2 = torch.nn.Dropout(p=0.2)

        self.conv_mid = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_mid = nn.InstanceNorm2d(cnn_channels, affine=True)
        self.s_bn_mid = nn.InstanceNorm2d(cnn_channels, affine=True)

        self.conv_output = nn.Conv2d(in_channels=cnn_channels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)
    
    

    def forward(self, x, altitude = None):

        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out) #multiple layers here
        out = self.bn_mid(self.conv_mid(out))
        
#         if self.use_do: out = self.do1(out)

        out = torch.add(out,residual)
    
        out = self.conv_output(out)

        return out



def make_model():
    
    model = _NetG(3)

    lf = nn.MSELoss()
    opt_lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

    model = model.cuda()

    criterion = lf.cuda()
    
    return model, criterion, optimizer



def prep_data_transform(data):
    lr_father = F.interpolate(data.unsqueeze(0), scale_factor = (1./2.), mode= 'bicubic') #downsize original: this is supposed to be the original input
    lr_father = F.interpolate(lr_father, (data.shape[1], data.shape[2]), mode= 'bicubic').squeeze(0)
        
    means = [torch.mean(lr_father[x,:,:]).item() for x in range(3)]
    stds = [torch.std(lr_father[x,:,:]).item() for x in range(3)]
    return transforms.Compose([
            transforms.Normalize(mean=means,
                                std=stds)
        ]), means, stds


dataset = HRClimateDataset(hr_dir = #<path to data>)

def inv(channel, data, means, stds):
    mean=(-means[channel]/stds[channel])
    std=(1.0/stds[channel])
    data = (data - mean)/std
    return data


def adjust_learning_rate(opt_lr, epoch, step = 3000):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt_lr * (0.1 ** (epoch // step))
    return lr


idx = list(range(len(dataset)))
epochs = 5000
scale_factors = [2.]

random.seed(1)

random.shuffle(idx)

no_samples = 0

results = []
best_outs = []

random_indexes = []
for i in idx: # random loop through data
    random_indexes.append(i)
    model, criterion, optimizer = make_model()
    
    no_samples += 1
    
    model.train()
    count = 0
    mses = [1e10]
    chn_0 = []
    chn_1 = []
    chn_0_test = []
    chn_1_test = []
    
    start_time = time.time()
    
    train_sets = []
    
    data_transform, means, stds = prep_data_transform(dataset[i]) # get data transform, mean and stds for this data point
    
    for scale_factor in scale_factors: # each datapoint through the scale factors, if multiple 
        train_sets.append(ZSSRTypeDataset(dataset[i], scale_factor = scale_factor, transform=data_transform, crop=False, data_aug=None))
    
    lr_0, hr_0 = train_sets[0][i]
    lr = lr_0
    hr = hr_0
    
    best_out = 0
    
    for epoch in range(epochs): # each scale factor epoch # times

        out = model(lr.cuda())

        mse_loss = criterion(out, hr[:,:2,:,:].cuda())

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        count+=1
        if(count%50==0): # check every 50 epochs to speed up            
            
            with torch.no_grad():
                model.eval()

                out_test = model(hr.cuda())
                
                if(mse_loss.item()<mses[np.argmin(mses)]):
                    mses.append(mse_loss.item())
                    best_out = out_test

    invs = []
    hrs = []
        
    for chn in range(2):
        invs.append(torch.clamp(inv(chn,best_out[:,chn,:,:], means, stds), min = 0.)) # compute inverse of out and hr
        hrs.append(torch.clamp(inv(chn,hr[:,chn,:,:], means, stds), min = 0.)) 

    chn_0.append((nn.MSELoss()(invs[0], dataset[i][0,:,:].cuda()).item())) # mse of out_father vs GT
    chn_1.append(nn.MSELoss()(invs[1], dataset[i][1,:,:].cuda()).item())

    results.append((chn_0[-1], chn_1[-1],          nn.MSELoss()(hrs[0].cuda(), dataset[i][0,:,:].cuda()).item(),          nn.MSELoss()(hrs[1].cuda(), dataset[i][1,:,:].cuda()).item())) #mse of cubic hr vs GT

                   
    print(f'Data point: {i}, Time taken: {time.time()-start_time}')
    print(f'Model_chn_0: {results[-1][0]}, Bicubic_chn_0: {results[-1][2]:}')
    print(f'Model_chn_1: {results[-1][1]}, Bicubic_chn_1: {results[-1][3]}')
    print('-'*32)
    
    best_outs.append((torch.cat((invs[0], invs[1]), 0).cpu().numpy(), torch.cat((hrs[0], hrs[1]), 0).cpu().numpy())) #out, hr tuple
    
    if(no_samples == 10):
        break


results_ar = np.array(results)

def crmse(data):
    return np.sqrt(data.mean())

print(f'Model_chn_0: {crmse(results_ar[:,0])}, Bicubic_chn_0: {crmse(results_ar[:,2])}')
print(f'Model_chn_1: {crmse(results_ar[:,1])}, Bicubic_chn_1: {crmse(results_ar[:,3])}')