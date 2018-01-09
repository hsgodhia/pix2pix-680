
# coding: utf-8

# In[ ]:

import os, torch, pandas as pd, numpy as np, pdb, re
import skimage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow
from torch.autograd import Variable
from IPython.core.debugger import set_trace
from torch.optim import lr_scheduler
import torch.nn.functional as F


# In[ ]:

def plot_image(img_tensor, name, cmap=None):    
    print('testin', name)
    fig = plt.figure()
    for i in range(3):
        a=fig.add_subplot(1,3,i+1)
        img = img_tensor[i]*0.5 + 0.5        
        pilimg = transforms.ToPILImage()(img)
        if img.size(0) == 1:
            plt.imshow(pilimg)
        else:
            plt.imshow(pilimg)
        plt.axis('off')
    plt.show()


# In[ ]:

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# In[ ]:

def cnnblock(in_c, out_c, transposed=False, bn=True, relu=True, dropout=False):
    layers = []
    if relu:
        layers.append(nn.ReLU(inplace=True))        
    else:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        layers.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
        
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
        
    if dropout:
        layers.append(nn.Dropout2d(0.5, inplace=True))
        
    return nn.Sequential(*layers)


# In[ ]:

class Facade(Dataset):
    def __init__(self, train=True):
        self.train = train
        if train:
            folder = './cmp_data/train/photos'
        else:
            folder = './cmp_data/test/photos'
        self.images = []
        for root, _, fnames in sorted(os.walk(folder)):
            for fname in fnames:
                if fname.endswith('.jpg'):
                    path = os.path.join(folder, fname)
                    self.images.append(path)
        self.images = sorted(self.images)
        self.transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        photo_path = self.images[idx]
        sk_path = photo_path.replace("photos", "sketches").replace(".jpg", ".png")        
        photo = Image.open(photo_path).resize((356,356), Image.BILINEAR)
        sketch = Image.open(sk_path).resize((356,356), Image.BILINEAR)
        
        sample = {'photo': self.transform(photo), 'sketch': self.transform(sketch), 'name': photo_path, 'sketch-name':sk_path}        
        return sample


# In[ ]:

class GeneratorUNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_filters):
        super(GeneratorUNet, self).__init__()
        #128 to 1 encoder
        self.layr1 = nn.Conv2d(input_nc, num_filters, 4, 2, 1, bias=False)
        
        self.layr2 = cnnblock(num_filters, num_filters*2, False, True, False, False)
        self.layr3 = cnnblock(num_filters*2, num_filters*2*2, False, True, False, False)
        self.layr4 = cnnblock(num_filters*2*2, num_filters*2**3, False, True, False, False)
        self.layr5 = cnnblock(num_filters*2**3, num_filters*2**3, False, True, False, False)
        self.layr6 = cnnblock(num_filters*2**3, num_filters*2**3, False, True, False, False)
        self.layr7 = cnnblock(num_filters*2**3, num_filters*2**3, False, False, False, False)
        self.layr8 = cnnblock(num_filters*2**3, num_filters*2**3, False, False, False, False)
        
        # decode 1 back to 128 with dropout for first layer
        self.dlayr8 = cnnblock(num_filters*2**3, num_filters*2**3, True, True, True, True)
        self.dlayr7 = cnnblock(num_filters*2**4, num_filters*2**3, True, True, True, True)
        self.dlayr6 = cnnblock(num_filters*2**4, num_filters*2**3, True, True, True, True)
        self.dlayr5 = cnnblock(num_filters*2**4, num_filters*2**3, True, True, True, False)
        self.dlayr4 = cnnblock(num_filters*2**4, num_filters*2*2, True, True, True, False)
        self.dlayr3 = cnnblock(num_filters*2*2*2, num_filters*2, True, True, True, False)
        self.dlayr2 = cnnblock(num_filters*2*2, num_filters, True, True, True, False)

        self.dlayr1 = nn.Sequential(*[
            nn.ReLU(), nn.ConvTranspose2d(num_filters*2, output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
    def forward(self, x):
        out1 = self.layr1(x)
        out2 = self.layr2(out1)
        out3 = self.layr3(out2)
        out4 = self.layr4(out3)
        out5 = self.layr5(out4)
        out6 = self.layr6(out5)
        out7 = self.layr7(out6)
        out8 = self.layr8(out7)


        dout8 = self.dlayr8(out8)        
        dout7 = self.dlayr7(torch.cat([dout8, out7], 1))             
        dout6 = self.dlayr6(torch.cat([dout7, out6], 1))
        dout5 = self.dlayr5(torch.cat([dout6, out5], 1))
        dout4 = self.dlayr4(torch.cat([dout5, out4], 1))
        dout3 = self.dlayr3(torch.cat([dout4, out3], 1))
        dout2 = self.dlayr2(torch.cat([dout3, out2], 1))
        dout1 = self.dlayr1(torch.cat([dout2, out1], 1))
        
        return dout1


# In[ ]:

up = nn.Upsample(size=(299, 299), mode='bilinear')
def get_pred(x, inception_model):
    x = up(x)
    x = inception_model(x)
    return F.softmax(x).data.cpu().numpy()


# In[ ]:

def test():
    class_mapping  = {0:149,1:227,2:245,3:483,4:493,5:667,6:712,7:791,8:831,9:834}
    input_nc, output_nc, num_filters = 1, 3, 64
    netG = GeneratorUNet(input_nc, output_nc, num_filters)
    netG.load_state_dict(torch.load('./cmp_g.pth'))
    netG.eval()
    
    if torch.cuda.is_available():
        netG.cuda()
        
    for i_batch, sample_batch in enumerate(test_dataloader):
        photo, sketch, path= sample_batch['photo'], sample_batch['sketch'], sample_batch['name']
    
        photo, sketch = Variable(photo), Variable(sketch)
        if torch.cuda.is_available():
            photo = photo.cuda()
            sketch = sketch.cuda()
        x_hat = netG(sketch)
        #preds[i_batch*BATCH_SIZE:i_batch*BATCH_SIZE + photo.size(0)] = get_pred(x_hat, inception_model)
        #true_preds[i_batch*BATCH_SIZE:i_batch*BATCH_SIZE + photo.size(0)] = get_pred(photo, inception_model)
        for i in range(len(path)):
            plot_image([sketch.data[i,:,:,:].cpu(), x_hat.data[i,:,:,:].cpu(), photo.data[i,:,:,:].cpu()], path[i])

    #print(i_batch)
    #g_preds = np.argmax(preds, 1)
    #e_preds = np.array(act_labels)
    #print(g_preds.shape)
    #print(e_preds.shape)
    #tot = np.mean(g_preds == e_preds)
    #print(tot)


# In[ ]:

BATCH_SIZE=1
test_dataset = Facade(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test()


# In[ ]:

print("this is with ")


# In[ ]:



