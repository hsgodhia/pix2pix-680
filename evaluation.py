
# coding: utf-8

# In[1]:

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
from torchvision.models.inception import inception_v3


# In[2]:

def plot_image(img_tensor, name, cmap=None):    
    print('testin', name)
    fig = plt.figure()
    for i in range(3):
        a=fig.add_subplot(1,3,i+1)
        img = img_tensor[i]*0.5 + 0.5        
        pilimg = transforms.ToPILImage()(img)
        if img.size(0) == 1:
            plt.imshow(pilimg, cmap='gray')
        else:
            plt.imshow(pilimg)
        plt.axis('off')
    plt.show()


# In[3]:

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# In[4]:

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


# In[5]:

class ImageNetData(Dataset):
    def __init__(self, train=True):
        folder = './data/imagenet'
        self.train_images = []
        self.train_labels = []
        self.train_image_names = []
        self.val_imgs = []
        self.test_images = []
        self.test_labels = []
        self.test_image_names = []

        self.images = []
        self.labels = []
        self.image_names = []
        self.label_idx = {'n02074367':0,'n02105412':1,'n02108915':2,'n02980441':3,'n03016953':4,'n03787032':5,'n03920288':6,'n04204347':7,'n04344873':8,'n04350905':9}

        for root, dirs, files in os.walk(folder):
            file_count = 0
            files = natural_sort(files)
            for file in files:
                a = np.array(Image.open(os.path.join(root, file))).shape
                if len(a)!=3:
                    continue
                # 400 train images
                if file_count < 400:
                    self.train_images.append(os.path.join(root, file))
                    self.train_labels.append(file.split('_')[0])
                    self.train_image_names.append(file)
                else:
                    self.test_images.append(os.path.join(root, file))
                    self.test_labels.append(file.split('_')[0])
                    self.test_image_names.append(file)
                file_count += 1
                
                # total 500 images
                if file_count == 500:
                    break

        if train:
            self.images =  self.train_images
            self.labels =  self.train_labels
            self.image_names = self.train_image_names
            perm = nprandom.permutation(len(self.train_image_names))
            perm = perm[:5]
            
            self.val_imgs = [self.image_names[i] for i in perm.tolist()]
            print("Validating on images", self.val_imgs)
        else:
            self.images =  self.test_images
            self.labels =  self.test_labels
            self.image_names = self.test_image_names

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        color_img = Image.open(self.images[idx]).resize((256,256), Image.BILINEAR)
        grayscale_img = color_img.convert('L')
        label =  self.labels[idx]  
        onehot = torch.zeros(10)
        label = label.lstrip()
        label = label.rstrip()
        onehot[self.label_idx[label]] = 1
        
        image_name = self.image_names[idx]
        sample = { 'color_img': self.transform(color_img), 'grayscale_img': self.transform(grayscale_img), 'label': onehot, 'real_label': self.label_idx[label], 'image_name':image_name}   
        return sample


# In[6]:

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


# In[7]:

up = nn.Upsample(size=(299, 299), mode='bilinear')
def get_pred(x, inception_model):
    x = up(x)
    x = inception_model(x)
    return F.softmax(x).data.cpu().numpy()


# In[ ]:

def test():
    class_mapping  = {0:149,1:227,2:245,3:483,4:493,5:667,6:712,7:791,8:831,9:834}
    input_nc, output_nc, num_filters = 1, 3, 64
    act_labels = []
    preds = np.zeros((len(test_dataset), 1000))
    true_preds = np.zeros((len(test_dataset), 1000))
    inception_model = inception_v3(pretrained=True, transform_input=True)
    inception_model.eval();
    netG = GeneratorUNet(input_nc, output_nc, num_filters)
    netG.load_state_dict(torch.load('./cmp_g.pth'))
    #netG.eval()
    if torch.cuda.is_available():
        netG.cuda()
        inception_model.cuda()
        
    for i_batch, sample_batch in enumerate(test_dataloader):
        photo, sketch, path, real_label = sample_batch['color_img'], sample_batch['grayscale_img'], sample_batch['image_name'], sample_batch['label']
        label_index = np.where(real_label.numpy() == 1)[1][0]        
        act_labels.append(class_mapping[label_index])
    
        photo, sketch = Variable(photo), Variable(sketch)
        if torch.cuda.is_available():
            photo = photo.cuda()
            sketch = sketch.cuda()
        x_hat = netG(sketch)
        preds[i_batch*BATCH_SIZE:i_batch*BATCH_SIZE + photo.size(0)] = get_pred(x_hat, inception_model)
        true_preds[i_batch*BATCH_SIZE:i_batch*BATCH_SIZE + photo.size(0)] = get_pred(photo, inception_model)

        for i in range(len(path)):
            plot_image([sketch.data[i,:,:,:].cpu(), x_hat.data[i,:,:,:].cpu(), photo.data[i,:,:,:].cpu()], path[i])

    print(i_batch)
    g_preds = np.argmax(preds, 1)
    p_preds = np.argmax(true_preds, 1)

    e_preds = np.array(act_labels)
    print(g_preds.shape)
    print(e_preds.shape)
    tot = np.mean(g_preds == e_preds)
    tot2 = np.mean(p_preds == e_preds)

    print(tot)
    print(tot2)


# In[ ]:

BATCH_SIZE=20
test_dataset = ImageNetData(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test()


# In[9]:

print("this is with ")


# In[ ]:



