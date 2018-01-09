
# coding: utf-8

# In[1]:

import os, torch, pandas as pd, numpy as np, pdb
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
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms


# In[2]:

up = nn.Upsample(size=(299, 299), mode='bilinear')
def get_pred(x):
    x = up(x)
    x = inception_model(x)
    return F.softmax(x).data.cpu().numpy()


# In[3]:

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
        self.transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        photo_path = self.images[idx]
        sk_path = photo_path.replace("photos", "sketches").replace(".jpg", ".png")        
        photo = Image.open(photo_path).resize((356,356), Image.BILINEAR)
        sketch = Image.open(sk_path).resize((356,356), Image.BILINEAR)
        
        sample = { 'photo': self.transform(photo), 'sketch': self.transform(sketch), 'name': photo_path, 'sketch-name':sk_path }        
        return sample


# In[17]:

# Get predictions
BATCH_SIZE=25
test_dataset = Facade(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
preds = np.zeros((len(test_dataset), 1000))


# In[6]:

inception_model = inception_v3(pretrained=True, transform_input=True)
inception_model.eval();


# In[18]:

for i, dictb in enumerate(test_dataloader, 0):
    batch = dictb['photo']    
    batchv = Variable(batch)
    batch_size_i = batch.size()[0]
    preds[i*BATCH_SIZE:i*BATCH_SIZE + batch_size_i] = get_pred(batchv)


# In[19]:

max_i = np.argmax(preds, 1)


# In[20]:

max_v = np.max(preds, 1)


# In[21]:

max_i[:10] # get the max class index


# In[22]:

max_v[:10]*100 # get the confidence scores for the class


# In[ ]:



