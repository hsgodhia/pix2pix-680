import os, torch, numpy as np, pdb, re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from matplotlib.pyplot import imshow
from torch.autograd import Variable
from IPython.core.debugger import set_trace
from torch.optim import lr_scheduler

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def plot_image(img_tensor, fName):
    # restore the normalization of the image before making PIL
    pilimg = transforms.ToPILImage()(img_tensor*0.5 + 0.5)    
    pilimg.save("./cmp_data/epochs/{}.jpg".format(fName))    

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
                if 'n02980441' not in file:
                    continue
                if file[0] == '.':
                    continue
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
            perm = np.random.permutation(len(self.train_image_names))
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

class Discriminator(nn.Module):
    def __init__(self, input_nc, num_filters):
        super(Discriminator, self).__init__()
        
        layers = []
        #256
        layers.append(nn.Conv2d(input_nc, num_filters, 4, 2, 1, bias=False))
        #128
        layers.append(cnnblock(num_filters, num_filters*2, False, True, False, False))
        #64
        layers.append(cnnblock(num_filters*2, num_filters*2*2, False, True, False, False))
        #32
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(num_filters*2*2, num_filters*2*2*2, 4, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(num_filters*2*2*2))
        #31
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(num_filters*2*2*2, 1, 4, 1, 1, bias=False))
        layers.append(nn.Sigmoid())
        #patch of 30*30
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)    

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

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_filters):
        super(Generator, self).__init__()
        #128 to 1 encoder
        self.layr1 = nn.Conv2d(input_nc, num_filters, 4, 2, 1, bias=False)
        
        self.layr2 = cnnblock(num_filters, num_filters*2, False, True, False, False)
        self.layr3 = cnnblock(num_filters*2, num_filters*2*2, False, True, False, False)
        self.layr4 = cnnblock(num_filters*2*2, num_filters*2**3, False, True, False, False)
        self.layr5 = cnnblock(num_filters*2**3, num_filters*2**3, False, True, False, False)
        self.layr6 = cnnblock(num_filters*2**3, num_filters*2**3, False, True, False, False)
        self.layr7 = cnnblock(num_filters*2**3, num_filters*2**3, False, False, False, False)
        
        # decode 1 back to 128 with dropout for first layer
        self.dlayr7 = cnnblock(num_filters*2**3, num_filters*2**3, True, True, True, True)
        self.dlayr6 = cnnblock(num_filters*2**3, num_filters*2**3, True, True, True, True)
        self.dlayr5 = cnnblock(num_filters*2**3, num_filters*2**3, True, True, True, True)
        self.dlayr4 = cnnblock(num_filters*2**3, num_filters*2*2, True, True, True, False)
        self.dlayr3 = cnnblock(num_filters*2*2, num_filters*2, True, True, True, False)
        self.dlayr2 = cnnblock(num_filters*2, num_filters, True, True, True, False)

        self.dlayr1 = nn.Sequential(*[
            nn.ReLU(), nn.ConvTranspose2d(num_filters, output_nc, 4, 2, 1, bias=False),
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
        dout7 = self.dlayr7(out7)
        dout6 = self.dlayr6(dout7)
        dout5 = self.dlayr5(dout6)
        dout4 = self.dlayr4(dout5)
        dout3 = self.dlayr3(dout4)
        dout2 = self.dlayr2(dout3)
        dout1 = self.dlayr1(dout2)
        
        return dout1

def init_model(mdl):
    for p in mdl.parameters():
        clname = p.__class__.__name__
        if clname.find('Conv') != -1:
            p.weight.data.normal_(0.0, 0.02)
        elif clname.find('Linear') != -1:
            p.weight.data.normal(0.0, 0.02)
        elif clname.find("Batch") != -1:
            p.weight.data.normal_(0.0, 0.02)
            p.bias.data.constant(0.0)

def train(niter):
    # --gray to rgb
    input_nc, output_nc, num_filters, l1_weight = 1, 3, 64, 60
    PATCH_SIZE = 30
    netG = GeneratorUNet(input_nc, output_nc, num_filters)
    init_model(netG)
    netG.train()
    # --assign probability to rgb image of being real/fake
    netD = Discriminator(input_nc + output_nc, num_filters) # --d is conditioned on the input image(grayscale)
    init_model(netD)
    netD.train()
    
    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    
    if torch.cuda.is_available():        
        netG.cuda()
        netD.cuda()        
        criterionBCE.cuda()
        criterionCAE.cuda()
    
    optimD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    overall_disc_loss_val = []
    total_loss_val = []
    for epoch in range(niter):
        disc_loss = []
        for i_batch, sample_batch in enumerate(train_dataloader):
            # -- label to identify soruce as real - 1 or fake - 0
            label_src_ = Variable(torch.FloatTensor(sample_batch['color_img'].size(0), 1, PATCH_SIZE, PATCH_SIZE))
            photo, sketch, path = sample_batch['color_img'], sample_batch['grayscale_img'], sample_batch['image_name']
            photo, sketch = Variable(photo), Variable(sketch)
            if torch.cuda.is_available():
                photo = photo.cuda()
                sketch = sketch.cuda()
                label_src_ = label_src_.cuda()

            # optimize D
            for p in netD.parameters(): 
                p.requires_grad = True            
            netD.zero_grad()
            
            # -- train on real
            label_src_.data.fill_(1)
            d_prob = netD(torch.cat([photo, sketch], 1)) # the discriminator (is conditioned concat with input)
            errD_real = criterionBCE(d_prob, label_src_)
            errD_real.backward()
            
            # -- train on fake
            label_src_.data.fill_(0)
            x_hat = netG(sketch)
            fake = x_hat.detach() # so we prevent backpropogating into G            
                
            d_prob = netD(torch.cat([fake, sketch], 1))            
            errD_fake = criterionBCE(d_prob, label_src_)
            errD_fake.backward()
            
            # half the objective value as an indirect way to make learning D slower than G
            errD = (errD_fake + errD_real)
            
            optimD.step()
            
            # optimize G
            for p in netD.parameters(): 
                p.requires_grad = False
            netG.zero_grad()    
            
            # -- calc L1 loss between generator and target
            l1_loss = criterionCAE(x_hat, photo)*l1_weight
            l1_loss.backward(retain_graph=True)
            # -- fool D
            
            label_src_.data.fill_(1)            
            d_prob = netD(torch.cat([x_hat, sketch], 1))
            errG_ = criterionBCE(d_prob, label_src_)
            errG_.backward()
            
            errG = errG_ + l1_loss*l1_weight
            
            optimG.step()
            
            disc_loss.append(errD.data[0])
            total_loss_val.append((errG + errD).data[0])
            
            if epoch % 10 == 0:
                for fli, flN in enumerate(path):
                    if flN in train_dataset.val_imgs:
                        plot_image(photo[fli, :,:,:].data.cpu(), "photo_{}_{}".format(epoch, flN))                
                        plot_image(sketch[fli,:,:,:].data.cpu(), "sketch_{}_{}".format(epoch, flN))
                        plot_image(fake[fli,:,:,:].data.cpu(), "fake_{}_{}".format(epoch, flN))

        overall_disc_loss_val.extend(disc_loss)
        print('Disciminator Loss after epoch:', epoch, ' is :', sum(disc_loss)/len(disc_loss))            
        
    torch.save(netG.state_dict(), './p2p_plain.pth')
    torch.save(netD.state_dict(), './p2p_plain.pth')
    np.save('total_loss_plain.npy', np.array(total_loss_val))
    np.save('overall_disc_loss_plain.npy', np.array(overall_disc_loss_val))

BATCH_SIZE=64
train_dataset = ImageNetData(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("Training on images:", len(train_dataset))
train(300)