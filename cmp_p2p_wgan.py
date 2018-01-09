import os, torch, pandas as pd, numpy as np, pdb
from skimage import io, transform
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
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()

def plot_image(img_tensor, fName, cmap=None):
    pilimg = transforms.ToPILImage()(img_tensor*0.5 + 0.5)    
    pilimg.save("./cmp_data/epochs/{}.jpg".format(fName))    

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
        # in wgans instead of a sigmoid the critic returns a score
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

def init_model(mdl):
    for p in mdl.parameters():
        clname = p.__class__.__name__
        if clname.find('Conv') != -1:
            init.xavier_normal(p.weight.data, gain=0.02)
        elif clname.find('Linear') != -1:
            init.xavier_normal(p.weight.data, gain=0.02)
        elif clname.find("Batch") != -1:
            init.normal(p.weight.data, 1, 0.02)
            init.constant(p.bias.data, 0.0)

def calc_gradient_penalty(netD, photo, sketch, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    b_size = photo.size(0)
    alpha = torch.rand(b_size, 1)
    alpha = alpha.expand(b_size, photo.nelement()/b_size).contiguous().view(b_size, 3, 256, 256)

    if use_cuda:
        alpha = alpha.cuda()    
        
    interpolates = alpha * photo + ((1 - alpha) * fake_data)        
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(torch.cat([interpolates, sketch], 1))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()    
    return gradient_penalty
    
def train(niter):
    # --gray to rgb
    input_nc, output_nc, num_filters, l1_weight, grad_pen_weight, cgan_weight = 1, 3, 64, 10, 100, 1
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
    one = torch.FloatTensor([1])
    mone = one * -1

    if torch.cuda.is_available():        
        one = one.cuda()
        mone = mone.cuda()
        netG.cuda()
        netD.cuda()          
        criterionBCE.cuda()
        criterionCAE.cuda()
    
    optimD = optim.RMSprop(netD.parameters(), lr=5e-5)#, betas=(0.5, 0.999))
    optimG = optim.RMSprop(netG.parameters(), lr=5e-5)#, betas=(0.5, 0.999))
        
    total_loss_val = []
    for epoch in range(2*niter + 1):
        gen_iterations, d_iter, i_batch, disc_loss = 0, 5, 0, []
        data_iter = iter(train_dataloader)
        
        while i_batch < len(train_dataloader):         
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                d_iter = 100
            else:
                d_iter = 5
                
            while d_iter > 0 and i_batch < len(train_dataloader):

                sample_batch = data_iter.next()
                i_batch += 1
                
                # -- label to identify soruce as real - 1 or fake - 0
                photo, sketch, path = sample_batch['photo'], sample_batch['sketch'], sample_batch['name']
                gp_alpha_tensor = torch.FloatTensor(photo.size(0), 1, 1, 1)
                photo, sketch = Variable(photo), Variable(sketch)
                if torch.cuda.is_available():
                    photo = photo.cuda()
                    gp_alpha_tensor = gp_alpha_tensor.cuda()
                    sketch = sketch.cuda()

                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
                    
                # optimize D    
                for p in netD.parameters(): 
                    p.requires_grad = True            
                
                netD.zero_grad()
                
                # -- train on real
                d_score_real = netD(torch.cat([photo, sketch], 1)) # the discriminator (is conditioned concat with input)
                d_score_real = d_score_real.mean()
                d_score_real.backward(mone)

                # -- train on fake
                x_hat = netG(sketch)
                fake = x_hat.detach() # so we prevent backpropogating into G            

                d_score_fake = netD(torch.cat([fake, sketch], 1))            
                d_score_fake = d_score_fake.mean()
                d_score_fake.backward(one)

                # -- compute the gradiet penalty
                #grad_pen = calc_gradient_penalty(netD, photo.data, sketch, fake.data)*grad_pen_weight
                #grad_pen.backward()

                d_loss = d_score_real - d_score_fake # + grad_pen
                optimD.step()            
                
                d_iter -= 1
                
    
            # optimize G
            for p in netD.parameters(): 
                p.requires_grad = False
            netG.zero_grad()    

            # -- calc L1 loss between generator and target
            l1_loss = criterionCAE(x_hat, photo)*l1_weight
            l1_loss.backward(retain_graph=True)
            # -- fool D
            d_score = netD(torch.cat([x_hat, sketch], 1))*cgan_weight
            d_score = d_score.mean()
            d_score.backward(mone)
            
            G_cost = -d_score
            optimG.step()
            gen_iterations += 1
            disc_loss.append(d_loss.data[0])
            total_loss_val.append((G_cost + l1_loss + d_loss).data[0])
            
            cur_name = path[0]
            if 'cmp_b0307' in cur_name or 'cmp_b0297' in cur_name or 'cmp_b0296' in cur_name or 'cmp_b0292' in cur_name:
                print("plotting {} after epoch: {}".format(path[0], epoch))
                plot_image(fake[0,:,:,:].data.cpu(), "fake_{}_{}".format(epoch, cur_name.split("/")[-1]))
                
        if epoch % 10 == 0:
            print('Disciminator Loss after epoch:', epoch, ' is :', sum(disc_loss)/len(disc_loss))            
            torch.save(netG.state_dict(), './cmp_g.pth')
            torch.save(netD.state_dict(), './cmp_d.pth')
    np.save('cmp_t_loss.npy', np.array(total_loss_val))

BATCH_SIZE=1
train_dataset = Facade(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
train(300)