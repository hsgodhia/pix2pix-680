
# coding: utf-8

# In[1]:

import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from itertools import ifilter
from random import randint


# In[2]:

BATCH_SIZE = 150000
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, hid_dim, pretrained=None):
        super(Word2Vec, self).__init__()
        self.hid_dim = hid_dim
        
        #these are by intent to learn separate embedding matrices, we return word_emb
        self.word_emb = nn.Embedding(vocab_size, hid_dim)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(pretrained)
        self.context_emb = nn.Embedding(vocab_size, hid_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, wrd, cntxt):
        wrd_vec = self.word_emb(wrd) # N * 1 * D
        cntxt_vec = self.context_emb(cntxt) # N * 5 * D
        res = torch.bmm(wrd_vec, cntxt_vec.view(BATCH_SIZE, self.hid_dim, -1))
        res = self.sigmoid(res) # N * 1 * 5
        res = res.squeeze(1) # for each mini-batch we have a probability score for the 5 contexts
        return res


# In[3]:

def process_data(min_freq, flName=None, lines=None):
    """
    flName: file to read which contain raw data
    min_freq: required minimum frequency to be considered in vocabulary
    returns: vocab, vocab_size, word2index, index2word
    """
    vocab, index2word, word2index = {}, {}, {}
    if flName is not None:
        with open(flName) as fp:
            lines = fp.readlines()
            
    for line in lines:
        wrds = line.split(" ") #a very basic tokenizer that only splits by space and no stemming or cleaning        
        for w in wrds:
            w = w.strip().lower()
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
    print("initial vocab size:{}".format(len(vocab)))
    for wrd in vocab:
        if vocab[wrd] >= min_freq:
            index2word[len(index2word)] = wrd
            word2index[wrd] = len(index2word) - 1
        else:
            vocab[wrd] = 0
    vocab_size = len(index2word)
    return vocab, vocab_size, word2index, index2word


# In[4]:

def get_sim(wrd, k, mat, word2index):
    if wrd not in word2index:
        return None
    vec = mat[word2index[wrd], :].unsqueeze(1)
    othrs = torch.mm(mat, vec)
    othrs, ind = torch.sort(othrs, 0, descending=True)
    topk = ind[:k]
    for i in range(topk.size()[0]):
        print(index2word[topk[i][0]])        


# In[5]:

def get_score(wrd1, wrd2, mat):
    if wrd1 not in word2index or wrd2 not in word2index:
        return 0.0
    vec1 = mat[word2index[wrd1]]
    vec2 = mat[word2index[wrd2]]
    return torch.dot(vec2, vec1)


# In[9]:

glove_path, dim, min_count, neg_exmpl = "glove.6B.50d.txt", 50, 5, 60

with open("ppdb-2.0-xxxl-lexical", "r") as fp:
    lines = fp.readlines()
    pairs = []
    for l in lines:
        dt = l.split("|||")
        wrd1, wrd2 = dt[1], dt[2]
        wrd1, wrd2 = wrd1.strip(), wrd2.strip()
        if ".pdf" not in wrd1 and ".pdf" not in wrd2 and wrd1.isalpha() and wrd2.isalpha():
            pairs.append(wrd1 + " " + wrd2)

vocab, vocab_size, word2index, index2word = process_data(min_count, lines=pairs)

print("Data ready: {} {}".format(vocab_size, len(pairs)))


# In[13]:

mdl = Word2Vec(vocab_size, dim)
mdl.load_state_dict(torch.load('./mdl_preglove_neg50_50d.pth', map_location=lambda storage, loc: storage))
w2vmat = mdl.word_emb.weight.data.cpu()
norm = torch.norm(w2vmat, 2, 1, True)
w2vmat = w2vmat / norm.expand_as(w2vmat.size())


# In[8]:

get_sim('young', 10, w2vmat, word2index)


# In[9]:

get_sim('sleep', 10, w2vmat, word2index)


# In[10]:

get_score('young', 'old', w2vmat)


# In[11]:

get_sim('eat', 10, w2vmat, word2index)


# In[12]:

get_sim('hi', 10, w2vmat, word2index)


# In[25]:

get_sim('the', 10, w2vmat, word2index)


# In[26]:

get_sim('angry', 10, w2vmat, word2index)


# In[27]:

get_sim('fear', 10, w2vmat, word2index)


# In[28]:

get_sim('coast', 10, w2vmat, word2index)


# In[29]:

get_sim('shore', 10, w2vmat, word2index)


# In[30]:

get_score('coast', 'shore', w2vmat)


# In[ ]:



