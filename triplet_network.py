import torch
import torch.utils
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from utility import *
import logging, os
import numpy as np
from image_loader import dataloader
from tripletloss import TripletLoss, TripletLoss_cosine

import argparse
def get_modeldir(modeldes):
    i = 0
    if not os.path.isdir(modeldes):
        os.mkdir(modeldes)
    while True:
        modeldir = modeldes + '/%d' % i
        if os.path.isdir(modeldir):
            i += 1
        else:
            os.mkdir(modeldir)
            return modeldir

def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def generate_rep(net, two_dfm, genver, _batch_size, verbose=False):
    inputs, labels = torch.from_numpy(two_dfm), torch.from_numpy(genver)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)

    total, correct = 0, 0
    sig_out, new_two_dfm = None, None
    for data, target in testloader:
        inputs, labels = data, target
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        new_sub_two_dfm = net(inputs)
        new_sub_two_dfm = new_sub_two_dfm.data.cpu().numpy()
        if new_two_dfm is not None:
            new_two_dfm = np.concatenate((new_two_dfm, new_sub_two_dfm), axis=0)
        else:
            new_two_dfm = new_sub_two_dfm
    new_two_dfm = norm(new_two_dfm)

    return new_two_dfm

def get_rep(filename, in_dir, dire, net_path, genlist, two_dfm, genver):

    start_time = time.time()
    if not os.path.exists(dire):
        os.mkdir(dire)
 
    
    #two_dfm, genver = load_2dfm(genlist, in_dir)
    np.savetxt(dire + 'data1.txt', two_dfm) 
    _batch_size = 1000
    
    
    net = torch.load(net_path)
    # print calc_MAP(test(net, two_dfm, genver, _batch_size), genver, [100, 350]) 
    new_two_dfm = generate_rep(net, two_dfm, genver, _batch_size)
    end_time = time.time()
    print('time: %fs' % ((end_time - start_time) / len(genver)))
    np.savetxt(dire + 'data0.txt', new_two_dfm) 
    np.savetxt(dire + 'version.txt', genver, fmt='%d')

def _test(net, two_dfm, genver, _batch_size, verbose=False):
    inputs, labels = torch.from_numpy(two_dfm), torch.from_numpy(genver)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)

    total, correct = 0, 0
    new_two_dfm = torch.Tensor([])
    for data, target in testloader:
        inputs, labels = data, target
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)

        #outputs = outputs.data.cpu().numpy()
       
        
        new_two_dfm = torch.cat((new_two_dfm, outputs.cpu()))

    n = new_two_dfm.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(new_two_dfm, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, new_two_dfm, new_two_dfm.t())
    dist = dist.clamp(min=1e-12).sqrt().detach().numpy()  # for numerical stability
    dist = norm(dist)
    
    return dist
def _test_cosine(net, two_dfm, genver, _batch_size, verbose=False):
    inputs, labels = torch.from_numpy(two_dfm), torch.from_numpy(genver)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)

    total, correct = 0, 0
    new_two_dfm = torch.Tensor([])
    for data, target in testloader:
        inputs, labels = data, target
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)
        new_two_dfm = torch.cat((new_two_dfm, outputs.cpu()))
    n = new_two_dfm.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = []
    inputs = new_two_dfm
    for i in range(n):
        dist.append(1 - F.cosine_similarity(inputs[i].expand_as(inputs),inputs) )
    dist = torch.stack(dist).detach().numpy()
    dist = norm(dist)
    return dist


def get_measure(two_dfm, genver):
    song_set_id = np.unique(genver)
    sum_conv = 0.
    iis, jis = [], []
    for i in range(7):
        if i == 0 or j == 6:
            n_jis = range(38)
        else:
            n_jis = range(75)
        iis += [i] * len(n_jis)
        jis += n_jis

    for id in song_set_id:
        song_set = two_dfm[np.where(genver == id)].reshape((-1, 12, 75))
        song_set = np.fft.ifftshift(song_set, axes=(1, 2))
        print(song_set[0, 2, 2], song_set[0, 10, 73])
        raw_input()
        song_set = song_set[:, iis, jis]

        cov = np.cov(song_set)
        sum_conv += np.linalg.cond(cov)
        
        print(cov.shape, np.linalg.cond(cov))
        raw_input()
        
    print(sum_conv / len(song_set_id))

class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        num_classes = weight
        #middle = 1024
        self.fc1 = nn.Linear(900, num_classes)
        #self.fc2 = nn.Linear(1000, 1000)
        #self.fc3 = nn.Linear(1000, 1000)
        #self.fc4 = nn.Linear(1000, 1000)
        #self.fc5 = nn.Linear(900, num_classes)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(self.fc3(x)
        #x = F.relu(self.fc4(x))
        #output = self.fc5(x)
        return x

def get_center_loss(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)

    target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)

    diff = centers_batch - features
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers

_batch_size = 150

ratio=1
weight=300
modeldes = '%s_best__300fea_3_5' % (os.path.basename(__file__).split('.')[0])

def train():
    modeldir = get_modeldir(modeldes)
    
    
    net = Net(weight)
    
    trainloader = dataloader()
    print('data complete')
    criterion = TripletLoss_cosine(margin=0.1)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=10, verbose=True,)
    alpha, num_classes, center_loss_weight = 1, 10, 1
    net.cuda()
    net.train()
    
    filelist = get_filelist('you_list1')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='you350_2dfm_npy/')
    best_MAP = 0
    for epoch in range(2000):
        running_loss = 0.
        total_prec = 0.
        NUM=1000
        for i  in range(NUM):
            #inputs, labels = data
            data_list, label_list = trainloader.get() # 50
            labels = torch.cat(label_list)
            inputs = torch.Tensor([])
            for data in data_list:
                data = torch.from_numpy(data) # (900,)
                data = data.unsqueeze(0)    #  (1, 900)
                inputs = torch.cat((inputs,data))  # (50, 900)

            inputs, labels = Variable(inputs.cuda(),requires_grad = True), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(inputs) # (50, N)
            loss , prec = criterion.forward(outputs, labels)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_prec+=prec.item()
            
            #print(prec.item())
        scheduler.step(loss)
        if epoch < 10 or epoch % 10 == 9:
            cur_model_name = modeldir + '/%d' % (epoch + 1)  
            torch.save(net, cur_model_name)
        #print('loss: %.3f '  % (running_loss/NUM), 'prec: %.3f ' %(total_prec/NUM))
        
        running_loss = 0.
        total_prec = 0.
        net.eval()
        
        dist = _test_cosine(net, two_dfm, genver, _batch_size)
        MAP, top10, rank1 = calc_MAP(dist, version,[100,350])
        if MAP > best_MAP:
            best_MAP = MAP
            torch.save(net, modeldir+'/best')
            with open(modeldir+'/best.txt','w') as f:
                f.write('%f'%MAP)
        print(epoch, MAP, top10, rank1 )
        net.train()

def get_MAP():
    '''
    # SHS100K-TEST
    filelist = get_filelist('filter_download3.3')
    version = get_version2(filelist)
    genlist, genver = generate_range(filelist, version, [8, 1, 1], 2)
    two_dfm, genver = load_2dfm(genlist)
    #version = genver
    
    print('test set is SHS100K-TEST')
    '''
    '''
    # songs2000
    filelist = get_filelist('songs2000_list')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='songs2000_2dfm_npy/')
    genlist = filelist
    print('songs2000')
    '''
    '''
    # coversong80
    filelist = get_filelist('songs80_list')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='songs80_2dfm_npy/')
    print('test set is coversong80')
    '''
    
    #youtube350
    filelist = get_filelist('you_list1')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='you350_2dfm_npy/')
    genlist = filelist
    print('test set is you350')
    
    ## model_paths = [ ]
    
    for model_path in model_paths:
        #net = torch.load(model_path)
        get_rep('filter_download3.3', 'youtube_2dfm_npy/', 'youdata2/', model_path, genlist,two_dfm, genver)
        os.system('./main_2dfm youdata2/ %d %d 300' % (len(genver), len(genver)))
        #dist = _test_cosine(net, two_dfm, genver, _batch_size)
        #print( calc_MAP(dist, version) )
       

    
if __name__=='__main__':
    #print(modeldes)
    get_MAP()
    #train()
