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
        outputs, new_sub_two_dfm = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        outputs, new_sub_two_dfm = outputs.data.cpu().numpy(), new_sub_two_dfm.data.cpu().numpy()
       
        if new_two_dfm is not None:
            new_two_dfm = np.concatenate((new_two_dfm, new_sub_two_dfm), axis=0)
            sig_out = np.concatenate((sig_out, outputs))
        else:
            new_two_dfm = new_sub_two_dfm
            sig_out = outputs
    new_two_dfm = norm(new_two_dfm)
    if verbose:
        print(np.count_nonzero(new_two_dfm) / len(new_two_dfm))
    return new_two_dfm

def get_rep(filename, in_dir, dire, net_path, genlist, two_dfm, genver):

    start_time = time.time()
    if not os.path.exists(dire):
        os.mkdir(dire)
 
    np.savetxt(dire + 'data1.txt', two_dfm) 
    _batch_size = 1000
    
    
    net = torch.load(net_path)
    # print calc_MAP(test(net, two_dfm, genver, _batch_size), genver, [100, 350]) 
    new_two_dfm = generate_rep(net, two_dfm, genver, _batch_size)
    end_time = time.time()
    print('time: %fs' % ((end_time - start_time) / len(genver)))
    np.savetxt(dire + 'data0.txt', new_two_dfm) 
    np.savetxt(dire + 'version.txt', genver, fmt='%d')

def test(net, two_dfm, genver, _batch_size, verbose=False):
    inputs, labels = torch.from_numpy(two_dfm), torch.from_numpy(genver)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)

    total, correct = 0, 0
    sig_out, new_two_dfm = None, None
    for data, target in testloader:
        inputs, labels = data, target
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs, new_sub_two_dfm = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        outputs, new_sub_two_dfm = outputs.data.cpu().numpy(), new_sub_two_dfm.data.cpu().numpy()
       
        if new_two_dfm is not None:
            new_two_dfm = np.concatenate((new_two_dfm, new_sub_two_dfm), axis=0)
            sig_out = np.concatenate((sig_out, outputs))
        else:
            new_two_dfm = new_sub_two_dfm
            sig_out = outputs
    # print correct * 1. / total
    
    new_two_dfm = norm(new_two_dfm)
    # print get_measure(two_dfm, genver)
    
    dis2d = get_dis2d4(new_two_dfm)
    if verbose:
        print( np.count_nonzero(new_two_dfm) / len(new_two_dfm))
    return dis2d  

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
        print( song_set[0, 2, 2], song_set[0, 10, 73])
        raw_input()
        song_set = song_set[:, iis, jis]

        cov = np.cov(song_set)
        sum_conv += np.linalg.cond(cov)
        
        print( cov.shape, np.linalg.cond(cov))
        raw_input()
        
    print( sum_conv / len(song_set_id))

class Net(nn.Module):
    def __init__(self, num_classes,weight):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(900, weight)
        self.fc2 = nn.Linear(weight, num_classes)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        #x = self.norm(x)
        output = self.fc2(x)
        return output, x

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


def get_weight(version):
    size = int(max(version)+1)
    C = torch.zeros(10000)
    for v in version:
        C[v]+=1
    return C
    
    


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
    # songs2000
    filelist = get_filelist('songs2000_list')
    version = get_version2(filelist)
    genlist = filelist
    two_dfm, genver = load_2dfm(filelist, in_dir='songs2000_2dfm_npy/')
    print('songs2000')
    
    '''
    # coversong80
    filelist = get_filelist('songs80_list')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='songs80_2dfm_npy/')
    print('test set is coversong80')
    
    #youtube350
    filelist = get_filelist('you_list1')
    version = get_version2(filelist)
    two_dfm, genver = load_2dfm(filelist, in_dir='you350_2dfm_npy/')
    print('test set is you350')
    '''
    model_paths = [
                 ]
    
    for model_path in model_paths:
        #net = torch.load(model_path)
        print(model_path)
        get_rep('songs2000_list', 'songs2000_2dfm_npy/', 'youdata2/', model_path, genlist,two_dfm, genver)        
        #get_rep('filter_download3.3', 'youtube_2dfm_npy/', 'youdata2/', model_path, genlist,two_dfm, genver)
        os.system('./main_2dfm youdata2/ %d %d 500' % (len(genver), len(genver)))
        #dist = _test_cosine(net, two_dfm, genver, _batch_size)
        #print( calc_MAP(dist, version) )


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('weight', type=int)
args = parser.parse_args()
weight = args.weight

_batch_size = 150
modeldes = '%s_%d_BEST' % (os.path.basename(__file__).split('.')[0], weight)

if args.train:
    modeldir = get_modeldir(modeldes)
    filelist = get_filelist('filter_download3.3')
    version = get_version2(filelist)
    genlist, genver = generate_range(filelist, version, [8, 1, 1], 0)
    two_dfm, genver = load_2dfm(genlist)
    num_class = int(max(genver)+1)
    num_class = 10000
    class_weight = get_weight(genver)
    #if ratio == 80000:
    #    ratio = len(two_dfm)
    #two_dfm, genver = two_dfm[: ratio, :], genver[: ratio]
    inputs, labels = torch.from_numpy(two_dfm), torch.from_numpy(genver)
    print(inputs.shape, labels.shape)
    
    net = Net(num_class, weight)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()#weight=class_weight.cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    alpha, num_classes, center_loss_weight = 1, 10, 1
    net.cuda()
    net.train()
    evl_filelist = get_filelist('you_list1')
    evl_version = get_version2(evl_filelist)
    evl_two_dfm, evl_genver = load_2dfm(evl_filelist, in_dir='you350_2dfm_npy/')
    best_MAP = 0;
    for epoch in range(200):
        running_loss = 0.
        for i, data in enumerate(trainloader, 0):
            x, y = data

            inputs = Variable( x.cuda() )
            labels = Variable( y.cuda() )
            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            
            outputs, _ = net(inputs)
            
            softmax_loss = criterion(outputs, labels)
            loss = softmax_loss
     
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        if epoch < 10 or epoch % 10 == 9:
            cur_model_name = modeldir + '/%d' % (epoch + 1)  
            torch.save(net, cur_model_name)
        #print( epoch,'loss: %.3f' % (running_loss / len(trainloader)))
        running_loss = 0.
    
        net.eval()
        dis2d = test(net, evl_two_dfm, evl_genver, _batch_size)
        MAP, top10, rank1 = calc_MAP(dis2d, evl_version,[100, 350])
        
        if MAP > best_MAP:
            best_MAP = MAP
            torch.save(net, modeldir+'/best')
            with open(modeldir+'/best.txt','w') as f:
                f.write('%f 10000 no weight'%MAP)
            print('best epoch!')
        print(epoch, MAP, top10, rank1 )
        net.train()
get_MAP()
