import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect

class WeightRandom(object):
    def __init__(self, dic):
        weights = [w for _,w in dic.items()]
        self.goods = [i for (i,(x,_)) in enumerate(dic.items())]
        self.total = sum(weights)
        self.acc = list(self.accumulate(weights))
    def accumulate(self, weights):#累和.如accumulate([10,40,50])->[10,50,100]
        cur = 0
        for w in weights:
            cur = cur+w
            yield cur
    def __call__(self):
        return self.goods[bisect.bisect_right(self.acc , random.uniform(0, self.total))]


def load_2dfm(filename, in_dir='youtube_2dfm_npy/', is_2d=False):

    set_id, version_id = filename.split('.')[0].split('_')
    set_id, version_id = int(set_id), int(version_id)
    in_path = in_dir + filename + '.npy'
    data = np.load(in_path) 

    return data, version_id


class dataloader():
    def __init__(self):
        self.dic_train = np.load('dict_class_filename_2dfm8.npy').item()
        self.dic_class_num_train = np.load('dict_class_num_2dfm8.npy').item()
        #self.dic_test = np.load('dict_class_filename_2dfm_train.npy').item()
        #self.dic_class_num_test = np.load('dict_class_num_2dfm_test.npy').item()
        self.wr = WeightRandom(self.dic_class_num_train)
        self.all_list = []
        for c,l in self.dic_train.items():
            v_list = []
            for line in l:
                data, label = load_2dfm(line)
                v_list.append(data)
            self.all_list.append(v_list)
        print(len(self.all_list))
    def get(self, num=10):
        # 返回num个数据
        label_list = []
        data_list = []
        for i in range(num):
            index = self.wr() #随机选取10个id
            #print(index)
            data_list = data_list + random.sample(self.all_list[index],5)
            label = torch.Tensor([i])
            label_list = label_list + [label,label,label,label,label]
            #path_list += random.sample(self.dic_train[index],5) #每个id随机抽取5首歌
        #data_list, label_list = load_2dfm(path_list)
        #print(path_list)
        return data_list, label_list



