from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utility import *
import time, os
import numpy as np

def get_filelist(path):
    with open(path, 'r') as fp:
        filelist = [line.rstrip() for line in fp]     
    return filelist

def get_version2(file_list):
    labels = []
    for filename in file_list:
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        labels.append(set_id)
    return np.array(labels)
# load data
def load_2dfm(file_list, in_dir='youtube_2dfm_npy/', is_2d=False):
    inputs, labels = [], []
    for filename in file_list:
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = in_dir + filename + '.npy'
        input = np.load(in_path) 
        if is_2d:
            input = np.reshape(input, (12, 75), 'F')
        inputs.append(input)
        labels.append(set_id)
    return np.array(inputs), np.array(labels)
def generate_range(filelist, version, ratios, pos=0):
    genlist, genver, genlists, genvers, hielist = [], [], [], [], []
    nom_ratio, a_ratio, c_ratio = 1. * sum(ratios), 0., 0.
    ratios = [ratio / nom_ratio for ratio in ratios]
    ratio = ratios.pop(0)
    for i, item in enumerate(filelist):
        if len(hielist) == 0 or version[i] != version[i - 1]:
            if len(hielist) != 0 and len(hielist[-1]) == 1:
                hielist.pop(-1)
            hielist.append([])
        hielist[-1].append(item)
    tot_cnt, set_set = len(filelist), range(len(hielist))

    # print 'cnt:', tot_cnt, 'range:', (0, len(hielist))
    for set_id in set_set:
        c_ratio = 1. * len(hielist[set_id]) / tot_cnt
        if a_ratio + c_ratio < ratio or len(ratios) == 0:
            gensublist = hielist[set_id]
            genver = genver + [set_id] * len(gensublist)
            genlist = genlist + gensublist
            a_ratio += c_ratio
        else:
            ratio, a_ratio = ratios.pop(0), c_ratio
            genlists.append(genlist)
            genvers.append(genver)
            genlist, genver = gensublist, [set_id] * len(gensublist)
    genlists.append(genlist)
    genvers.append(genver)
    
    # print [len(genlists[i]) for i in range(len(genlists))]
    return genlists[pos], genvers[pos]




def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def generate_rep(model, two_dfm, genver, _batch_size, verbose=False):
     
    new_two_dfm = model.transform(two_dfm)
    new_two_dfm = norm(new_two_dfm)
    if verbose:
        print(np.count_nonzero(new_two_dfm) / len(new_two_dfm))
    return new_two_dfm

def get_rep(filename, in_dir, dire, model, genlist, two_dfm, genver):

    start_time = time.time()
    if not os.path.exists(dire):
        os.mkdir(dire)
 
    np.savetxt(dire + 'data1.txt', two_dfm) 
    _batch_size = 1000
    
    
    
    # print calc_MAP(test(net, two_dfm, genver, _batch_size), genver, [100, 350]) 
    new_two_dfm = generate_rep(model, two_dfm, genver, _batch_size)
    end_time = time.time()
    print('time: %fs' % ((end_time - start_time) / len(genver)))
    np.savetxt(dire + 'data0.txt', new_two_dfm) 
    np.savetxt(dire + 'version.txt', genver, fmt='%d')
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
    print('test set is you350')
    
    
    models = [lda50,lda100,lda200]
    i=50
    for model in models:

        get_rep('you_list1', 'you350_2dfm_npy/', 'youdata2/', model, genlist,two_dfm, genver)
        os.system('./main_2dfm youdata2/ %d %d %d' % (100, 350,i))
        i=i*2
        #dist = _test_cosine(net, two_dfm, genver, _batch_size)
        #print( calc_MAP(dist, version) )

        
        
        

lda200 = LinearDiscriminantAnalysis(n_components=200)

filelist = get_filelist('filter_download3.3')
version = get_version2(filelist)
genlist, genver = generate_range(filelist, version, [8, 1, 1], 0)
two_dfm, genver = load_2dfm(genlist)
version = genver


lda200 = lda200.fit(two_dfm, version)
print('lda200')
        
get_MAP()


