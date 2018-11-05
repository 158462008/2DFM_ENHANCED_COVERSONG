import numpy as np
# from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool
import os, random
import torch
import torch.utils
from torch.autograd import Variable
# from scipy.spatial.distance import cosine
# other
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

def generate_test(filelist, version, set_cnt, ver_cnt, set_set=None, ratio=None):
    genlist, genver, hielist = [], [], []
    for i, item in enumerate(filelist):
        if len(hielist) == 0 or version[i] != version[i - 1]:
            hielist.append([])
        hielist[-1].append(item)
    if ratio is None:
        if set_set is None:  
            import random
            set_set = random.sample(range(len(hielist)), set_cnt)
    else:
        if ratio > 0.5:
            set_set = range(int(len(hielist) * ratio))
        else:
            set_set = range(int(len(hielist) * (1 - ratio)), len(hielist))
    
    for set_id in set_set:
        gensublist = hielist[set_id][: ver_cnt if ver_cnt < len(hielist) else len(hielist)]
        genver = genver + [set_id] * len(gensublist)
        genlist = genlist + gensublist

    return genlist, genver

def generate_subset(set_list=[], N=80):
    sub_list, final_set_list = [], []
    if not len(set_list):
        fullset_list = os.listdir('../old_youtube')
        set_list = random.sample(fullset_list, N)
    for set_id in set_list:
        if set_id[0].isdigit() and len(os.listdir('../old_youtube/' + set_id)) != 1:
            ver_list = os.listdir('../old_youtube/' + set_id)
            sub_list = sub_list + ver_list
            final_set_list.append(set_id)
    with open('set_list', 'w') as fp:
        for set_id in final_set_list:
            fp.write(set_id + '\n')
    with open('sub_list', 'w') as fp:
        for ver in sub_list:
            fp.write(ver + '\n')
    return final_set_list, sub_list

# computation
def calc_quick_precision(array1d):
    precision = 0
    for i in xrange(0, 1000, 2):
        if array1d[i] < array1d[i+1]:
            precision += 1
    print(precision)

def calc_MAP(array2d, version, que_range=None):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0
    
    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):
            if version[u] == version[v]:
                version_cnt += 1
                per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)


def calc_error(tran_A, tran_X1, tran_X2, m, tao, h_list):
    N = min(len(tran_X1), len(tran_X2))
    if N <= (m-1)*tao+min(h_list):
        print('prediction error. N must not smaller that (m-1)*tao+h')
    tran_R2 = construct_tran_R(tran_X2, m, tao, 1)
    tran_e_X2 = np.matrix(tran_R2) * np.matrix(tran_A)
    error_list = []
    for h in h_list:
        var = np.array([np.var(tran_X2[(m-1)*tao+h: N][i], ddof=1) for i in range(12)])
        square_error = tran_e_X2[:N-(m-1)*tao-h] - tran_X2[(m-1)*tao+h: N]
        square_error = np.multiply(square_error, square_error)
        error = 0
        for i in range(12):
            error += np.sum(square_error[:, i]) / (var[i] if var[i] else 1)
        error /= ((N-(m-1)*tao-h) * 12)
        error_list.append(error)
    return error_list

def construct_tran_R(tran_X, m, tao, h):
    N = len(tran_X)
    if N <= (m-1)*tao+1:
        print('construct R error. N must not smaller that (m-1)*tao+1')
    tran_Z = np.array([tran_X[i-(m-1)*tao: i+1: tao][::-1].ravel() for i in range((m-1)*tao, N-h)])
    tran_Z = np.insert(tran_Z, tran_Z.shape[1], 1, axis=1)
    return tran_Z

def calc_tran_A(tran_X, m, tao, h):
    N = len(tran_X)
    if N <= (m-1)*tao+h:
        print( 'calc A error. N must not smaller that (m-1)*tao+h')
    tran_R = np.matrix(construct_tran_R(tran_X, m, tao, h))
    tran_t_X = np.matrix(tran_X[(m-1)*tao+h: N])
    print( tran_R)
    print( tran_R.getT() * tran_R)
    print( np.linalg.det(tran_R.getT() * tran_R))
    # print ((tran_R.getT() * tran_R).getI())
    print( tran_t_X.getT() * tran_R * ((tran_R.getT() * tran_R).getI()))
    return np.linalg.lstsq(tran_R, tran_t_X)[0]

# def get_dis2d(seqs, condition, transition=12):
#     if condition.get('norm') <> None:
#         dis2d = np.zeros((len(seqs), len(seqs)))
#         for i, seq1 in enumerate(seqs):
#             for j, seq2 in enumerate(seqs):
#                 dis2d[i][j] = min(np.linalg.norm(seq1 - np.roll(seq2, k, axis=0), ord=condition['norm']) for k in xrange(transition))
#     else:
#         dis2d = np.ones((len(seqs), len(seqs)))
#         for k in xrange(transition):
#             dis2d = np.minimum(dis2d, cdist(seqs, np.roll(seqs, k, axis=1), 'cosine'))
#     return dis2d
# 
# def get_dis2d3(seqs, is_cos=False):
#     start_time = time.time()
#     dis2d = np.zeros((len(seqs), len(seqs)))
#     for i, seq1 in enumerate(seqs):
#         for j, seq2 in enumerate(seqs):
#             if is_cos:
#                 dis2d[i][j] = cosine(seq1, seq2)
#             else:
#                 dis2d[i][j] = np.linalg.norm(seq1 -seq2)
#     end_time = time.time()
#     print 'time: %fs' % (end_time - start_time)
#     return dis2d

def get_dis2d4(seqs, verbose=False):
    start_time = time.time()
    dis2d = np.zeros((len(seqs), len(seqs)))
    for i, seq1 in enumerate(seqs):
        idx = np.where(seq1 != 0)
        x = seq1[idx].squeeze()
        for j, seq2 in enumerate(seqs):
            y = seq2[idx].squeeze()
            dis2d[i][j] = 1 - np.dot(x, y)
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d

def get_dis2d7(seqs, chunks, verbose=False):
    start_time = time.time()
    N = len(chunks)
    dis2d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(len(chunks[i])):
                seq1, seq2 = seqs[chunks[i][0] + k], seqs[chunks[j][0] + k]
                dis2d[i][j] += np.linalg.norm(seq1 - seq2)
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d

def get_dis2d6(seqs, chunks, verbose=False):
    start_time = time.time()
    N = len(chunks)
    dis2d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(*chunks[i]):
                for l in range(*chunks[j]):
                    seq1, seq2 = seqs[k], seqs[l]
                    dis2d[i][j] += np.linalg.norm(seq1 - seq2)
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d

def get_dis2d5(seqs, chunks, verbose=False):
    start_time = time.time()
    N = len(chunks)
    dis2d = np.empty((N, N))
    dis2d[:] = np.inf
    for i in range(N):
        for j in range(N):
            for k in range(*chunks[i]):
                for l in range(*chunks[j]):
                    seq1, seq2 = seqs[k], seqs[l]
                    dis2d[i][j] = np.min((np.linalg.norm(seq1 - seq2), dis2d[i][j]))
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d

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

def load_2dfm_seq(file_list, Cin=200, in_dir='2dfm_seq_npy/', is_norm=False, is_con=True, is_pow=False, verbose=False, is_del_sep=None, is_sam_sep=None):
    inputs, labels, chunks = [], [], []
    for i, filename in enumerate(file_list):
        if verbose and i % 10000 == 0:
            print( i)
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = in_dir + filename + '.npy'
        if not os.path.exists(in_path):
            continue
        input = np.load(in_path)
             
        if is_norm:
            input = input - np.mean(input, axis=0)
            input = input / np.std(input, axis=0)
        if is_con:
            input = np.concatenate((input, input[:, :11]), axis=1)
        if is_pow:
            input = np.power(input, 2)
        if input.shape[0] < Cin:
            input = np.concatenate((input, np.zeros((Cin - input.shape[0], input.shape[1]))), axis=0)
        length = len(input)

        if is_del_sep:
            hop, win = is_del_sep
            l = len(inputs)
            for i in range(0, len(input), hop):
                if i + win <= len(input):
                    inputs.append(input[i: i + win, :])
                    labels.append(set_id)
            r = len(inputs)
            chunks.append((l, r))
        elif is_sam_sep:
            num = is_sam_sep - 1
            l = len(inputs)
            ls, rs = 0, Cin
            for i in range(num + 1):
                inputs.append(input[ls: rs, :])
                labels.append(set_id + i * 10000)
                if num:
                    ls, rs = ls + (length - Cin) / num, rs + (length - Cin) / num
            r = len(inputs)
            chunks.append((l, r))
        elif Cin != 0:
            input = input[: Cin, :] 
            inputs.append(input)
            labels.append(set_id)
    if is_del_sep or is_sam_sep:
        return np.array(inputs), np.array(labels), chunks
    elif Cin != 0:
        return np.array(inputs), np.array(labels)

def load_sub_seq(file_list, H=75, step=75, in_dir='2dfm_seq_npy/'):
    inputs, labels, chunk = [], [], []
    for i, filename in enumerate(file_list):
        if i % 10000 == 0:
            print( i)
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = in_dir + filename + '.npy'
        input = np.load(in_path)
        chunk.append(0)
        for i in xrange(0, input.shape[0] - H, step):
            inputs.append(input[i: i + H, :])
            labels.append(set_id)
            chunk[-1] += 1
    return np.array(inputs), np.array(labels), np.array(chunk)


def generate_test_pairs(feas, labels):
    pair_fea, pair_label = [], []
    for i, _ in enumerate(feas):
        for j, _ in enumerate(feas): 
            pair_fea.append(np.concatenate((feas[i: i + 1, :], feas[j: j + 1, :])))
            if labels[i] == labels[j]:
                pair_label.append(1)
            else:
                pair_label.append(0)
    return np.array(pair_fea), np.array(pair_label)


def generate_pairs(feas, labels):
    r_inv, i2ver, ver_cnt = [-1], [], 1
    pair_fea, pair_labels = [], []
    # 1 means similar, otherwise
    for i, label in enumerate(labels):
        i2ver.append(ver_cnt)
        if i + 1 < len(labels):
            if label != labels[i + 1]:
                r_inv.append(i)
                ver_cnt += 1
        else:
            r_inv.append(i)
    full_rang = range(len(feas))
    for i, fea in enumerate(feas):
        ver = i2ver[i]
        l, r = r_inv[ver - 1] + 1, r_inv[ver]
        rang = range(l, r + 1)
        if l == r:
            j = i
        else:
            while True:
                j = random.choice(rang)
                if i != j:
                    break
        while True:
            k = random.choice(full_rang)
            if k < l or k > r:
                break
        if i < 20:
            print( i, j, k)
        pair_fea.append(np.concatenate((feas[i: i + 1], feas[j: j + 1])))
        pair_fea.append(np.concatenate((feas[i: i + 1], feas[k: k + 1])))
        pair_labels = pair_labels + [1, 0]
    return np.array(pair_fea), np.array(pair_labels)

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
'''
def test_metric(net, two_dfm, genver, _batch_size, verbose=False):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(two_dfm), torch.from_numpy(genver))
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)
    reps = None
    for data, _ in testloader:
        inputs = data
        inputs = Variable(inputs.cuda())
        n_reps = net.get_rep(inputs)
        n_reps = n_reps.data.cpu().numpy()
        if reps is not None:
            reps = np.concatenate((reps, n_reps))
        else:
            reps = n_reps 
    
    inputs, _ = generate_test_pairs(reps, genver)
    dataset = MyDataset(inputs, np.tile(genver, len(genver)))
    testloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=False)
    sims = None
    
    for xs, ys, _ in testloader:
        xs, ys = Variable(xs.cuda()), Variable(ys.cuda())
        n_sims = net.get_sim(xs, ys)
        n_sims = n_sims.data.cpu().numpy()
        if sims is not None:
            sims = np.concatenate((sims, n_sims))
        else:
            sims = n_sims
    # sims = sims[:, 0]
    return sims.reshape(len(genver), len(genver))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feas, labels):
        self.feas, self.labels = feas, labels

    def __len__(self):
        return len(self.feas)

    def __getitem__(self, idx):
        return self.feas[idx, 0], self.feas[idx, 1], self.labels[idx]
'''
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
        outputs, new_sub_two_dfm = outputs.data.cpu().numpy(), new_sub_two_dfm.data.cpu().numpy()
        if new_two_dfm is not None:
            new_two_dfm = np.concatenate((new_two_dfm, new_sub_two_dfm), axis=0)
            sig_out = np.concatenate((sig_out, outputs))
        else:
            new_two_dfm = new_sub_two_dfm
            sig_out = outputs
    new_two_dfm = norm(new_two_dfm)
    dis2d = get_dis2d4(new_two_dfm)
    if verbose:
        print( np.count_nonzero(new_two_dfm) / len(new_two_dfm))
    return dis2d  



