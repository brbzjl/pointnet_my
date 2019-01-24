##author rubing bai 18.01.2019
import os
import numpy as np
import sys
from collections import deque
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import data_prep_util
#import indoor3d_util

# Constants
data_dir = os.path.join(BASE_DIR, 'data')
indoor3d_data_dir = os.path.join(data_dir, 'IKG')
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 6]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_dir, 'IKG_indoor3d_sem_seg_hdf5_data')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return
def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=100.0, stride=50.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)


def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:3]  # data_label[:,0:6]
    #data[:, 3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])
    block_data_list = []
    block_label_list = []

    #data_batch, label_batch = room2blocks(data, label, [max_room_x,max_room_y],[0,0],
    #                                     block_data_list, block_label_list,
    #                                    num_point, block_size=1)
    t1 = time.clock()
    data_batch, label_batch = room2blocks(data, label,num_point)
    t2 = time.clock()
    print(t2-t1)
    #data_batch, label_batch = room2blocks(data, label, num_point, block_size=10.0, stride=10.0)
    data_batch, label_batch = np.concatenate(data_batch, 0), np.concatenate(label_batch, 0)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 6))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 4] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, 0:3] = data_batch
    return new_data_batch, label_batch



def room2blocks(data, label,num_point):
    level = 0
    lastlevel = False
    block_data_list = []
    block_label_list = []
    queue = deque([])
    split_num = 2
    limitmax = np.amax(data, 0)[0:2]
    limitmin = np.amin(data, 0)[0:2]
    # put the first 4 coordinates into the queue
    queue = cal_block_coordinates(queue, limitmax,limitmin, split_num,level)
    while len(queue):
        cur = 0
        last = len(queue)
        while cur<last:
            block_start_x, block_start_y,stride_x, stride_y = queue.popleft()
            cur += 1
            xcond = (data[:, 0] <= block_start_x + stride_x) & (data[:, 0] >= block_start_x)  # inside of this block
            ycond = (data[:, 1] <= block_start_y + stride_y) & (data[:, 1] >= block_start_y)
            cond = xcond & ycond
            numpointclock = np.sum(cond)
            if level>3:
                # randomly subsample data
                #print(numpointclock)
                if numpointclock > 0.5*num_point:
                    block_data = data[cond, :]
                    block_label = label[cond]
                    # randomly subsample data
                    block_data_sampled, block_label_sampled = \
                        sample_data_label(block_data, block_label, num_point)
                    block_data_list.append(np.expand_dims(block_data_sampled, 0))
                    block_label_list.append(np.expand_dims(block_label_sampled, 0))
            else:
                #limit_ofGoDeepepr = max(100 * num_point / (4 ** level),num_point)
                limit_ofGoDeepepr = max(100 * num_point / (4 ** level), 10000)
                if numpointclock > limit_ofGoDeepepr:
                    limitmax = [block_start_x + stride_x, block_start_y + stride_y]
                    limitmin = [block_start_x, block_start_y]
                    queue = cal_block_coordinates(queue,limitmax, limitmin, split_num,level)
                else:
                    ncond =  (cond==False)
                    data =data[ncond]
                    label =label[ncond]
        level += 1
        if level >4:
            break
    return block_data_list, block_label_list

def cal_block_coordinates(queue, limitmax, limitmin, split_num,level):
    scale = 0
    #in each direction the length will be divided into 3 parts
    if level == 3:
        scale = 1
    stride_x = int(np.ceil((limitmax[0] - limitmin[0]) / (split_num)))
    stride_y = int(np.ceil((limitmax[1] - limitmin[1]) / (split_num)))
    for i in range(split_num+scale):
        for j in range(split_num+scale):
            queue.append([limitmin[0] + i * stride_x/(1+scale),
                          limitmin[1] + j * stride_y/(1+scale),
                          stride_x / (1 ),stride_y/ (1 )])
    return queue

def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label
def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)
if __name__=='__main__':
    sample_cnt = 0
    for i, data_label_filename in enumerate(data_label_files):
        print(data_label_filename)
        data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=10,stride=5,
                                                     random_sample=False, sample_num=None)
        print('{0}, {1}'.format(data.shape, label.shape))
        for _ in range(data.shape[0]):
            fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

        sample_cnt += data.shape[0]
        insert_batch(data, label, i == len(data_label_files)-1)
        break
    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))
