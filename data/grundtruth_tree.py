import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import sys
import time
from collections import deque
# import pptk
SHOW_DATA = False
NUM_POINT = 4096
SCANER_INDEX = '1'
# --------------------------------------
# ----- PARAMETERS of HDF5 -----
# --------------------------------------
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 6]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util

sys.path.append(os.path.join(ROOT_DIR, 'data'))
# --------------------------------------
# ----- data direction  -----
# --------------------------------------
output_folder = '/media/rubing/hdd/data_label/IKG_hdf5_test_grundtruth_tree'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
path_label = '/media/rubing/hdd/groundtruth/'  # labels/20170405/label/' #20170405 for test ,20170331 for train
path_xyz = '/media/rubing/hdd/strips/'  # 20170405/xyz/'
output_all_filelist_dir = os.path.join(output_folder, 'all_files.txt')
# path_label = os.path.join(ROOT_DIR, 'hdd/labels/20170331/label/')
# path_xyz =os.path.join(ROOT_DIR, 'hdd/strips/20170331/xyz/')

# --------------------------------------
fout_all_files = open(output_all_filelist_dir, 'w')
output_filename_prefix = os.path.join(output_folder, ''.join([SCANER_INDEX, '_ply_data_all']))


def combine_label_data(path_label, path_xyz, output_folder):
    labelfiles = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(path_label)
                  for f in fnmatch.filter(files, ''.join(['*_label.npy']))]
    sample_cnt = 0
    for i, labelfilename in enumerate(labelfiles):
        # print(len(labelfilename))
        labelfilenamePart = labelfilename.split('/')
        filename_ = labelfilenamePart[-1].split('_')
        namepre_ = filename_[0] + '_' + filename_[1] + '_', filename_[2] + '_', filename_[3] + '_'
        namepre_ = ''.join(namepre_)
        xname_ = namepre_ + 'worldx.npy'
        yname_ = namepre_ + 'worldy.npy'
        zname_ = namepre_ + 'worldz.npy'

        file_xyz = [xname_, yname_, zname_]
        data_xyz = []
        for i, name in enumerate(file_xyz):
            try:
                xyzfiles = [os.path.join(dirpath, f)
                              for dirpath, dirnames, files in os.walk(path_xyz)
                              for f in fnmatch.filter(files, name)]
                data_xyz.append(np.load(xyzfiles[0]))
            except IOError:
                print('no related xyz data' + namepre_)

        if not data_xyz:
            continue

        label = np.load(labelfilename)
        label_majority = np.argmax(label, axis=-1)
        idxlabel = np.where(np.max(label, axis=-1) != 0)
        #remove no all zero labels
        label_majority = label_majority[idxlabel]
        #collect data from x y z files
        data_label = np.dstack((data_xyz[0][idxlabel], data_xyz[1][idxlabel], data_xyz[2][idxlabel]))
        data_label = np.reshape(data_label, (-1, 3))

        try:
            label_majority = np.expand_dims(label_majority, axis=1)
            print(data_label.shape,label_majority.shape)
            data_label = np.concatenate((data_label, label_majority), axis = 1)
            # remove the all zero points
            idx = np.where(np.max(data_label[:, 0:3], axis=-1) != 0)
            data_label = data_label[idx]
            # z_road is the z mean value of the road
            road_idx = np.where(label_majority == 0)
            z_road = np.mean(data_label[road_idx, 2])
            if z_road is None:
                z_road = 95
            # minmun value of xyz
            xy_min = np.amin(data_label, axis=0)[0:2]

            data_label[:, 0:2] -= xy_min[0:2]  # take care of this z value
            data_label[:, 2] -= z_road
        except ValueError:
            print('the dim in some aixs is different')


        try:
            data, label = room2blocks_plus_normalized(data_label, NUM_POINT)
            # v = pptk.viewer(data[1:10,:, 0:3])
            print('{0}, {1}'.format(data.shape, label.shape))
            sample_cnt += data.shape[0]
            insert_batch(data, label, i == len(labelfiles) - 1)
        except ValueError:
            print('the blocks somehow is empty')


    print("Total samples: {0}".format(sample_cnt))
    fout_room = open(os.path.join(output_folder, 'room_filelist.txt'), 'w')
    fout_room.write(str(sample_cnt))
    fout_room.close()
    fout_all_files.close()


def room2blocks_plus_normalized(data_label, num_point):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:3]  # data_label[:,0:6]
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    t1 = time.clock()
    data_batch, label_batch = room2blocks(data, label, num_point)
    t2 = time.clock()
    print(t2 - t1)
    # v = pptk.viewer(data_batch[5, :, 0:3])

    new_data_batch = np.zeros((data_batch.shape[0], num_point, 6))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 4] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2] / max_room_z
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
                    ncond = (cond == False)
                    data =data[ncond]
                    label =label[ncond]
        level += 1
        if level >4:
            break
    data_batch, label_batch = np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)
    return data_batch, label_batch


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
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size + data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size + data_size] = label
        buffer_size += data_size
    else:  # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert (capacity >= 0)
        if capacity > 0:
            h5_batch_data[buffer_size:buffer_size + capacity, ...] = data[0:capacity, ...]
            h5_batch_label[buffer_size:buffer_size + capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        fout_all_files.write(h5_filename + '\n')
        # -------------store the data to show -----------------
        if SHOW_DATA:
            fout = open(output_filename_prefix + '_' + str(h5_index) + '.txt', 'w')
            for batch in range(h5_batch_data.shape[0]):
                for p in range(h5_batch_data.shape[1]):
                    fout.write('v %f %f %f \n' % (h5_batch_data[batch, p, 0], h5_batch_data[batch, p, 1],
                                                  h5_batch_data[batch, p, 2]))
            fout.close()
        # --------------------------------------------------
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...],
                               data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        fout_all_files.write(h5_filename + '\n')
        # -------------store the data to show -----------------
        if SHOW_DATA:
            fout = open(output_filename_prefix + '_' + str(h5_index) + '.txt', 'w')
            for batch in range(h5_batch_data.shape[0]):
                for p in range(h5_batch_data.shape[1]):
                    fout.write('v %f %f %f \n' % (h5_batch_data[batch, p, 0], h5_batch_data[batch, p, 1],
                                                  h5_batch_data[batch, p, 2]))
            fout.close()
        # --------------------------------------------------
        h5_index += 1
        buffer_size = 0
    return


if __name__ == '__main__':
    combine_label_data(path_label, path_xyz, output_folder)
