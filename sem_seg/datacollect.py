import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import sys
import time
#import pptk
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
output_folder = '/media/rubing/hdd/data_label/IKG_hdf5_test'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
path_label = '/media/rubing/hdd/labels/20170405/label/' #20170405 for test ,20170331 for train
path_xyz = '/media/rubing/hdd/strips/20170405/xyz/'
output_all_filelist_dir = os.path.join(output_folder, 'all_files.txt')
#path_label = os.path.join(ROOT_DIR, 'hdd/labels/20170331/label/')
#path_xyz =os.path.join(ROOT_DIR, 'hdd/strips/20170331/xyz/')

# --------------------------------------
fout_all_files = open(output_all_filelist_dir, 'w')
output_filename_prefix = os.path.join(output_folder, ''.join([SCANER_INDEX,'_ply_data_all']))

def combine_label_data(path_label,path_xyz,output_folder):

    labelfiles = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(path_label)
        for f in fnmatch.filter(files, ''.join(['*_Scanner_',SCANER_INDEX, '_labelhist.npy']))]

    sample_cnt = 0
    for i,labelfilename in enumerate(labelfiles):
        #print(len(labelfilename))
        labelfilenamePart = labelfilename.split('/')
        filename_ = labelfilenamePart[-1].split('_')
        namepre_ = filename_[0]+'_'+filename_[1]+'_',filename_[2]+'_',filename_[3]+'_'
        namepre_ = ''.join(namepre_)
        xname_ = namepre_ + 'worldx.npy'
        yname_ = namepre_ + 'worldy.npy'
        zname_ = namepre_ + 'worldz.npy'

        file_xyz = [path_xyz + xname_, path_xyz + yname_,path_xyz + zname_]
        data_xyz = []
        for i, name in enumerate(file_xyz):
            try:
                data_xyz.append(np.load(name))
            except IOError:
                print('no related xyz data' + namepre_)

        if not data_xyz:
            continue

        label = np.load(labelfilename)
        label_majority = np.argmax(label, axis=-1)
        idxlabel = np.where(np.max(label, axis=-1) == 0)
        label_majority[idxlabel] = -1
        
        data_label = np.dstack((data_xyz[0], data_xyz[1], data_xyz[2]))
        # z_road is the z mean value of the road
        road_idx = np.where(label_majority == 0)
        z_road = np.mean(data_label[road_idx][:,2])

        data_label = np.reshape(data_label, (-1, 3))
        idx = np.where(np.max(data_label[:, 0:3], axis=-1) != 0)
        # minmun value of xyz
        xy_min = np.amin(data_label[idx], axis=0)[0:2]

        data_label[:, 0:2] -= xy_min[0:2]#take care of this z value
        data_label[:, 2] -= z_road
        data_label = np.reshape(data_label, (3000,-1,3))

        try:
            data_label = np.dstack((data_label,label_majority))
            data, label = room2blocks_plus_normalized(data_label, NUM_POINT, block_size=10, stride=5)

            #v = pptk.viewer(data[1:10,:, 0:3])
            print('{0}, {1}'.format(data.shape, label.shape))
            sample_cnt += data.shape[0]
            insert_batch(data, label, i == len(labelfiles) - 1)
        except ValueError:
            print('the dim in some aixs is different')

    print("Total samples: {0}".format(sample_cnt))
    fout_room = open(os.path.join(output_folder, 'room_filelist.txt'), 'w')
    fout_room.write(str(sample_cnt))
    fout_room.close()
    fout_all_files.close()
def room2blocks_plus_normalized(data_label, num_point, block_size, stride):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    t1 = time.clock()
    data_batch, label_batch = room2blocks(data_label ,num_point, block_size, stride)
    t2 = time.clock()
    print(t2-t1)
    #v = pptk.viewer(data_batch[5, :, 0:3])
    xyz_max = np.amax(np.amax(data_label, axis=0), axis=0)[0:3]
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 6))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0] / xyz_max[0]
        new_data_batch[b, :, 4] = data_batch[b, :, 1] / xyz_max[1]
        new_data_batch[b, :, 5] = data_batch[b, :, 2] / xyz_max[2]
    new_data_batch[:, :, 0:3] = data_batch
    return new_data_batch, label_batch
def room2blocks(data_label, num_point, block_size, stride):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping patern.
    """
    length = len(data_label[0])
    assert (stride <= block_size)
    # Collect blocks
    block_data_list = []
    block_label_list = []
    # Get num of blocks
    num_block = int(np.ceil((length - block_size) / stride)) + 1
    for i in range(num_block):
        block_temp = data_label[:, i*stride:(i*stride + block_size),:]
        block_temp = np.reshape(block_temp, (-1, 4))
        idx = np.where(np.min(block_temp[:, 0:3], axis=-1) >= -0.2)
        block_temp = block_temp[idx]
        #remove the data which its label value is -1
        idx_minuslabel = np.where(block_temp[:, -1] != -1)
        block_temp = block_temp[idx_minuslabel]

        #xyz_min = np.amin(block_temp, axis=0)[0:3]
        #xyz_min[0:2] -= [block_size / 2,block_size / 2]
        #block_data = block_temp[:,0:3]-xyz_min
        block_data = block_temp[:, 0:3]
        block_label = block_temp[:,-1]
        if len(block_label) < 100:  # discard block if there are less than 100 pts.
            continue
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)
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
        fout_all_files.write(h5_filename + '\n')
        #-------------store the data to show -----------------
        if SHOW_DATA:
            fout = open(output_filename_prefix + '_' + str(h5_index) + '.txt', 'w')
            for batch in range(h5_batch_data.shape[0]):
                for p in range(h5_batch_data.shape[1]):
                    fout.write('v %f %f %f \n' % (h5_batch_data[batch, p, 0], h5_batch_data[batch, p, 1],
                                                  h5_batch_data[batch, p, 2]))
            fout.close()
        #--------------------------------------------------
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
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

