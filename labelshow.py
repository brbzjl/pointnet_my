import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import sys
import pptk
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'data'))
output_folder = os.path.join(BASE_DIR, 'data/IKG')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
#label = np.load("/media/rubing/hdd/labels/20170331/label/170331_083409_Scanner_1_labelhist.npy")
#idx = np.where(np.max(label,axis=-1)!=0)


#path_label = '/media/rubing/hdd/labels/20170331/label/'
#path_xyz = '/media/rubing/hdd/strips/20170331/xyz/'
path_label = os.path.join(BASE_DIR, 'hdd/labels/20170331/label/')
path_xyz =os.path.join(BASE_DIR, 'hdd/strips/20170331/xyz/')
labelfiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path_label)
    for f in fnmatch.filter(files, '*_Scanner_1_labelhist.npy')]

data_label_list = []
for labelfilename in labelfiles:
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
    for i,name in enumerate(file_xyz):
        data_xyz.append(np.load(name))
    label = np.load(labelfilename)
    label_majority = np.argmax(label, axis=-1)
    #idx = np.where(np.max(label, axis=-1) == 0)
    #label_majority[idx] = -1
    data_lable_ = np.dstack((data_xyz[0], data_xyz[1], data_xyz[2]))
    data_lable_ = np.dstack((data_lable_,label_majority))
    data_label_list.append(data_lable_)
data_label = np.concatenate(data_label_list, 0)
data_label = np.reshape(data_label,(-1,4))
##remove 0 0 0 data
idx = np.where(np.max(data_label[:,0:3], axis=-1) != 0)
data_label = data_label[idx]
##remove the coordinates of the grids in utm coordinates
#utm_grid = data_label[:, 0:2]//1000
#utm_grid = utm_grid*1000
#data_label[:, 0:2] -= utm_grid
xyz_min = np.amin(data_label, axis=0)[0:3]
data_label[:, 0:3] -= xyz_min
'''
idx = np.where((data_label[:,0]) == 0)
idy = np.where((data_label[:,1]) == 0)
idz = np.where((data_label[:,2]) == 0)
'''
## save the data as a npy
xyz_max = np.amax(data_label, axis=0)[0:3]
out_filename = namepre_+'.npy'
np.save(os.path.join(output_folder, out_filename),data_label)
'''
fout = open('output_datalabel.txt', 'w')
for i in range(data_label.shape[0]):
    fout.write('%f %f %f %d\n' % \
                  (data_label[i,0], data_label[i,1], data_label[i,2],
                   data_label[i,3]))
fout.close()
'''
v = pptk.viewer(data_label[:, 0:3])