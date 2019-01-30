import argparse
import os
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
data_dir = os.path.join(ROOT_DIR, 'data')
indoor3d_data_dir = os.path.join(data_dir, 'IKG')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'eval/evaluate_city'))
pointnet_seg = importlib.import_module('pointnet_seg')
Eval = importlib.import_module('iou')
import indoor3d_util
import provider
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
#parser.add_argument('--model_path', required=True, help='model checkpoint file path')
#parser.add_argument('--dump_dir', required=True, help='dump folder path')
#parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
#parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
#parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', default=True, action='store_true', help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(BASE_DIR, 'log/model6.ckpt')
GPU_INDEX = FLAGS.gpu
DUMP_DIR = os.path.join(BASE_DIR, 'log/')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

NUM_CLASSES = 20

data_folder = '/media/rubing/hdd/data_label/IKG_hdf5_test_temp'
ALL_FILES = provider.getDataFiles(os.path.join(data_folder, 'all_files.txt'))
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = pointnet_seg.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred ,end_points = pointnet_seg.get_model(pointclouds_pl, is_training_pl)
        loss = pointnet_seg.get_loss(pred, labels_pl,end_points)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    #log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    total_correct = 0
    total_seen = 0
    label_pre_all = []
    label_gt_all = []
    for room_path in ALL_FILES:

        gt, pre = eval_one_epoch(sess, ops, room_path)
        label_pre_all.append(pre)
        label_gt_all.append(gt)
        #print('all room eval accuracy: %f'% (total_correct / float(total_seen)))
        #break
    label_pre_all = np.concatenate(label_pre_all, 0)
    label_gt_all = np.concatenate(label_pre_all, 0)
    pred = np.asarray(label_pre_all.reshape((1, -1, 1)), dtype=np.uint8)
    gt = np.asarray((label_gt_all).reshape((1, -1, 1)), dtype=np.uint8)
    dict = Eval.get_iou(pred=np.asarray(label_pre_all.reshape((1, -1, 1)), dtype=np.uint8),
                        gt=np.asarray((label_gt_all).reshape((1, -1, 1)), dtype=np.uint8))
    print (dict['classScores'])
    print (dict['averageScoreClasses'])

def eval_one_epoch(sess, ops, room_path):

    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    if FLAGS.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-3]+'_predtest.txt'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-3]+'_gttest.txt'), 'w')
    current_data, current_label = provider.loadDataFile(room_path)
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)
    idx = np.where(current_label == 255)
    current_label[idx] = 19
    # Get room dimension..

    data_size = current_data.shape[0]
    num_batches = data_size // BATCH_SIZE
    print(data_size)

    label_pre_list = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 0:6],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        #if FLAGS.no_clutter:
        #    pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
        #else:
        pred_label = np.argmax(pred_val, 2) # BxN

        label_pre_list.append(pred_label)

        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx+b, :, :]
            l = current_label[start_idx+b,:]#grundtruth

            pts[:,3] *= 200#max_room_x
            pts[:,4] *= 200#max_room_y
            pts[:,5] *= 100#max_room_z
            #pts[:,3:6] *= 255.0
            pred = pred_label[b, :]#predict label

            for i in range(NUM_POINT):
                color = indoor3d_util.g_label2color[pred[i]]
                color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
                if FLAGS.visu:
                    fout.write('v %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,3], pts[i,4], pts[i,5], color_gt[0], color_gt[1], color_gt[2]))

        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    print('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    print('eval accuracy: %f'% (total_correct / float(total_seen)))

    if FLAGS.visu:
        fout.close()
        fout_gt.close()
    return current_label, np.concatenate(label_pre_list, 0)


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    #LOG_FOUT.close()
