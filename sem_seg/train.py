import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
model_sem_seg = importlib.import_module('model_sem_seg')
sys.path.append(os.path.join(ROOT_DIR, 'eval/evaluate_city'))
Eval = importlib.import_module('iou')
import provider
import tf_util
import time
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

LOG_test_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_test_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

traindata_folder = '/media/rubing/hdd/data_label/IKG_hdf5_train_temp'
ALL_FILES_TRAIN = provider.getDataFiles(os.path.join(traindata_folder, 'all_files.txt'))
testdata_folder = '/media/rubing/hdd/data_label/IKG_hdf5_test_grundtruth'
ALL_FILES_TEST = provider.getDataFiles(os.path.join(testdata_folder, 'all_files.txt'))

Npoints = 6
reader_shapes = {'data': [NUM_POINT, Npoints],
                         'labels': [NUM_POINT]}
reader_dtypes = {'data': tf.float32, 'labels': tf.int32}


def read_fn(data_folder_,ALL_FILES_):
    # We define a `read_fn` and iterate through the `file_references`, which
    # can contain information about the data to be read (e.g. a file path):
    # Load ALL data
    for h5_filename in ALL_FILES_:
        data_batch, label_batch = provider.loadDataFile(os.path.join(data_folder_, h5_filename))

        for i, label in enumerate(label_batch):
            yield {'data': data_batch[i, :, (6-Npoints):6], 'labels': label}
    return


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
def log_test_string(out_dict):
    LOG_test_FOUT.write(json.dumps(out_dict)+'\n')#
    LOG_test_FOUT.flush()
    print(out_dict)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        dataset = tf.data.Dataset.from_generator(read_fn, reader_dtypes, reader_shapes, args = ([traindata_folder, ALL_FILES_TRAIN]))
        dataset = dataset.repeat(None).shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(1)
        iterator = dataset.make_initializable_iterator()
        next_dict = iterator.get_next()

        dataset_valid = tf.data.Dataset.from_generator(read_fn, reader_dtypes, reader_shapes, args = ([testdata_folder, ALL_FILES_TEST]))
        dataset_valid = dataset_valid.repeat(None).shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(1)

        iterator_valid = dataset_valid.make_initializable_iterator()
        next_dict_valid = iterator_valid.get_next()

        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = model_sem_seg.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            #pred, end_points = model_sem_seg.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            #loss = model_sem_seg.get_loss(pred, labels_pl, end_points)

            pred = model_sem_seg.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = model_sem_seg.get_loss(pred, labels_pl)
            pred_softmax = tf.nn.softmax(pred)

            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        sess.run(iterator.initializer)
        sess.run(iterator_valid.initializer)
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'pred_softmax':pred_softmax}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer, next_dict)

            
            # Save the variables to disk.
            if (epoch+1) % 10 == 0:
                log_test_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                eval_one_epoch(sess, ops, test_writer, next_dict_valid)

                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_02042019.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, next_dict):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)

    file_size = np.loadtxt(os.path.join(traindata_folder, 'room_filelist.txt'))
    print(file_size)
    num_batches = int(file_size) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    t1 = time.clock()
    for batch_idx in range(num_batches):

        if batch_idx % 500 == 0:
            t2 = time.clock()
            print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
            print(t2 - t1)
            t1 = t2
        current_data, current_label = sess.run([next_dict['data'],next_dict['labels']])

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']:  current_label,
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)

        correct = np.sum(pred_val == current_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer, next_dict_valid):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    log_test_string('----')
    label_pre_all = []
    label_gt_all = []
    file_size = np.loadtxt(os.path.join(testdata_folder, 'room_filelist.txt'))
    print(file_size)
    num_batches = int(file_size) // BATCH_SIZE
    for batch_idx in range(num_batches):
        current_data, current_label = sess.run([next_dict_valid['data'], next_dict_valid['labels']])
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        pred_label = np.argmax(pred_val, 2)  # BxN
        label_pre_all.append(pred_label)
        label_gt_all.append(current_label)
        break
    label_pre_all = np.concatenate(label_pre_all, 0)
    label_gt_all = np.concatenate(label_pre_all, 0)
    pred = np.asarray(label_pre_all.reshape((1, -1, 1)), dtype=np.uint8)
    gt = np.asarray((label_gt_all).reshape((1, -1, 1)), dtype=np.uint8)
    dict = Eval.get_iou(pred=np.asarray(label_pre_all.reshape((1, -1, 1)), dtype=np.uint8),
                        gt=np.asarray((label_gt_all).reshape((1, -1, 1)), dtype=np.uint8))
    log_test_string(dict['classScores'])
    log_test_string(dict['averageScoreClasses'])
         


if __name__ == "__main__":

    #for h5_filename in ALL_FILES:
    #    data_batch, label_batch = provider.loadDataFile(os.path.join(data_folder, h5_filename))
    #    i=1

    train()
    LOG_FOUT.close()
    LOG_test_FOUT.close()
