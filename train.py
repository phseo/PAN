"""
Code for producing the resutls for the FFN-SPN network

# we ran the following tests:
"""

from __future__ import division
import numpy as np
import theano.tensor as T
import theano

theano.config.floatX = 'float32'

import lasagne
from libs.confusionmatrix import ConfusionMatrix
from libs.spatialelem import *
from libs.spatialatt import *
from libs.theano_util import *
from libs.batch_norm import *
import os

import logging
import argparse
import uuid
np.random.seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument("-datasize", type=int, default=50000)
parser.add_argument("-lr", type=str, default="0.003")
parser.add_argument("-decayinterval", type=int, default=10)
parser.add_argument("-decayfac", type=float, default=1.5)
parser.add_argument("-nodecay", type=int, default=30)
parser.add_argument("-optimizer", type=str, default='rmsprop')
parser.add_argument("-dropout", type=float, default=0.5)
parser.add_argument("-datapath", type=str, default='data/mdist/mdist.npz')
args = parser.parse_args()

np.random.seed(123)
TOL = 1e-5
num_batch = 200
dim = 100
channel=3
num_classes = 5
NUM_EPOCH = 300
LR = float(args.lr)
MONITOR = False
MAX_NORM = 5.0
LOOK_AHEAD = 50
DATA_SIZE = args.datasize

num_questions = 10

output_folder = "logs_%d_lr_%s/" % (DATA_SIZE-20000, args.lr)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(output_folder, "results.log"), mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info('#'*80)
for name, val in sorted(vars(args).items()):
    sep = " "*(35 - len(name))
    logger.info("#{}{}{}".format(name, sep, val))
logger.info('#'*80)


org_drp = args.dropout
sh_drp = theano.shared(lasagne.utils.floatX(args.dropout))

M = T.matrix()
W_ini = lasagne.init.GlorotUniform()
W_ini_gru = lasagne.init.GlorotUniform()
W_proc_ini = lasagne.init.GlorotUniform()
W_class_init = lasagne.init.GlorotUniform()

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    mnist_sequence = args.datapath

    from lasagne.layers import dnn
    conv = dnn.Conv2DDNNLayer
    pool = lasagne.layers.MaxPool2DLayer
elif _platform == "darwin":
    mnist_sequence = args.datapath
    conv = lasagne.layers.Conv2DLayer
    pool = lasagne.layers.MaxPool2DLayer

print "Filename:", mnist_sequence

data = np.load(mnist_sequence)

imgs = data['data']/data['data'].max().astype(theano.config.floatX)
#if dim != 100:
#    imgs = batch_imresize(imgs, dim/100.)

refs = data['refs']
qs = data['questions'].astype('int32')
ys = data['labels'].astype('int32')


xref_train, xref_valid, xref_test = refs[:DATA_SIZE-20000], refs[DATA_SIZE-20000:DATA_SIZE-10000], refs[DATA_SIZE-10000:DATA_SIZE]
xq_train, xq_valid, xq_test = qs[:DATA_SIZE-20000], qs[DATA_SIZE-20000:DATA_SIZE-10000], qs[DATA_SIZE-10000:DATA_SIZE]
y_train, y_valid, y_test = ys[:DATA_SIZE-20000].astype('int32'), ys[DATA_SIZE-20000:DATA_SIZE-10000].astype('int32'), ys[DATA_SIZE-10000:DATA_SIZE].astype('int32')

test_scale = data['scales'][DATA_SIZE-10000:DATA_SIZE]

batches_train = xref_train.shape[0] // num_batch
batches_valid = xref_valid.shape[0] // num_batch

sym_xim = T.tensor4(name='sym_xim')
sym_xq = T.ivector(name='sym_xq')
sym_y = T.ivector(name='sym_y')

##############################
#   LOCALIZATION NETWORK
##############################
# image part of localization
l_in = lasagne.layers.InputLayer((None, channel, dim, dim), input_var=sym_xim)
l_q_in = lasagne.layers.InputLayer((None,), input_var=sym_xq)

l_conv0 = conv(l_in, num_filters=32, filter_size=(3, 3),
                   name='l_conv0_loc', W=W_ini)
l_pool0 = pool(l_conv0, pool_size=(2, 2))
l_satt0 = satt(l_pool0, l_q_in, (49, 49), l_name='l_satt0', nonlinearity=lasagne.nonlinearities.sigmoid, dropout=False, context_box=(3,3))
l_bn0 = BatchNormLayer(l_satt0)

l_conv1 = conv(l_bn0, num_filters=32, filter_size=(3, 3),
                   name='l_conv1_loc', W=W_ini)
l_pool1 = pool(l_conv1, pool_size=(2, 2))
l_satt1 = satt(l_pool1, l_q_in, (23, 23), l_name='l_satt1', nonlinearity=lasagne.nonlinearities.sigmoid, dropout=False, context_box=(3,3))
l_bn1 = BatchNormLayer(l_satt1)

l_conv2 = conv(l_bn1, num_filters=32, filter_size=(3, 3),
                   name='l_conv2_loc', W=W_ini)
l_pool2 = pool(l_conv2, pool_size=(2, 2))
l_satt2 = satt(l_pool2, l_q_in, (10, 10), l_name='l_satt2', nonlinearity=lasagne.nonlinearities.sigmoid, dropout=False, context_box=(3,3))
l_bn2 = BatchNormLayer(l_satt2)

l_conv3 = conv(l_bn2, num_filters=32, filter_size=(3, 3),
                   name='l_conv3_loc', W=W_ini)
l_pool3 = pool(l_conv3, pool_size=(2, 2))
l_satt3, l_att = satt(l_pool3, l_q_in, (4, 4), l_name='l_satt3', rtn_att_map=True, dropout=False)

l_avg_out = lasagne.layers.Pool2DLayer(l_satt3, (4, 4), mode='average_exc_pad')

l_bn2_out = BatchNormLayer(l_avg_out)
l_drp_out = lasagne.layers.DropoutLayer(l_bn2_out)

l_lin_out = lasagne.layers.DenseLayer(l_drp_out, num_units=num_classes,
                                      W=W_class_init,
                                      name='class1', nonlinearity=lasagne.nonlinearities.softmax)
l_out = l_lin_out


#test = lasagne.layers.get_output(l_out).eval({sym_xim: imgs[xref_train[:num_batch]], sym_xq:xq_train[:num_batch]})

output_train = lasagne.layers.get_output(l_out, deterministic=False)
output_eval, l_A_eval = lasagne.layers.get_output([l_out, l_att], deterministic=True)



# cost
cost = T.nnet.categorical_crossentropy(output_train+TOL, sym_y)
cost = T.mean(cost)


all_params = lasagne.layers.get_all_params(l_out)
trainable_params = lasagne.layers.get_all_params(l_out, trainable=True)


#for p in trainable_params:
#    print p.name

all_grads = T.grad(cost, trainable_params)
sh_lr = theano.shared(lasagne.utils.floatX(LR))

# adam works with lr 0.001
updates, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_norm=MAX_NORM, return_norm=True)

if args.optimizer == 'rmsprop':
    updates = lasagne.updates.rmsprop(updates, trainable_params,
                                      learning_rate=sh_lr)
elif args.optimizer == 'adam':
    updates = lasagne.updates.adam(updates, trainable_params,
                                   learning_rate=sh_lr)

if MONITOR:
    add_output = all_grads + updates.values()

    f_train = theano.function([sym_xim, sym_xq, sym_y], [cost, output_train, norm
                                               ] + add_output,
                              updates=updates)
else:
    f_train = theano.function([sym_xim, sym_xq, sym_y], [cost, output_train, norm],
                              updates=updates)
f_eval = theano.function([sym_xim, sym_xq], [output_eval, l_A_eval])

best_valid = 0
result_test = 0
look_count = LOOK_AHEAD

cost_train_lst = []
last_decay = 0
scaleTotalNum = []
for i in range(5):
    scaleTotalNum.append(np.logical_and(test_scale > (i+1)*0.5, test_scale <= (i+2)*0.5).sum())
scaleTotalNum = np.array(scaleTotalNum)

for epoch in range(NUM_EPOCH):
    # eval train
    shuffle = np.random.permutation(xref_train.shape[0])

    for i in range(batches_train):
        idx = shuffle[i*num_batch:(i+1)*num_batch]
        xim_batch = imgs[xref_train[idx]]
	xq_batch = xq_train[idx]
        y_batch = y_train[idx]
        train_out = f_train(xim_batch, xq_batch, y_batch)
        cost_train, _, train_norm = train_out[:3]
#        print cost_train
        if MONITOR:
            print str(i) + "-"*44 + "GRAD NORM  \t UPDATE NORM \t PARAM NORM"
            all_mon = train_out[3:]
            grd_mon = train_out[:len(all_grads)]
            upd_mon = train_out[len(all_grads):]
            for pm, gm, um in zip(trainable_params, grd_mon, upd_mon):
                if '.b' not in pm.name:
                    pad = (40-len(pm.name))*" "
                    print "%s \t %.5e \t %.5e \t %.5e" % (
                        pm.name + pad,
                        np.linalg.norm(gm),
                        np.linalg.norm(um),
                        np.linalg.norm(pm.get_value())
                    )
        cost_train_lst += [cost_train]

    conf_train = ConfusionMatrix(num_classes)
    for i in range(xref_train.shape[0] // 1000):
        probs_train, _ = f_eval(imgs[xref_train[i*1000:(i+1)*1000]], xq_train[i*1000:(i+1)*1000])
        preds_train_flat = probs_train.reshape((-1, num_classes)).argmax(-1)
        conf_train.batch_add(
            y_train[i*1000:(i+1)*1000].flatten(),
            preds_train_flat
        )

    if last_decay > args.decayinterval and epoch > args.nodecay:
        last_decay = 0
        old_lr = sh_lr.get_value(sh_lr)
        new_lr = old_lr / args.decayfac
        sh_lr.set_value(lasagne.utils.floatX(new_lr))
        print "Decay lr from %f to %f" % (float(old_lr), float(new_lr))
    else:
        last_decay += 1

    # valid
    conf_valid = ConfusionMatrix(num_classes)
    for i in range(batches_valid):
        xim_batch = imgs[xref_valid[i*num_batch:(i+1)*num_batch]]
	xq_batch = xq_valid[i*num_batch: (i+1)*num_batch]
        y_batch = y_valid[i*num_batch:(i+1)*num_batch]
        probs_valid, _ = f_eval(xim_batch, xq_batch)
        preds_valid_flat = probs_valid.reshape((-1, num_classes)).argmax(-1)
        conf_valid.batch_add(
            y_batch.flatten(),
            preds_valid_flat
        )

    # test
    conf_test = ConfusionMatrix(num_classes)
    batches_test = xref_test.shape[0] // num_batch
    all_y, all_preds = [], []
    scaleCorrNum = [0]*5
    for i in range(batches_test):
        xim_batch = imgs[xref_test[i*num_batch:(i+1)*num_batch]]
	xq_batch = xq_test[i*num_batch:(i+1)*num_batch]
        y_batch = y_test[i*num_batch:(i+1)*num_batch]
        scale_batch = test_scale[i*num_batch:(i+1)*num_batch]
        probs_test, A_test = f_eval(xim_batch, xq_batch)
        preds_test_flat = probs_test.reshape((-1, num_classes)).argmax(-1)
        conf_test.batch_add(
            y_batch.flatten(),
            preds_test_flat
        )

        for j in range(5):
            eq = y_batch.flatten() == preds_test_flat
            scaleCorrNum[j] += eq[np.logical_and(scale_batch > (j+1)*0.5, scale_batch <= (j+2)*0.5)].sum()

        all_y += [y_batch]
        all_preds += [probs_test.argmax(-1)]

    scaleCorrNum = np.array(scaleCorrNum)
    scaleAcc = scaleCorrNum/scaleTotalNum
    np.savez(os.path.join(output_folder, "res_test"),
             probs=probs_test, preds=probs_test.argmax(-1),
             xim=xim_batch, y=y_batch, A=A_test,
             all_y=np.vstack(all_y),
             all_preds=np.vstack(all_preds))

    logger.info(
        "*Epoch {} Acc Valid {}, Acc Train = {}, Acc Test = {}".format(
            epoch,
            conf_valid.accuracy(),
            conf_train.accuracy(),
            conf_test.accuracy())
    )
    log_scale_acc = '  '
    for j in range(5):
        log_scale_acc += ' %.1f: %.3f,' %((j+1)*0.5, scaleAcc[j])
    logger.info(log_scale_acc)

    if conf_valid.accuracy() > best_valid:
        best_valid = conf_valid.accuracy()
        look_count = LOOK_AHEAD
        all_param_values = [p.get_value() for p in all_params]
        np.save(os.path.join(output_folder, "best_model"), all_param_values)
        result_test = conf_test.accuracy()
    else:
        look_count -= 1

    if look_count <= 0:
        logger.info("TEST SCORE: {}".format(result_test))
        break
