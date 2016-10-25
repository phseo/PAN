from collections import OrderedDict
from scipy.misc import imresize
import numpy as np
from libs.spatialelem import *
from libs.batch_norm  import *
import lasagne
from libs.spatialatt import *
import theano.tensor as T
import theano

from lasagne.layers import dnn
conv = dnn.Conv2DDNNLayer
def unzipp(shared_list):
	new_params = OrderedDict()
	for s in shared_list:
		new_params[s.name] = s.get_value()
	return new_params


def batch_imresize(batch, resize_factor):
	num_batch, num_channels, height, width = batch.shape
	out_height = int(height * resize_factor)
	out_width = int(width * resize_factor)

	batch = batch.transpose([0, 2, 3, 1])
	new_batch = np.zeros([num_batch, out_height, out_width, num_channels], dtype=batch.dtype)
	for i in range(num_batch):
		new_batch[i] = imresize(batch[i], (out_height, out_width))

	return new_batch.transpose([0, 3, 1, 2])


def satt2(img, ref, img_shape, ref_size=10, num_int_ch=32, l_name=None, dropout=None, W_ini=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.softmax, batch_norm=True, rtn_att_map=False, context_box=(1, 1)):
	if l_name:
		l_img_name = l_name+'[img]'
		l_q_name = l_name+'[q]'
		l_att_name = l_name+'[mix]'
	else:
		l_img_name = l_q_name = l_att_name = None

	#l_img = conv(img, num_filters=num_int_ch, filter_size=(1, 1), name=l_img_name, W=W_ini, nonlinearity=None)
	l_img = conv(img, num_filters=num_int_ch, filter_size=context_box, name=l_img_name, W=W_ini, nonlinearity=None, pad='same')

	l_q = lasagne.layers.EmbeddingLayer(ref, input_size=ref_size, output_size=num_int_ch, W=W_ini, name=l_q_name)
	l_mix = SpatialElemwiseMergeLayer(l_img, l_q, T.add)
	l_nonlin = lasagne.layers.NonlinearityLayer(l_mix)

	if batch_norm:
		l_bn = BatchNormLayer(l_nonlin)
	else:
		l_bn = l_nonlin
	if dropout:
		l_drp = lasagne.layers.DropoutLayer(l_bn)
	else:
		l_drp = l_nonlin
	l_weight = conv(l_drp, num_filters=1, filter_size=(1,1), name=l_att_name, W=W_ini, nonlinearity=None)
	l_weight_flat = lasagne.layers.FlattenLayer(l_weight)
	l_prob_flat = lasagne.layers.NonlinearityLayer(l_weight_flat, nonlinearity)
	l_prob = lasagne.layers.ReshapeLayer(l_prob_flat, (-1,)+img_shape)

	return l_prob

def satt(img, ref, img_shape, ref_size=10, num_int_ch=32, l_name=None, dropout=None, W_ini=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.softmax, batch_norm=True, rtn_att_map=False, context_box=(1, 1)):
	if l_name:
		l_img_name = l_name+'[img]'
		l_q_name = l_name+'[q]'
		l_att_name = l_name+'[mix]'
	else:
		l_img_name = l_q_name = l_att_name = None

	#l_img = conv(img, num_filters=num_int_ch, filter_size=(1, 1), name=l_img_name, W=W_ini, nonlinearity=None)
	l_img = conv(img, num_filters=num_int_ch, filter_size=context_box, name=l_img_name, W=W_ini, nonlinearity=None, pad='same')

	l_q = lasagne.layers.EmbeddingLayer(ref, input_size=ref_size, output_size=num_int_ch, W=W_ini, name=l_q_name)
	l_mix = SpatialElemwiseMergeLayer(l_img, l_q, T.add)
	l_nonlin = lasagne.layers.NonlinearityLayer(l_mix)

	if batch_norm:
		l_bn = BatchNormLayer(l_nonlin)
	else:
		l_bn = l_nonlin
	if dropout:
		l_drp = lasagne.layers.DropoutLayer(l_bn)
	else:
		l_drp = l_nonlin
	l_weight = conv(l_drp, num_filters=1, filter_size=(1,1), name=l_att_name, W=W_ini, nonlinearity=None)
	l_weight_flat = lasagne.layers.FlattenLayer(l_weight)
	l_prob_flat = lasagne.layers.NonlinearityLayer(l_weight_flat, nonlinearity)
	l_prob = lasagne.layers.ReshapeLayer(l_prob_flat, (-1,)+img_shape)
	l_att_out = SpatialAttentionLayer(img, l_prob)

	if rtn_att_map:	
		return l_att_out, l_prob
	else:
		return l_att_out

