#! /usr/bin/env python
caffe_root = '/Users/yanan/caffe/' # SET YOUR CAFFE_ROOT HERE 
import os
import sys
import csv
import glob
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from pylab import *
import tempfile
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

def conv(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='xavier', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='xavier', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    for i, p in enumerate(top_k):
        print  100*probs[p], labels[p]
        return labels[p]

def disp_vehicle_preds(net, image):
    label = disp_preds(net, image, ['0', '1'], 1, name='vehicle')
    return label

def build_net(data, label=None, train=True, num_classes=2,
             classifier_name='fc8', learn_all=False):
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1 = conv(n.data, 5, 4, stride=1, param=param)
    n.pool1 = max_pool(n.conv1, 2, stride=2)
    n.conv2 = conv(n.pool1, 5, 2, stride=1, param=param)
    n.pool2 = max_pool(n.conv2, 2, stride=2)
    ip1, relu1 = fc_relu(n.pool2, nout=10, param=learned_param)
    ip2 = L.InnerProduct(ip1, num_output=2, param=learned_param)
   
    if not train:
        n.probs = L.Softmax(ip2)

    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(ip2, n.label)
        n.acc = L.Accuracy(ip2, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

def vehicle_net(train=True, learn_all=False, subset=None, batch_size=1050):
    if subset is None:
        subset = 'train' if train else 'test'
    source = '%s.txt' % subset
    vehicle_data, vehicle_label = L.ImageData(
        source=source,
        batch_size=batch_size,  ntop=2, is_color=False, shuffle=False)
    if train: 
        learn_all = learn_all
        label = vehicle_label
    else:
        learn_all = False
        label = None
    return build_net(data=vehicle_data, label=label, train=train,
                    num_classes=2,
                    classifier_name='vehicle',
                    learn_all=learn_all)

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 100  # Test after every 100 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1 
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 10 iterations.
    s.display = 10

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training 
    # -- as long as we have that much data to train.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_vehicle/finetune_vehicle'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=10):
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

def eval_vehicle_net(weights):
    file_names = []
    with open('test.txt', 'r') as sourcefile:
        for line in sourcefile:
            file_names.append(line.split(' ')[0])  
    batch_size = len(file_names)
    test_net = caffe.Net(vehicle_net(train=False, batch_size=batch_size), weights, caffe.TEST)
    test_net.forward()
    data_batch = test_net.blobs['data'].data.copy()
    classifications= []
    for i in range(0, batch_size):
        image = data_batch[i]
        label = disp_vehicle_preds(test_net, image)
        classifications.append([file_names[i].split('/')[-1], label])
    with open('classifications.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(classifications)

def train_vehicle_net():

    niter = 200  # number of iterations to train

    vehicle_solver_filename = solver(vehicle_net(train=True))
    vehicle_solver = caffe.get_solver(vehicle_solver_filename)

    print 'Running solvers for %d iterations...' % niter
    solvers = [('scratch', vehicle_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    train_loss = loss['scratch']
    train_acc = acc['scratch']
    vehicle_weights = weights['scratch']

    print 'train loss:', train_loss
    print 'train acc:', train_acc
    print 'vehicle weights:', vehicle_weights


if __name__ == "__main__":

    #Here's the model that I trained
    weights = 'weights.vehicle.caffemodel'

    #Set your test dataset dir path here
    test_imgs = glob.glob("images/test/*.jpg")
    #Set your train dataset dir path here (If you want to retrain this network)
    train_imgs = glob.glob("images/train/*.jpg")
   
    #Preparing data for training, 
    #however because I've trained the model, 
    #there's no need to create this file for now.
    with open("train.txt", 'w') as outfile:
        for f in train_imgs:
            if 'pos-' in f:
                outfile.write(f + " " + "1\n" )
            elif 'neg-' in f:
                outfile.write(f + " " + "0\n")

    #Preparing data for testing,
    #every line is constructed with filename + sapce + dummylabel
    with open("test.txt", 'w') as outfile:
        for f in test_imgs:
            outfile.write(f + " " + "0\n" )

    #Run testing data through the net
    #Then write predicted results into csv file
    eval_vehicle_net(weights)
