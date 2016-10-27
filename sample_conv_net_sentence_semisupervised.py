"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import cPickle
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import warnings
import sys
import os
import time
from shutil import copyfile, rmtree
from process_data import process_data
from conv_net_classes import *


warnings.filterwarnings("ignore")


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory


def remove_dir(directory):
    if os.path.exists(directory):
        print 'removing directory %s'%directory
        rmtree(directory)

# different non-linearities


def ReLU(x):
    y = T.maximum(0.0, x)
    return y


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return y


def Tanh(x):
    y = T.tanh(x)
    return y


def Iden(x):
    y = x
    return y


def train_conv_net(datasets, U, img_w=300, filter_hs=[3, 4, 5], hidden_units=[100, 2], dropout_rate=[0.5],
                   shuffle_batch=True, n_epochs=25, batch_size=50, lr_decay = 0.95, conv_non_linear="relu",
                   activations=[Iden], sqr_norm_lim=9, non_static=True, test_batch=1000, savename="predictions",
                   savetofile=False):
    """
    Train a simple conv net
    :param img_h = sentence length (padded where necessary)
    :param img_w = word vector length (300 for word2vec)
    :param filter_hs = filter window sizes
    :param hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    :param sqr_norm_lim = s^2 in the paper
    :param lr_decay = adadelta decay parameter
    """

    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    #test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))


    o = 0
    test_predictions = []
    test_predictions_p = []
    len_test = datasets[1].shape[0]
    print "Classifying from a test made up of", len_test, "sentences"
    while o < len_test:
        print "Classifying sentences from", o, "to", min(o+test_batch, len_test)
        test_set_x = datasets[1][o:min(o+test_batch, len_test),:img_h]
        test_size = test_set_x.shape[0]
        test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
        test_pred_layers = []
        for conv_layer in conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = classifier.predict(test_layer1_input)
        test_y_pred_p = classifier.predict_p(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, y))
        test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)
        predict_val = theano.function([x], test_y_pred, allow_input_downcast=True)
        predict_p = theano.function([x], test_y_pred_p, allow_input_downcast=True)

        test_predictions.append(predict_val(test_set_x))
        test_predictions_p.append(predict_p(test_set_x))

        o +=test_batch
    if savetofile:
        cPickle.dump(test_predictions, open("%s.pkl"%savename, "wb"))
        cPickle.dump(test_predictions_p, open("%s.pkl"%(savename+'_p'), "wb"))
    return test_perf, test_predictions, test_predictions_p


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def get_idx_from_sent(sent, word_idx_map, max_l=70, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=70, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)        
        else:  
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     


def make_idx_data_holdout(revs, word_idx_map, max_l=70, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    added_train = 0
    added_test = 0
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        if rev["split"]=='test':
            test.append(sent)
            added_test +=1
        else:
            train.append(sent)
            sent.append(rev["y"])
            added_train +=1
    print "ADded train:", added_train
    print "Added test:", added_test

    print "ADding train stuff"
    train = np.array(train, dtype="int")
    print "ADding test stuff"

    test = np.array(test, dtype="int")
    return [train, test]


def write_in_txt(data, data_p, filename='./test_data.txt'):
     f = open(filename, 'wb')
     classes = cPickle.load(open(data))
     probabilities = cPickle.load(open(data_p))

     for v, vp in zip(classes, probabilities):
        for va, vap in zip(v, vp):
            f.write(str(va)+ ' ' + str(vap) + '\n')
     f.close()
     del classes
     del probabilities


def process_prediction_probs(prediction_probs, n_intances_to_add, pool_src, pool_trg):
    probs = numpy.array([], dtype="float32")
    for batch in prediction_probs:
        probs=numpy.append(probs,batch)
    probs = probs.reshape(-1, 2)
    top_positive_positions = probs.argsort(axis=0)[:, 0][:n_intances_to_add]
    top_negative_positions = probs.argsort(axis=0)[:, 1][:n_intances_to_add]
    positive_lines_src = []
    positive_lines_trg = []
    negative_lines = []
    neutral_lines_src = []
    neutral_lines_trg = []

    pool_file_src = open(pool_src)
    pool_file_trg = open(pool_trg)

    for i, (line_src, line_trg) in enumerate(zip (pool_file_src, pool_file_trg)):
        if i in top_negative_positions:
            negative_lines.append(line_src)
        elif i in top_positive_positions:
            positive_lines_src.append(line_src)
            positive_lines_trg.append(line_trg)
        else:
            neutral_lines_src.append(line_src)
            neutral_lines_trg.append(line_trg)

    pool_file_src.close()
    pool_file_trg.close()

    return positive_lines_src, positive_lines_trg, negative_lines, neutral_lines_src, neutral_lines_trg


def semisupervised_selection(data_dir, dest_dir, initial_pos_filename, initial_neg_filename, initial_pool_filename, w2v_file,
                             word_vectors="-rand", src_lan='en', trg_lan='de', non_static=True, n_iter=10,
                             test_batch=7000, instances_to_add=50000, debug=False):

    """
    Performs a semisupervised text selection over a pool of sentences based on initial positive/negative files.
    The steps that takes are:
        1. Classify the pool according the positive/negative samples through a CNN
        2. Take the most positive and most negative sentences from the pool and includes it into the positive/negative training samples
        3. With this extended positive/negative sets, trains another CNN and backs to 1.

    :param data_dir: Directoty where the data files are
    :param initial_pos_filename: Initial "in-domain" corpus
    :param initial_neg_filename: Initial "out-of-domain" corpus
    :param initial_pool_filename: Pool of sentences where to perform the selection
    :param w2v_file: Word2vec file (for the CNN input)
    :param word_vectors: To use word vectors from word2vec or random word vector
    :param non_static: Non-static CNNs
    :param n_iter: Number of iterations carried out by the proccess
    :param test_batch: Classify the pool with this batch
    :param instances_to_add: Number of instances to add at each iteration
    :return:
    """
    pos_filename_src = data_dir + '/' + initial_pos_filename + '.' + src_lan
    in_domain_file = open(pos_filename_src, 'r')
    in_domain = in_domain_file.readlines()
    in_domain_file.close()
    pos_filename_trg = data_dir + '/' + initial_pos_filename + '.' + trg_lan

    neg_filename_src = data_dir + '/' + initial_neg_filename + '.' + src_lan

    pool_filename_src = data_dir + '/' + initial_pool_filename + '.' + src_lan
    pool_filename_trg = data_dir + '/' + initial_pool_filename + '.' + trg_lan

    for i in range(n_iter):

        print "------------------ Starting iteration", i, "------------------"
        new_pos_filename_src = dest_dir + '/' + initial_pos_filename + '_' + str(i) + '.' + src_lan
        new_pos_filename_trg = dest_dir + '/' + initial_pos_filename + '_' + str(i) + '.' + trg_lan

        new_pos_filename_src_tmp = dest_dir + '/' + initial_pos_filename + 'tmp' + '.' + src_lan
        if debug:
            new_neg_filename_src_tmp = dest_dir + '/' + initial_neg_filename + 'tmp' + '.' + src_lan

        new_neg_filename_src = dest_dir + '/' + initial_neg_filename + '_' +  str(i) + '.' + src_lan

        new_pool_filename_src = dest_dir + '/' + initial_pool_filename + '_' + str(i) + '.' + src_lan
        new_pool_filename_trg = dest_dir + '/' + initial_pool_filename + '_' + str(i) + '.' + trg_lan

        if i > 0:
            copyfile(pos_filename_src, new_pos_filename_src_tmp)
            copyfile(pos_filename_src, new_pos_filename_src)
            copyfile(pos_filename_trg, new_pos_filename_trg)

        with open(new_pos_filename_src_tmp, "a") as f:
            for line in in_domain:
                f.write(line)

        copyfile(neg_filename_src, new_neg_filename_src)
        copyfile(pool_filename_src, new_pool_filename_src)
        copyfile(pool_filename_trg, new_pool_filename_trg)

        x = process_data(w2v_file, new_pos_filename_src_tmp, new_neg_filename_src, new_pool_filename_src)
        revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

        if word_vectors=="-rand":
            print "using: random vectors"
            U = W2
        elif word_vectors=="-word2vec":
            print "using: word2vec vectors"
            U = W
        else:
            raise NotImplementedError, "Choose between -rand or -word2vec options"

        results = []
        datasets = make_idx_data_holdout(revs, word_idx_map, max_l=70,k=300, filter_h=5)
        perf, predictions, prediction_probs = train_conv_net(datasets, U, lr_decay=0.95, filter_hs=[3,4,5],
                                                              conv_non_linear="relu", hidden_units=[200,100,2],
                                                              shuffle_batch=True, n_epochs=14, sqr_norm_lim=9,
                                                              non_static=non_static, batch_size=128, dropout_rate=[0.5],
                                                              test_batch=test_batch, savename="predictions_" + str(i),
                                                             savetofile=False)
        positive_lines_src, positive_lines_trg, negative_lines, neutral_lines_src, neutral_lines_trg = \
            process_prediction_probs(prediction_probs, instances_to_add, pool_filename_src, pool_filename_trg)

        print "Adding", len(positive_lines_src), "positive lines"
        print "Positive sample:", positive_lines_src[0], "---", positive_lines_trg[0]
        print "Adding", len(negative_lines), "negative lines"
        print "Negative sample:", negative_lines[0]

        print "Adding", len(neutral_lines_src), "neutral lines"
        print "Neutral sample:", neutral_lines_src[0], "---", neutral_lines_trg[0]

        new_pos_file_src = open(new_pos_filename_src, 'a')
        new_pos_file_trg = open(new_pos_filename_trg, 'a')

        new_neg_file = open(new_neg_filename_src, 'a')
        if debug:
            new_neg_file_tmp = open(new_neg_filename_src_tmp, 'a')

        new_pool_file_src = open(new_pool_filename_src, 'w')
        new_pool_file_trg = open(new_pool_filename_trg, 'w')

        for line in positive_lines_src:
            new_pos_file_src.write(line)
        for line in positive_lines_trg:
            new_pos_file_trg.write(line)

        for line in negative_lines:
            new_neg_file.write(line)
            if debug:
                new_neg_file_tmp.write(line)
        for line in neutral_lines_src:
            new_pool_file_src.write(line)
        for line in neutral_lines_trg:
            new_pool_file_trg.write(line)

        new_pos_file_src.close()
        new_pos_file_trg.close()

        new_neg_file.close()

        new_pool_file_src.close()
        new_pool_file_trg.close()

        if debug:
            new_neg_file_tmp.close()

        pos_filename_src = new_pos_filename_src
        pos_filename_trg = new_pos_filename_trg

        neg_filename_src = new_neg_filename_src

        pool_filename_src = new_pool_filename_src
        pool_filename_trg = new_pool_filename_trg

        print "perf: " + str(perf)
        results.append(perf)
        print str(np.mean(results))
        #write_in_txt("predictions_"+str(i), "predictions_"+str(i), 'train_tags_probs' + str(i) + '.txt')


if __name__ == "__main__":

    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")

    data_root = '/media/HDD_2TB/DATASETS/cnn_polarity/'
    data_dir = data_root + 'DATA/Emea-en-fr'
    initial_pos_filename = 'test_positivo'
    initial_neg_filename = 'test_negativo'
    initial_pool_filename= 'training_test'
    src_lan = 'en'
    trg_lan = 'fr'
    dest_dir = data_root + 'selection/Emea-en-fr_test_' + src_lan + trg_lan

    reload = False
    if not reload:
        remove_dir(dest_dir)
        create_dir_if_not_exists(dest_dir)

    w2v_file = data_root + 'DATA/GoogleNews-vectors-negative300.bin'
    semisupervised_selection(data_dir, dest_dir, initial_pos_filename, initial_neg_filename, initial_pool_filename,
                             w2v_file, word_vectors=word_vectors, non_static=non_static, n_iter=10,
                             src_lan=src_lan, trg_lan=trg_lan, test_batch=5000, instances_to_add=1500, debug=True)


