import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd



def build_data_hold_out(pos_file, neg_file, pool_file, clean_string=True):
    """
    Loads data and split into two sets.
    """
    revs = []
    pos_file = pos_file
    neg_file = neg_file
    test_file = pool_file
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        nsents = 0
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": 'train'}
            revs.append(datum)
            nsents += 1
        print pos_file,"contained", nsents, "sentences"
    with open(neg_file, "rb") as f:
        nsents = 0
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": 'train'}
            revs.append(datum)
            nsents += 1

        print neg_file, "contained", nsents, "sentences"

    with open(test_file, "rb") as f:
        nsents = 0
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": 'test'}
            revs.append(datum)
            nsents += 1

        print test_file,"contained", nsents, "sentences"

    return revs, vocab



def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def process_data(w2v_file, pos_file, neg_file, pool_file, outfile=None, k=300):
    print "loading data..."
    print "\t Positive file:", pos_file
    print "\t Negative file:", neg_file
    print "\t Pool file:", pool_file
    revs, vocab = build_data_hold_out(pos_file, neg_file, pool_file, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "Data loaded!"
    print "Total number of sentences: " + str(len(revs))
    print "Vocab size: " + str(len(vocab))
    print "Max sentence length: " + str(max_l)
    print "Loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "Word2vec loaded!"
    print "Num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k)
    print "Getting word matrix (known words)"
    W, word_idx_map = get_W(w2v, k)
    rand_vecs = {}
    print "Getting word matrix (unknown words)"
    add_unknown_words(rand_vecs, vocab, k)
    W2, _ = get_W(rand_vecs, k)
    if outfile:
        cPickle.dump([revs, W, W2, word_idx_map, vocab], open(outfile, "wb"))
    else:
        return revs, W, W2, word_idx_map, vocab
if __name__=="__main__":
    w2v_file = sys.argv[1]
    data_root = '/media/HDD_2TB/DATASETS/cnn_polarity/'
    data_folder = [data_root + 'data/test.en', data_root + 'data/test_negativo.en', data_root  + 'data/training.en']
    outfile = 'mr.pkl'
    process_data(w2v_file, data_folder, outfile)