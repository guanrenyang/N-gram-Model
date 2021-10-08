from os import dup, linesep, pardir, readlink
from typing import ItemsView
import arpa
from arpa.models.base import UNK, ARPAModel
import numpy
from arpa.api import ARPAModelSimple, dump, dumpf, dumps, load, loadf
from math import log10
from tqdm import tqdm

train_data_path = './Dataset/train_set.txt'
test_data_path = './Dataset/test_set.txt'
dev_data_path = './Dataset/dev_set.txt'
toy_data_path = './Dataset/toy_set.txt'
arpaModel_file_path = './Model/tri-gram.arpa'

uni_gram_count = {}
bi_gram_count = {}
tri_gram_count = {}

model = ARPAModelSimple()

def load_data(file_path: str):
    '''input data path. Return a list of tokens. Token is s single word. The input data shoud contain only one line and no `enter` in the end'''
    file = open(file_path, 'r')
    tokens = file.readline().split(' ')
    file.close()
    return tokens

def train(tokens):
    def add_in_dict(dict_name:dict, key, value):
        if key not in dict_name:
            dict_name[key] = []
        dict_name[key].append(value)

    def add_one_smoothing():
        '''感觉这里对add one理解有些问题，明天再看看'''

        # calculate 1-gram probability
        print('=====calculate 1-gram probability(add one smooth)=====')
        with tqdm(total=len(uni_gram_count)) as pbar:
            for key, value in uni_gram_count.items():
                uni_gram_probability[key]=(value+1)/(total_tokens)
                model.add_entry((key,), log10(uni_gram_probability[key]), bo=0)
                pbar.update(1)
        # calculate 2-gram probability
        print('=====calculate 2-gram probability(add one smooth)=====')
        with tqdm(total=len(bi_gram_count)) as pbar:
            for prev_word, word_list in bi_gram_count.items():
                given_prev_word_probability = {}
                for word in word_list:
                    given_prev_word_probability[word] = given_prev_word_probability.get(word, 0) + 1

                for key, value in given_prev_word_probability.items():
                    given_prev_word_probability[key] = (value+1)/(len(word_list)+total_tokens)
                    model.add_entry((prev_word, key), log10(given_prev_word_probability[key]), bo=0)

                bi_gram_probability[prev_word] = given_prev_word_probability
                pbar.update(1)
        
        # calculate 3-gram probability
        print('=====calculate 3-gram probability(add one smooth)=====')
        with tqdm(total=len(tri_gram_count)) as pbar:
            for prev_word, word_list in tri_gram_count.items():
                given_prev_word_probability = {}
                for word in word_list:
                    given_prev_word_probability[word] = given_prev_word_probability.get(word, 0) + 1
                for key, value in given_prev_word_probability.items():
                    given_prev_word_probability[key] = (value+1)/(len(word_list)+total_tokens)
                    model.add_entry(prev_word+(key,), log10(given_prev_word_probability[key]), bo=0)
                    
                bi_gram_probability[prev_word] = given_prev_word_probability
                pbar.update()
        model.add_entry(UNK, 0, bo=0)

        print('=====saving .arpa file in '+arpaModel_file_path+'=====')
        dumpf(model, arpaModel_file_path)
        print('=====finish saving=====')
        return 
    
    uni_gram_probability ={}
    bi_gram_probability ={}
    tri_gram_probability ={}    
    
    total_tokens = len(tokens)
    total_unigram = total_tokens + 2 # +2 includes <s> and </s>
    total_bigram = 0
    total_trigram = 0
    
    
    # count 1-gram, 2-gram, and 3-gram
    print('=====counting 1,2,3-grams=====')
    with tqdm(total=len(tokens)) as pbar:
        for current_token_index, current_token in enumerate(tokens):
            uni_gram_count[current_token] = uni_gram_count.get(current_token, 0) + 1
            if current_token_index==0:
                uni_gram_count['<s>'] = uni_gram_count.get('<s>', 0) + 1
                add_in_dict(bi_gram_count, '<s>', current_token)
                total_bigram += 1
            elif current_token_index==1: # the 2nd token
                prev_token = tokens[current_token_index-1]
                add_in_dict(bi_gram_count, prev_token, current_token)
                total_bigram += 1
                add_in_dict(tri_gram_count, ('<s>',prev_token), current_token)
                total_trigram += 1
            else:
                prev_token = tokens[current_token_index-1]
                prev_prev_token = tokens[current_token_index-2]
                
                add_in_dict(bi_gram_count, prev_token, current_token)
                total_bigram += 1
                
                add_in_dict(tri_gram_count, (prev_prev_token, prev_token), current_token)
                total_trigram += 1
                if current_token_index == total_tokens - 1:
                    uni_gram_count['</s>'] = uni_gram_count.get('</s>', 0) + 1
                    add_in_dict(bi_gram_count, current_token, '</s>')
                    total_bigram += 1
                    add_in_dict(tri_gram_count, (prev_token, current_token), '</s>')  
                    total_trigram += 1
            pbar.update(1)
        

    model.add_count(1, total_unigram) # add the total number of 1-gram
    model.add_count(2, total_bigram) # add the total number of 2-gram
    model.add_count(3, total_trigram) # add the total number of 3-gram
    
    add_one_smoothing()

    return uni_gram_probability, bi_gram_probability, tri_gram_probability
def test(tokens):
    ppl, _ = cal_ppl(model, tokens)
    print(f'Perplexity in testing data is: {ppl}')

def cal_ppl(model, sequence):

    length = len(sequence)
    logP = 0
    print('test sequence:',sequence)
    logP = logP + model.log_p(('<s>',sequence[0])) + model.log_p(('<s>',sequence[0],sequence[1]))

    for i,ch in enumerate(sequence[2:]):
        logP = logP + model.log_p((sequence[i],sequence[i+1],sequence[i+1]))

    logP = logP + model.log_p((sequence[-2],sequence[-1],'</s>'))

    logP= logP / -(length-1)
    ppl = 10**logP
    
    return ppl, logP   


if __name__ == '__main__':
    
    print('==========loading training data==========')
    train_tokens = load_data(train_data_path)

    print('==========loading testing data==========')
    test_tokens = load_data(toy_data_path)

    print('==========loading dev data==========')
    dev_tokens = load_data(toy_data_path)

    print('==========training==========')
    train(train_tokens)
    
    print('==========testing==========')
    test(test_tokens)

