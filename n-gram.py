from os import dup, linesep, pardir, readlink
from typing import ItemsView
from arpa.models.base import  ARPAModel
from arpa.api import ARPAModelSimple, dump, dumpf, dumps, load, loadf
from math import log10
from tqdm import tqdm

import arpa

# The path of train, test, and dev data. Relative or absolute paths are all acceptable with relative path by default.
train_data_path = './Dataset/train_set.txt'
test_data_path = './Dataset/test_set.txt'
dev_data_path = './Dataset/dev_set.txt'
toy_data_path = './Dataset/toy_set.txt'

# The `.arpa` format model is stored in `tri-gram.arpa`
arpaModel_file_path = './Model/tri-gram.arpa' 

# The path of result file
result_file_path = './Result/result.txt'
result_file = open(result_file_path, 'a')

UnKnown = '<unknown>'
# The following three py-dictionaries are used for counting 1, 2, and 3 grams.
# `uni_gram_count`: `key` is token and `value` is the number of times it appears.
# `bi_gram_count`: `key` is the leading one token of a 2-gram, `value` is a dictionary of the numbers of all subsequent tokens that appear after the leading token .
#                   e.g. {'this is': {'dog': 2, 'cat': 1}} -> 'this is dog' appears twice and this is cat appears twice
# `tri_gram_count`: very similar to `bi_gram_count`, the only difference is that `key` is a tuple with two elements.
# ***We repeatedly count repeated values***
uni_gram_count = {}
bi_gram_count = {}
tri_gram_count = {}

# The following three py-dictionaries are used for indexing the conditional probability (not log_10) of n-gram.
# For example:
#   uni_gram_probability['cat'] == 0.2 -> p('cat')=0.2
#   bi_gram_probability['cat'] == {'eat': 0.2, 'fish' =0.3} -> p('eat'|'cat') = 0.2, p('fish'|'cat')=0.3
#   tri_gram_probability[('cat', 'eat')] = {'fish': 0.8, 'pear': 0.2} -> p('fish'|'cat','eat')=0.8
uni_gram_probability ={}
bi_gram_probability ={}
tri_gram_probability ={} 

# `ARPAModelSimple` provides an convinent interface for operating `.arpa` files and n-gram models
model = ARPAModelSimple()

def load_data(file_path: str):
    '''
    Load data from file and returen a list of tokens.
    The input data must contain only only line.
    Args:
        file_path: str. Relative path and absolute path are all acceptable
    Return: 
        -> list. A list of tokens. e.g. ['this', 'is', 'a', 'dog']
    '''
    file = open(file_path, 'r')
    tokens = file.readline().split(' ')
    file.close()
    return tokens

def train(tokens):
    '''
    Train the n-gram model and store the model in an `arpa` file
    
    Args:
        tokens: list. A list of tokens.
    Return:
        No return value
    '''
    def add_in_dict(dict_name:dict, key, value):
        '''
        Add a n-gram to list `bi_gram_count` and `tri_gram_count`.
        '''
        if key not in dict_name:
            dict_name[key] = {}
        dict_name[key].get(value, 0) + 1

    def add_one_smoothing():
        '''
        Calculate probabilities of 1, 2, and 3 grams accroding to `uni_gram_count`, `bi_gram_count`, and `tri_gram_count`
        This is a inner function of function `train()`, so it has no passing arguments.
        
        ***`uni_gram_count`, `bi_gram_count`, `tri_gram_count`, `total_unigram`, `total_bigram`, `total_trigram`, 'tokens_length'  are the virtual passing parameters***
        '''
        
        # calculate 1-gram probability
        print('\n=====calculate 1-gram probability(add one smooth)=====')
        total_unigram_with_repeat = sum(uni_gram_count.values()) + 1 # +1 for UNK
        total_unigram = len(uni_gram_count)  
        
        with tqdm(total=len(uni_gram_count)) as pbar:

            for key, value in uni_gram_count.items():
                
                uni_gram_probability[key]=(value+1)/(total_unigram_with_repeat + total_unigram) 

                model.add_entry((key,), log10(uni_gram_probability[key]), bo=0) # the probability value in `arpa` file is log_10(probability)
                
                pbar.update(1)

        # calculate 2-gram probability
        print('\n=====calculate 2-gram probability(add one smooth)=====')
        with tqdm(total=len(bi_gram_count)) as pbar:
            for prev_word, word_dict in bi_gram_count.items():
                
                conditional_total_bigram_with_repeat = sum(word_dict)
                conditional_total_bigram = len(word_dict)
                
                bi_gram_probability[prev_word] = {}
                
                for key, value in word_dict.items():

                    bi_gram_probability[prev_word][key] = (value+1)/(conditional_total_bigram_with_repeat+conditional_total_bigram)
                    
                    model.add_entry((prev_word, key), log10(bi_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
        
        

        # calculate 3-gram probability
        print('\n=====calculate 3-gram probability(add one smooth)=====')
        with tqdm(total=len(tri_gram_count)) as pbar:

            for prev_word, word_dict in tri_gram_count.items():
                
                conditional_total_trigram_with_repeat = sum(word_dict)
                conditional_total_trigram = len(word_dict)
                
                tri_gram_probability[prev_word] = {}
                
                for key, value in word_dict.items():
                    
                    tri_gram_probability[prev_word][key] = (value+1)/(conditional_total_trigram_with_repeat+conditional_total_trigram)
                    
                    model.add_entry((prev_word, key), log10(tri_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)

        model.add_entry((UnKnown,), 0, bo=0) # assumse count(UNKNOWN) = 1

        print('\n=====saving .arpa file in '+arpaModel_file_path+'=====')
        dumpf(model, arpaModel_file_path)
        print('\n=====finish saving=====')
        return 
    
    # Store the number of 1-grams, 2-grams, and 3-grams in the training set.
    # They are properly set after counting
    total_unigram = 0
    total_bigram = {}
    total_trigram = {}
       
    tokens_length = len(tokens)

    
    # count 1-gram, 2-gram, and 3-gram
    print('=====counting 1,2,3-grams=====')
    with tqdm(total=len(tokens)) as pbar:
        for current_token_index, current_token in enumerate(tokens):
            # Add unigram in `uni_gram_count`
            
            # To be Deleted
            # total_unigram += (0 if current_token in uni_gram_count else 1)

            uni_gram_count[current_token] = uni_gram_count.get(current_token, 0) + 1

            if current_token_index==0:
                # Discard '<s>'
                # uni_gram_count['<s>'] = uni_gram_count.get('<s>', 0) + 1
                # add_in_dict(bi_gram_count, '<s>', current_token)
                # total_bigram += 1
                continue
            # Discard '<s>'
            elif current_token_index==1: # the 2nd token

                prev_token = tokens[current_token_index-1]
                
                # To be deleted
                # total_bigram += (0 if (prev_token in bi_gram_count and current_token in bi_gram_count[prev_token]) else 1)
                # '''
                # if we had never saw the leading token, we haven't seen the bigram. + 1
                # if we saw the leading token but haven't seen the corresponding token, we haven't seen the bigram. + 1
                # if we saw the leading token and the corresponding token, we saw the bigram. +0
                # '''

                add_in_dict(bi_gram_count, prev_token, current_token)
                
               
                # Discard '<s>'
                # add_in_dict(tri_gram_count, ('<s>',prev_token), current_token)
                # total_trigram += 1
            else:
                prev_token = tokens[current_token_index-1]
                prev_prev_token = tokens[current_token_index-2]
                
                # Te be deleted
                # total_bigram += (0 if (prev_token in bi_gram_count and current_token in bi_gram_count[prev_token]) else 1)

                add_in_dict(bi_gram_count, prev_token, current_token)
                
                # To be deleted
                # total_trigram += (0 if ((prev_prev_token, prev_token) in tri_gram_count and current_token in tri_gram_count[(prev_prev_token, prev_token)]) else 1)
                
                add_in_dict(tri_gram_count, (prev_prev_token, prev_token), current_token)
                
                # Discard '</s>'
                # if current_token_index == tokens_length - 1:
                #     uni_gram_count['</s>'] = uni_gram_count.get('</s>', 0) + 1
                #     add_in_dict(bi_gram_count, current_token, '</s>')
                #     total_bigram += 1
                #     add_in_dict(tri_gram_count, (prev_token, current_token), '</s>')  
                #     total_trigram += 1
            pbar.update(1)
        

    model.add_count(1, total_unigram) # add the total number of 1-gram
    model.add_count(2, total_bigram) # add the total number of 2-gram
    model.add_count(3, total_trigram) # add the total number of 3-gram
    
    add_one_smoothing()

    return uni_gram_probability, bi_gram_probability, tri_gram_probability
def test(tokens):
    ppl, _ = cal_ppl(model, tokens)
    print(f'Perplexity in testing data is: {ppl}')
    result_file.write(f'Perplexity in testing data is: {ppl}')

def cal_ppl(model, sequence):

    length = len(sequence)
    s = 0 # log probability

    with tqdm(total=len(sequence)) as pbar:
        for i, _ in enumerate(sequence):

            # handle unknown words
            if sequence[i] not in model.vocabulary():
                sequence[i] = UnKnown
                
            if i==0:
                s += model.log_p((sequence[i],))
            elif i==1:
                s += model.log_p((sequence[i-1],sequence[i]))
            else:
                s += model.log_p((sequence[i-2],sequence[i-1],sequence[i]))
            
            pbar.update(1)

    s= s / -(length-1)
    ppl = 10**s
    
    return ppl, s   


if __name__ == '__main__':
    
    print('\n==========loading training data==========')
    train_tokens = load_data(train_data_path)

    print('\n==========loading testing data==========')
    test_tokens = load_data(test_data_path)

    print('\n==========loading dev data==========')
    dev_tokens = load_data(dev_data_path)

    print('\n==========training==========')
    train(train_tokens)

    print('\n==========testing==========')
    result_file.write("\n Add-one smoothing:\n")
    test(test_tokens)

