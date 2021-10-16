from arpa.models.base import  ARPAModel
from arpa.api import ARPAModelSimple, dump, dumpf, dumps, load, loadf
from math import  log10
from tqdm import tqdm
import argparse
import numpy as np


# The path of train, test, and dev data. Relative or absolute paths are all acceptable with relative path by default.
train_data_path = './Dataset/train_set.txt'
test_data_path = './Dataset/test_set.txt'
dev_data_path = './Dataset/dev_set.txt'

# The path of result file
result_file_path = ''

arpa_file_path=''

UNKNOWN = '<unknown>'
# The following three py-dictionaries are used for counting 1, 2, and 3 grams.
# `uni_gram_count`: `key` is token and `value` is the number of times it appears.
# `bi_gram_count`: `key` is the leading one token of a 2-gram, `value` is a dictionary of the numbers of all subsequent tokens that appear after the leading token .
#                   e.g. {'this is': {'dog': 2, 'cat': 1}} -> 'this is dog' appears twice and this is cat appears twice
# `tri_gram_count`: very similar to `bi_gram_count`, the only difference is that `key` is a tuple with two elements.
# ***We repeatedly count repeated values***
uni_gram_count = {}
bi_gram_count = {}
tri_gram_count = {}

delta = 0.00001 # log(0) -> log(0+delta)
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

def train(tokens, smoothing_method = 'addictive', vocab_threshold=0):
    '''
    Train the n-gram model and store the model in an `arpa` file
    
    Args:
        tokens: list. A list of tokens.
        arpa_file_path: str. `train` function will store the `arpa` format in `arpa_file_path`. The path must end with `.arpa`
        smoothing_method: str. Specify the smoothing method. It could be `addictive` or `absolute_discounting` or
                                                                         `good_turing` or `linear_discounting` or `katz_back_off`
        vocab_threshold: int. Filter out words with frequency < `vocab_threshold` (extrict less than)
    Return:
        No return value, but sorte
    '''
    def add_in_dict(dict_name:dict, key, value):
        '''
        Add a n-gram to list `bi_gram_count` and `tri_gram_count`.
        '''
        if key not in dict_name:
            dict_name[key] = {}
        dict_name[key][value] = dict_name[key].get(value, 0) + 1
   
    def absolute_discounting_smoothing(uni_b=0.1,bi_b=0.1,tri_b=0.1):
        V = len(uni_gram_count)

        print('\n=====calculate 1-gram probability(absolute discounting smooth)=====')
        total_unigram_with_repeat = sum(uni_gram_count.values())
        with tqdm(total=len(uni_gram_count)) as pbar:
            for key, value in uni_gram_count.items():
                    
                uni_gram_probability[key]=value/total_unigram_with_repeat
                if uni_gram_probability[key] != 0:
                    model.add_entry((key,), log10(uni_gram_probability[key]), bo=0)                
                pbar.update(1)
        # calculate 2-gram probability
        print('\n=====calculate 2-gram probability(absolute discounting smooth)=====')
        with tqdm(total=len(bi_gram_count)) as pbar:
            bi_k=0
            bi_n1=0
            bi_n2=0
            bi_R = V**2
            bi_N = 0
            for prev_word, word_dict in bi_gram_count.items():
                bi_k=bi_k+len(word_dict)
                bi_N=bi_N+sum(word_dict.values())
                #compute n_1,n_2
                for key in bi_gram_count[prev_word]:
                    if bi_gram_count[prev_word][key] == 1:
                        bi_n1=bi_n1+1
                    elif bi_gram_count[prev_word][key] == 2:
                        bi_n2=bi_n2+1
            bi_n0 = V**2-bi_k 
            bi_b = {}           

            # Take the upper bound of b as the value of b
            if bi_n1 != 0 and bi_n2 != 0:
                bi_b=bi_n1/((bi_n1+2*bi_n2))
            for prev_word, word_dict in bi_gram_count.items():
                conditional_total_bigram = len(word_dict)
                conditional_total_bigram_with_repeat = sum(word_dict.values())             
                bi_gram_probability[prev_word] = {}            
                max_2 = max(word_dict.values())
                p_2 = np.zeros(max_2+1) # n_2[r] denotes the number of binary groups that start with prev_word and occur r times
                p_2Absolute = np.zeros(max_2+1) # Normalized probability values

                if bi_n0 != 0:
                    # Calculate p(r) for r > 0
                    for r in range(1,max_2+1):
                        p_2[r]=(r-bi_b)/bi_N
                    p_2[0]=bi_b*(bi_R-bi_n0)/(bi_N*bi_n0)
                    

                    # Normalization
                    sum_2=0
                    for i in range(0,max_2+1):
                        sum_2+=p_2[i]
                    for i in range(0,max_2+1):
                        p_2Absolute[i] = p_2[i]/sum_2

                else:
                    for i in range(0,max_2+1):
                        p_2Absolute[i] = i/conditional_total_bigram_with_repeat
                # Update the probability values in the probability dictionary
                for key, value in word_dict.items():
                    if p_2Absolute[bi_gram_count[prev_word][key]]!=0:
                        bi_gram_probability[prev_word][key] = p_2Absolute[bi_gram_count[prev_word][key]]
                    
                        model.add_entry((prev_word,key), log10(bi_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
                if p_2Absolute[0] !=0:
                    model.add_entry((prev_word,UNKNOWN), log10(p_2Absolute[0]), bo=0)

        # calculate 3-gram probability
        print('\n=====calculate 3-gram probability(absolute discounting smooth)=====')
        with tqdm(total=len(tri_gram_count)) as pbar:
            tri_k=0
            tri_n1=0
            tri_n2=0
            tri_R=V**3
            tri_N=0
            for prev_word, word_dict in tri_gram_count.items():
                tri_k=tri_k+len(word_dict)
                tri_N=tri_N+sum(word_dict.values())
                # Statistics n_1,n_2
                for key in tri_gram_count[prev_word]:
                    if tri_gram_count[prev_word][key] == 1:
                        tri_n1=tri_n1+1

                    elif tri_gram_count[prev_word][key] == 2:
                        tri_n2=tri_n2+1
            tri_n0 = V**3-tri_k
            tri_b = {}

            # Take the upper bound of b as the value of b
            if tri_n1 != 0 and tri_n2 != 0:
                tri_b=tri_n1/(tri_n1+2*tri_n2)
            for prev_word, word_dict in tri_gram_count.items():
                conditional_total_trigram_with_repeat = sum(word_dict.values())
                #conditional_total_trigram = len(word_dict)
                    
                tri_gram_probability[prev_word] = {}

                max_3 = max(word_dict.values())
                p_3 = np.zeros(max_3+1) # n_3[r] indicates the number of triples starting with prev_word and occurring r times
                p_3Absolute = np.zeros(max_3+1) # Normalized probability values

                if tri_n0 != 0:
                    # Calculate p(r) for r > 0
                    for r in range(1,max_3+1):
                        p_3[r]=(r-tri_b)/tri_N
                    p_3[0]=tri_b*(tri_R-tri_n0)/(tri_N*tri_n0)
                    # End of p(r) calculation

                    # Normalization
                    sum_3=0
                    for i in range(0,max_3+1):
                        sum_3+=p_3[i]
                    for i in range(0,max_3+1):
                        p_3Absolute[i] = p_3[i]/sum_3
                else:
                    for i in range(0,max_3+1):
                        p_3Absolute[i] = i/conditional_total_trigram_with_repeat
                # Update the probability values in the probability dictionary
                for key, value in word_dict.items():
                    if p_3Absolute[tri_gram_count[prev_word][key]]!=0:
                        tri_gram_probability[prev_word][key] = p_3Absolute[tri_gram_count[prev_word][key]]
                    
                        model.add_entry(prev_word+(key,), log10(tri_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
                if p_3Absolute[0]!=0:
                    model.add_entry(prev_word+(UNKNOWN,), log10(p_3Absolute[0]), bo=0)

        model.add_entry((UNKNOWN,), 0, bo=0) # assumse count(<unknown>,...) = 1
        print('\n=====saving .arpa file in '+arpa_file_path+'=====')
        dumpf(model, arpa_file_path)
        print('\n=====finish saving=====')
        return 
    def addictive_smoothing():
        '''
        Calculate probabilities of 1, 2, and 3 grams accroding to `uni_gram_count`, `bi_gram_count`, and `tri_gram_count`
        This is a inner function of function `train()`, so it has no passing arguments.
        
        ***`uni_gram_count`, `bi_gram_count`, `tri_gram_count`, `total_unigram`, `total_bigram`, `total_trigram`, 'tokens_length'  are the virtual passing parameters***
        '''
        vocab_size = len(uni_gram_count.keys())

        # calculate 1-gram probability
        print('\n=====calculate 1-gram probability(addictive smooth)=====')
        total_unigram_with_repeat = sum(uni_gram_count.values())
        
        with tqdm(total=len(uni_gram_count)) as pbar:
            for key, value in uni_gram_count.items():
                uni_gram_probability[key]=value / total_unigram_with_repeat
                pbar.update(1)

        mean_uni_prob = np.mean(list(uni_gram_probability.values()))

        # calculate 2-gram probability
        print('\n=====calculate 2-gram probability(addictive smooth)=====')
        uni_bo = {}
        # calculate 2-gram probability for r>0
        with tqdm(total=len(bi_gram_count)) as pbar:
            for prev_word, word_dict in bi_gram_count.items():
                
                conditional_total_bigram_with_repeat = sum(word_dict.values())

                bi_gram_probability[prev_word] = {}
                
                for key, value in word_dict.items():
                    bi_gram_probability[prev_word][key] = (value+0.5)/(conditional_total_bigram_with_repeat + mean_uni_prob * vocab_size)

                uni_bo[prev_word] = -log10(conditional_total_bigram_with_repeat + mean_uni_prob * vocab_size)
                pbar.update(1)


        # calculate 3-gram probability
        print('\n=====calculate 3-gram probability(addictive smooth)=====')
        bi_bo = {}
        with tqdm(total=len(tri_gram_count)) as pbar:
            for prev_word, word_dict in tri_gram_count.items():
                
                conditional_total_trigram_with_repeat = sum(word_dict.values())
                
                tri_gram_probability[prev_word] = {}
                
                for key, value in word_dict.items():
                    tri_gram_probability[prev_word][key] = (value+mean_uni_prob)/(conditional_total_trigram_with_repeat+ mean_uni_prob * vocab_size)
                bi_bo[prev_word] = -log10(conditional_total_trigram_with_repeat + mean_uni_prob*vocab_size)
                pbar.update(1)

        for key, value in uni_gram_probability.items():
            model.add_entry((key,), log10(value), bo = uni_bo.get(key, 0))
        for prev_word, word_dict in bi_gram_probability.items():
            for word, value in word_dict.items():
                model.add_entry((prev_word, word), log10(value), bi_bo.get((prev_word, word), 0))
        for prev_word, word_dict in tri_gram_probability.items():
            for word, value in word_dict.items():
                model.add_entry(prev_word+(word,), log10(value))
        print('\n=====saving .arpa file in '+arpa_file_path+'=====')
        dumpf(model, arpa_file_path)
        print('\n=====finish saving=====')
        return 
    def good_turing_smoothing(uni_limit=5,bi_limit=5,tri_limit=5):
        # calculate 1-gram probability
        print('\n=====calculate 1-gram probability(good turing smooth)=====')
        total_unigram_with_repeat = sum(uni_gram_count.values()) 
        
        with tqdm(total=len(uni_gram_count)) as pbar:
            # The part of calculating r* starts here

            # The maximum number of statistics is limit
            max_1 = uni_limit 
            n_1 = np.zeros(max_1+1) # n_1[r] denotes the number of monomials that occur r times
            r_1star = np.zeros(max_1+1) # r_1star[r] denotes the r adjusted r*
            p_1 = np.zeros(max_1+1) # Store the probability value after calculating r*
            p_1Good_turing = np.zeros(max_1+1) # Normalized probability values

            for key in uni_gram_count:
                if uni_gram_count[key] <= max_1: # Here the judgment is added
                    n_1[uni_gram_count[key]]=n_1[uni_gram_count[key]]+1 # Count the number of occurrences of r-tuples
            for r in range(1,max_1):
                if n_1[r] != 0 and n_1[r+1] != 0: 
                    r_1star[r]=(r+1)*n_1[r+1]/n_1[r] 
                else:
                    r_1star[r]=r
            r_1star[max_1]=max_1 # r*[max] and p[0] are handled separately

            p_1[0] = n_1[1]/total_unigram_with_repeat
            for r in range(1,max_1+1):
                p_1[r]=r_1star[r]/total_unigram_with_repeat # Calculate the probability after r*
            # Normalization
            sum_1=0
            for i in range(0,max_1+1):
                sum_1+=p_1[i]
            for i in range(0,max_1+1):
                p_1Good_turing[i] = p_1[i]/sum_1
            for key in uni_gram_count:
                if uni_gram_count[key] <= max_1:
                    uni_gram_probability[key] = p_1Good_turing[uni_gram_count[key]]
                else:
                    uni_gram_probability[key] = uni_gram_count[key]/total_unigram_with_repeat
                model.add_entry((key,), log10(uni_gram_probability[key]), bo=0) # the probability value in `arpa` file is log_10(probability)
                
                pbar.update(1)

            uni_gram_probability[UNKNOWN] = p_1Good_turing[0]
            model.add_entry((UNKNOWN,), uni_gram_probability[UNKNOWN], bo=0)
            
            
        # calculate 2-gram probability
        print('\n=====calculate 2-gram probability(good turing smooth)=====')
        with tqdm(total=len(bi_gram_count)) as pbar:
            for prev_word, word_dict in bi_gram_count.items():                
                conditional_total_bigram_with_repeat = sum(word_dict.values())
                bi_gram_probability[prev_word] = {}

            #计算r*从这里开始

            #这里改变了max_2的值
                max_2 = bi_limit
                n_2 = np.zeros(max_2+1) #n_2[r]表示以prev_word开头，出现r次的二元组的个数
                r_2star = np.zeros(max_2+1) #r_2star[r]表示r调整后的r*
                p_2 = np.zeros(max_2+1) #存放计算r*后的概率值
                p_2Good_turing = np.zeros(max_2+1) #归一化后的概率值
                for key, value in word_dict.items():

                    #这里增加判断
                    if word_dict[key] <= max_2:
                        n_2[word_dict[key]]=n_2[word_dict[key]]+1 #统计出现r次元组的个数
                for r in range(1,max_2):
                    if n_2[r] != 0 and n_2[r+1] != 0:
                        r_2star[r]=(r+1)*n_2[r+1]/n_2[r] #套公式
                    else:
                        r_2star[r] = r
                r_2star[max_2]=max_2 # r*[max] and p[0] are handled separately

                p_2[0] = (n_2[1]) if n_2[1]!=0 else 1 /conditional_total_bigram_with_repeat
                for r in range(1,max_2+1):
                    p_2[r]=r_2star[r]/conditional_total_bigram_with_repeat 
                # Normalization
                sum_2=0
                for i in range(0,max_2+1):
                    sum_2+=p_2[i]
                for i in range(0,max_2+1):
                    p_2Good_turing[i] = p_2[i]/sum_2

                for key, value in word_dict.items():
                    if word_dict[key] <= max_2:
                        bi_gram_probability[prev_word][key] = p_2Good_turing[bi_gram_count[prev_word][key]]
                    else:
                        bi_gram_probability[prev_word][key] = bi_gram_count[prev_word][key]/conditional_total_bigram_with_repeat   
                    model.add_entry((prev_word, key), log10(bi_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
                model.add_entry((prev_word, UNKNOWN), log10(p_2Good_turing[0]+delta), bo=0)

            model.add_entry((UNKNOWN,), 0, bo=0) # assumse count(<unknown>,...) = 1
            print('\n=====saving .arpa file in '+arpa_file_path+'=====')
            dumpf(model, arpa_file_path)
            print('\n=====finish saving=====')
            return 
    def Linear_discounting_smoothing(uni_b=0.1,bi_b=0.1,tri_b=0.1):
        V = len(uni_gram_count)
        print('\n=====calculate 1-gram probability(Linear discounting smooth)=====')
        total_unigram_with_repeat = sum(uni_gram_count.values())
        with tqdm(total=len(uni_gram_count)) as pbar:
            for key, value in uni_gram_count.items():
                    
                uni_gram_probability[key]=value/total_unigram_with_repeat
                if uni_gram_probability[key] != 0:
                    model.add_entry((key,), log10(uni_gram_probability[key]), bo=0)                
                pbar.update(1)
        # calculate 2-gram probability
        print('\n=====calculate 2-gram probability(Linear discounting smooth)=====')
        with tqdm(total=len(bi_gram_count)) as pbar:
            bi_k=0
            bi_n1=0
            #bi_n2=0
            #bi_R = V**2
            bi_N = 0
            for prev_word, word_dict in bi_gram_count.items():
                bi_k=bi_k+len(word_dict)
                bi_N=bi_N+sum(word_dict.values())
                # count n_1,n_2
                for key in bi_gram_count[prev_word]:
                    if bi_gram_count[prev_word][key] == 1:
                        bi_n1=bi_n1+1
                    #elif bi_gram_count[prev_word][key] == 2:
                        #bi_n2=bi_n2+1
            bi_n0 = V**2-bi_k 
            bi_alpha = bi_n1/bi_N
            for prev_word, word_dict in bi_gram_count.items():
                conditional_total_bigram = len(word_dict)
                conditional_total_bigram_with_repeat = sum(word_dict.values())             
                bi_gram_probability[prev_word] = {}            
                max_2 = max(word_dict.values())
                p_2 = np.zeros(max_2+1) # n_2[r] denotes the number of binary groups that start with prev_word and occur r times
                p_2Linear = np.zeros(max_2+1) # Normalized probability values

                if bi_n0 != 0:
                    # Calculate p(r) for r > 0
                    for r in range(1,max_2+1):
                        p_2[r]=r*(1-bi_alpha)/bi_N
                    p_2[0]=bi_alpha/bi_n0

                    # Normalization
                    sum_2=0
                    for i in range(0,max_2+1):
                        sum_2+=p_2[i]
                    for i in range(0,max_2+1):
                        p_2Linear[i] = p_2[i]/sum_2

                else:
                    for i in range(0,max_2+1):
                        p_2Linear[i] = i/conditional_total_bigram_with_repeat
                # Update the probability values in the probability dictionary
                for key, value in word_dict.items():
                    if p_2Linear[bi_gram_count[prev_word][key]]!=0:
                        bi_gram_probability[prev_word][key] = p_2Linear[bi_gram_count[prev_word][key]]
                    
                        model.add_entry((prev_word,key), log10(bi_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
                if p_2Linear[0] !=0:
                    model.add_entry((prev_word, UNKNOWN), log10(p_2Linear[0]), bo=0)

        # calculate 3-gram probability
        print('\n=====calculate 3-gram probability(Linear discounting smooth)=====')
        with tqdm(total=len(tri_gram_count)) as pbar:
            tri_k=0
            tri_n1=0

            tri_N=0
            for prev_word, word_dict in tri_gram_count.items():
                tri_k=tri_k+len(word_dict)
                tri_N=tri_N+sum(word_dict.values())
                # count n_1,n_2
                for key in tri_gram_count[prev_word]:
                    if tri_gram_count[prev_word][key] == 1:
                        tri_n1=tri_n1+1

            tri_n0 = V**3-tri_k
            tri_alpha=tri_n1/tri_N
            for prev_word, word_dict in tri_gram_count.items():
                conditional_total_trigram_with_repeat = sum(word_dict.values())
                #conditional_total_trigram = len(word_dict)
                    
                tri_gram_probability[prev_word] = {}
                
                max_3 = max(word_dict.values())
                p_3 = np.zeros(max_3+1) 
                p_3Linear = np.zeros(max_3+1) # Normalized probability values

                if tri_n0 != 0:
                    # Calculate p(r) for r > 0
                    for r in range(1,max_3+1): 
                        p_3[r]=r*(1-tri_alpha)/tri_N
                    p_3[0]=tri_alpha/tri_n0

                    # Normalization
                    sum_3=0
                    for i in range(0,max_3+1):
                        sum_3+=p_3[i]
                    for i in range(0,max_3+1):
                        p_3Linear[i] = p_3[i]/sum_3
                else:
                    for i in range(0,max_3+1):
                        p_3Linear[i] = i/conditional_total_trigram_with_repeat
                # Update the probability values in the probability dictionary
                for key, value in word_dict.items():
                    if p_3Linear[tri_gram_count[prev_word][key]]!=0:
                        tri_gram_probability[prev_word][key] = p_3Linear[tri_gram_count[prev_word][key]]
                    
                        model.add_entry(prev_word+(key,), log10(tri_gram_probability[prev_word][key]), bo=0)

                pbar.update(1)
                if p_3Linear[0]!=0:
                    model.add_entry(prev_word+(UNKNOWN,), log10(p_3Linear[0]), bo=0)

        model.add_entry((UNKNOWN,), 0, bo=0) # assumse count(<unknown>,...) = 1
        print('\n=====saving .arpa file in '+arpa_file_path+'=====')
        dumpf(model, arpa_file_path)
        print('\n=====finish saving=====')
        return     
    def katz_back_off():
        print('\n=====calculate 1-gram probability(katz smooth)=====')

        # compute N_r
        total_unigram_with_repeat = sum(uni_gram_count.values())
        max_uni = max(uni_gram_count.values())
        n_uni = [0 for i in range(max_uni+1)] # N[r] mean the num of grams appearing r times
        list_uni_count_values = list(uni_gram_count.values())
        for i in list_uni_count_values:
            n_uni[i] += 1


        # compute d_r
        d_uni = [0 for _ in range(max_uni+1)]
        for r in range(1, max_uni):
            d_uni[r] = ((r+1)*n_uni[r+1]/(r*n_uni[r]))  if n_uni[r]!=0 and n_uni[r+1]!=0  else 1
        d_uni[-1] = 1 # The maximum value is handled specially because there is no n_r+1

        #compute 1-gram probability for r>0
        for key, r in uni_gram_count.items():
            uni_gram_probability[key] = r * d_uni[r] / total_unigram_with_repeat
        
        assert n_uni[0]==0 
                
        print('\n=====calculate 2-gram probability(katz smooth)=====')
        
        # compute p_katz for r>0
        for prev_word, word_dict in tqdm(bi_gram_count.items()):
            
            bi_gram_probability[prev_word] = {}
            total_bigram_with_repeat = sum(word_dict.values())
            max_bi = max(word_dict.values())
            
            # compute N_r for n>0
            n_bi = [0 for _ in range(max_bi+1)]
            for i in list(word_dict.values()):
                n_bi[i] += 1

            # compute N_r finish

            # compute d_r for r>0
            d_bi = [0 for _ in range(max_bi+1)]
            for r in range(1, max_bi):
                d_bi[r] = ((r+1)*n_bi[r+1] / (r*n_bi[r])) if n_bi[r]!=0 and n_bi[r+1]!=0 else 1 
            # compute d_r done
            d_bi[-1] = 1
            
            # compute p_katz for r>0
            for key, r in word_dict.items():
                bi_gram_probability[prev_word][key] = r * d_bi[r] / total_bigram_with_repeat

            
        # compute alpha(back-off weight) for tokens
        sum_p_ML = {} # The sum of the 1-gram probabilities of the second element of the binary group that has occurred
        sum_p_katz = {} # The sum of the katz probabilities of the occurrence of the binary group
        alpha_uni ={} # alpha value of a tuple
        print('=====compute 1-gram backoff(alpha)=====')
        for word in uni_gram_count.keys():
            sum_p_ML[word] = 0
            sum_p_katz[word] = 0

        with tqdm(total=len(uni_gram_count)**2) as pbar:
            for prev_word, word_dict in bi_gram_count.items():
                for current_word in word_dict.keys():
                    sum_p_katz[prev_word] += bi_gram_probability[prev_word][current_word] 
                    sum_p_ML[prev_word] += uni_gram_probability[current_word]
                    pbar.update(1)

        for word in tqdm(uni_gram_count.keys()):
            alpha_uni[word] = (1-sum_p_katz.get(word, 0))/(1 - sum_p_ML.get(word, 0)) if sum_p_ML.get(word, 0)<1 else delta
        
        print('\n=====calculate 3-gram probability(katz smooth)=====')
        
        # compute p_katz for r>0
        for prev_word, word_dict in tqdm(tri_gram_count.items()):
            
            tri_gram_probability[prev_word] = {}
            total_tri_gram_with_repeat = sum(word_dict.values())
            max_tri = max(word_dict.values())
            
            # compute N_r for n>0
            n_tri = [0 for _ in range(max_tri+1)]
            for i in list(word_dict.values()):
                n_tri[i] += 1
            # compute N_r finish

            # compute d_r for r>0
            d_tri = [0 for _ in range(max_tri+1)]
            for r in range(1, max_tri):
                d_tri[r] = ((r+1)*n_tri[r+1] / (r*n_tri[r])) if n_tri[r]!=0 and n_tri[r+1]!=0 else 1
            d_tri[-1] = 1
            # compute d_r done

            # compute p_katz for r>0
            for key, r in word_dict.items():
                tri_gram_probability[prev_word][key] = r * d_tri[r] / total_tri_gram_with_repeat

            
        print("compute alpha(back-off weight) for 3-gram not exist and 2-gram exist")
        sum_p_ML = {} # The sum of the 2-gram probabilities of the third element of the triplet that has appeared
        sum_p_katz = {} # The sum of the katz probabilities of the occurring triples
        alpha_bi ={} # The alpha value of the binary
        
        for prev_word, word_dict in bi_gram_count.items():
            for word in word_dict.keys():
                sum_p_ML[(prev_word, word)] = 0
                sum_p_katz[(prev_word, word)] = 0

        for leading_2gram, word_dict in tqdm(tri_gram_count.items()):
                for current_word in word_dict.keys():
                        sum_p_katz[leading_2gram] += tri_gram_probability[leading_2gram][current_word] 
                        sum_p_ML[leading_2gram] += bi_gram_probability[leading_2gram[1]][current_word]

        for prev_word, word_dict in tqdm(bi_gram_count.items()):
            for word, value in word_dict.items():
                leading_2gram = (prev_word, word)
                alpha_bi[leading_2gram] = (1-sum_p_katz.get(leading_2gram, 0))/(1 - sum_p_ML.get(leading_2gram, 0)) if sum_p_ML.get(leading_2gram, 0)!=1 else delta
                    
        # add probabilty in model:
        for key, value in uni_gram_probability.items():
            model.add_entry((key,), log10(value+delta), bo=log10((alpha_uni.get(key, 1)) if alpha_uni.get(key, 1)>0 else delta))
        for prev_word, word_dict in bi_gram_probability.items():
            for key, value in word_dict.items():
                model.add_entry((prev_word, key), log10(value+delta), bo=log10((alpha_bi.get((prev_word, key), 1)) if alpha_bi.get((prev_word, key), 1)>0 else delta))
        for prev_word, word_dict in tri_gram_probability.items():
            for key, value in word_dict.items():
                model.add_entry(prev_word+(key,), log10(value))
                
        print('\n=====saving .arpa file in '+arpa_file_path+'=====')
        dumpf(model, arpa_file_path)
        print('\n=====finish saving=====')
        return 
    # Store the number of 1-grams, 2-grams, and 3-grams in the training set.
    # They are properly set after counting

    total_unigram = 0
    total_bigram = 0
    total_trigram = 0
       
    tokens_length = len(tokens)

    # count 1-gram, 2-gram, and 3-gram
    print('=====counting 1-grams=====')
    with tqdm(total = len(tokens)) as pbar:
        for current_token_index, current_token in enumerate(tokens):
            # Add unigram in `uni_gram_count`
            uni_gram_count[current_token] = uni_gram_count.get(current_token, 0) + 1
            pbar.update(1)
    
    print('\n=====filter out low-frequency words=====')
    filtered = []
    uni_gram_count [UNKNOWN] = 0
    for key, value in uni_gram_count.items():
        if value < vocab_threshold:
            uni_gram_count[UNKNOWN] += + value
            filtered.append(key)
    if uni_gram_count[UNKNOWN]==0: # In case of 
        uni_gram_count[UNKNOWN] = 1 
    for key in filtered:
        uni_gram_count.pop(key)
    total_unigram = len(uni_gram_count)

    print('=====counting 2,3-grams=====')

    with tqdm(total=len(tokens)) as pbar:
        for current_token_index, current_token in enumerate(tokens):
            if current_token not in uni_gram_count.keys(): # replace tokens that are not in the vocabulary with UNKNOWN
                current_token = UNKNOWN

            if current_token_index==0 :
                if '<s>' in uni_gram_count.keys():
                    add_in_dict(bi_gram_count, '<s>', current_token)
                else:
                    add_in_dict(bi_gram_count, UNKNOWN, current_token)
                total_bigram += 1
            elif current_token_index==1: # the 2nd token

                prev_token = tokens[current_token_index-1] if tokens[current_token_index-1] in uni_gram_count.keys() else UNKNOWN
                add_in_dict(bi_gram_count, prev_token, current_token)
                total_bigram+=1

                # to discard '<s>', uncomment the lines below
                if '<s>' in uni_gram_count.keys():
                    add_in_dict(tri_gram_count, ('<s>',prev_token), current_token)
                else:
                    add_in_dict(tri_gram_count, (UNKNOWN,prev_token), current_token)
                total_trigram += 1
            else:
                prev_token = tokens[current_token_index-1] if tokens[current_token_index-1] in uni_gram_count.keys() else UNKNOWN
                prev_prev_token = tokens[current_token_index-2] if tokens[current_token_index-2] in uni_gram_count.keys() else UNKNOWN

                add_in_dict(bi_gram_count, prev_token, current_token)
                total_bigram += 1

                add_in_dict(tri_gram_count, (prev_prev_token, prev_token), current_token)
                total_trigram += 1

                # to discard '</s>', comment the following lines
                if current_token_index == tokens_length - 1:
                    
                    if '</s>' in uni_gram_count.keys():
                        add_in_dict(bi_gram_count, current_token, '</s>')
                        total_bigram += 1

                        add_in_dict(tri_gram_count, (prev_token, current_token), '</s>')  
                        total_trigram += 1
                    else:
                        add_in_dict(bi_gram_count, current_token, UNKNOWN)
                        total_bigram += 1

                        add_in_dict(tri_gram_count, (prev_token, current_token), UNKNOWN)  
                        total_trigram += 1

            pbar.update(1)
                    
    
    model.add_count(1, total_unigram) # add the total number of 1-gram
    model.add_count(2, total_bigram) # add the total number of 2-gram
    model.add_count(3, total_trigram) # add the total number of 3-gram

    if smoothing_method=='addictive':
        addictive_smoothing()
    elif smoothing_method=='absolute_discounting':
        absolute_discounting_smoothing()
    elif smoothing_method=='good_turing':
        good_turing_smoothing()
    elif smoothing_method=='katz_back_off':
        katz_back_off()
    elif smoothing_method=='linear_discounting':
        Linear_discounting_smoothing()
    else:
        print('Error: please input legal smoothing method')

def test(tokens, result_file):
    def cal_ppl(model, sequence):

        length = len(sequence)
        s = 0 # log probability

        with tqdm(total=len(sequence)) as pbar:
            for i, _ in enumerate(sequence):

                # handle unknown words
                if sequence[i] not in model.vocabulary():
                    sequence[i] = UNKNOWN
                    
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
    ppl, _ = cal_ppl(model, tokens)
    print(f'Perplexity in testing data is: {ppl}')
    result_file.write(f'Perplexity in testing data is: {ppl} \n')

# smoothing_method: str. Specify the smoothing method. It could be `addictive` or `absolute_discounting` or
#                                                                          `good_turing` or `linear_discounting` or `katz_back_off`

if __name__ == '__main__':
    
    addictive_arpa_file_path = './Model/tri-gram-addictive.arpa'
    absolute_discounting_arpa_file_path = './Model/tri-gram-absolute-discounting.arpa'
    good_turing_arpa_file_path = './Model/tri-gram-good-turing.arpa'
    linear_discounting_arpa_file_path = './Model/tri-gram-linear-discounting.arpa'
    katz_back_off_arpa_file_path = './Model/tri-gram-katz.arpa'

    addictive_result_file_path = './Result/addictive-result.txt'
    absolute_discounting_result_file_path = './Result/absolute-dicounting-result.txt'
    good_turing_result_file_path = './Result/good-turing-result.txt'
    linear_discounting_result_file_path = './Result/linear-discounting-result.txt'
    katz_back_off_result_file_path = './Result/katz-back-off-result.txt'

    # init arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--Smoothing', default='katz_back_off')
    args = parser.parse_args()

    # init `arpa_file_path`
    if args.Smoothing == 'addictive':
        arpa_file_path += addictive_arpa_file_path
        result_file_path += addictive_result_file_path
    elif args.Smoothing == 'absolute_discounting':
        arpa_file_path += absolute_discounting_arpa_file_path
        result_file_path += absolute_discounting_result_file_path
    elif args.Smoothing == 'good_turing':
        arpa_file_path += good_turing_arpa_file_path
        result_file_path += good_turing_result_file_path
    elif args.Smoothing == 'linear_discounting':
        arpa_file_path += linear_discounting_arpa_file_path
        result_file_path += linear_discounting_result_file_path
    elif args.Smoothing == 'katz_back_off':
        arpa_file_path += katz_back_off_arpa_file_path
        result_file_path += katz_back_off_result_file_path
    else:
        print('ERROR: Input illegal smoothing method')

    print('\n==========loading training data==========')
    train_tokens = load_data(train_data_path)

    print('\n==========loading testing data==========')
    test_tokens = load_data(test_data_path)

    print('\n==========loading dev data==========')
    dev_tokens = load_data(dev_data_path)

    print('\n==========training, vocab_threshold--1==========')
    train(train_tokens, smoothing_method=args.Smoothing, vocab_threshold=1)

    print('\n==========testing==========')
    result_file = open(result_file_path, 'a')
    test(dev_tokens, result_file)