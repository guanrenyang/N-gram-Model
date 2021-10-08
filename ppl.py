import enum
import arpa


def cal_ppl(model, seq1):

    s = 0
    s = s + model.log_p('<s> '+seq1[0]) + model.log_p('<s>'+' '+seq1[0]+' '+seq1[1])

    print('<s>', model.log_p('<s>'))
    print('<s> '+seq1[0], model.log_p('<s> '+seq1[0]))
    print('<s> '+seq1[0]+' '+seq1[1], model.log_p('<s> '+seq1[0]+' '+seq1[1]))

    for i,ch in enumerate(seq1[2:]):
        current_trigram = seq1[i]+' '+seq1[i+1]+' '+seq1[i+2]
        s = s + model.log_p(current_trigram)

        print(current_trigram, model.log_p(current_trigram))

    s = s + model.log_p(seq1[-2]+' '+seq1[-1]+' '+'</s>')

    print(seq1[-2]+' '+seq1[-1]+' '+'</s>', model.log_p(seq1[-2]+' '+seq1[-1]+' '+'</s>'))

    s= s / (-13)
    ppl = 10**s
    print(s,ppl)
    
    return ppl, s

lm=arpa.loadf('./Model/tri-gram.arpa')
model = lm[0]

seq1 = ('<s>, ')
seq2 = "019033910051"
seq3 = "120033910006"
seq4 = "120033910013"

ppl_1, s_1 = cal_ppl(model, seq1)
ppl_2, s_2 = cal_ppl(model, seq2)
ppl_3, s_3 = cal_ppl(model, seq3)
ppl_4, s_4 = cal_ppl(model, seq4)

print('sequence 1: PPL ',ppl_1, 'log_p', s_1)
print('sequence 2: PPL ',ppl_2, 'log_p', s_2)
print('sequence 3: PPL ',ppl_3, 'log_p', s_3)
print('sequence 4: PPL ',ppl_4, 'log_p', s_4)
