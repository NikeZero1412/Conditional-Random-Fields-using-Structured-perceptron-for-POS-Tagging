# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:18:06 2016

@author: HOME
"""

from __future__ import division

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k].
    Behavior undefined if ties (python docs might give clues)"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def goodness_score(seq, A_factor, B_factors):
    # the total "goodness" score of the proposed sequence
    N = len(B_factors)
    score = 0
    score += sum(A_factor[seq[t],seq[t+1]] for t in range(N-1))
    score += sum(B_factors[t][seq[t]] for t in range(N))
    return score

def exhaustive(A_factor, B_factors, output_vocab):
    # the exhaustive decoding algorithm.
    N = len(B_factors)  # length of entire sentence

    def allpaths(sofar):
        # Recursively generate all sequences given a prefix "sofar".
        # this probably could be redone cleverly as a python generator
        retpaths = []
        if len(sofar)==N:
            return [sofar]
        for sym in output_vocab:
            newpath = sofar[:] + [sym]
            retpaths += allpaths(newpath)
        return retpaths

    path_scores = {}
    for path in allpaths([]):
        path = tuple(path)  # tuple version can be used as dict key
        score = goodness_score(path, A_factor, B_factors)
        path_scores[path] = score
    bestseq = dict_argmax(path_scores)
    return bestseq

def viterbi(A_factor, B_factors, output_vocab):
    """
    A_factor: a dict of key:value pairs of the form
        {(curtag,nexttag): score}
    with keys for all K^2 possible neighboring combinations,
    and scores are numbers.  We assume they should be used ADDITIVELY, i.e. in log space.
    higher scores mean MORE PREFERRED by the model.

    B_factors: a list where each entry is a dict {tag:score}, so like
    [ {Noun:-1.2, Adj:-3.4}, {Noun:-0.2, Adj:-7.1}, .... ]
    each entry in the list corresponds to each position in the input.

    output_vocab: a set of strings, which is the vocabulary of possible output
    symbols.

    RETURNS:
    the tag sequence yvec with the highest goodness score
    """

    N = len(B_factors)   # length of input sentence
    
    # viterbi log-prob tables
    V = [{tag:None for tag in output_vocab} for t in range(N)]
    # backpointer tables
    # back[0] could be left empty. it will never be used.
    back = [{tag:None for tag in output_vocab} for t in range(N)]
    # Storing back pointers
    backprop=[{tag:{tag:None for tag in output_vocab} for tag in output_vocab} for t in range(N)]            
    
    for t in range(0,N):
        for tag in output_vocab:
            viterbi_list=[]
            for prev_tag in output_vocab:
                # Handling t=0 case
                if t==0:
                    viterbi_list.append(B_factors[t][tag])
                    V[t][tag]= max(viterbi_list)
                    #back[t][k]=0 ## Initializing to zero, even though it isn't used.
                    
                else:
                    viterbi_list.append(V[t-1][prev_tag]+ A_factor[(prev_tag,tag)] + B_factors[t][tag])
                    backprop[t][tag][prev_tag]=(V[t-1][prev_tag]+ A_factor[(prev_tag,tag)])
                    V[t][tag]= max(viterbi_list)
                   
                    back[t][tag]=dict_argmax(backprop[t][tag])
    #print "Viterbi List is : \n \n",V 
    #print "\nBack pointer list is : \n \n" ,back 
                    
               
        

    # todo implement backtrace also
    output_seq=[None for t in range(N)]
    output_seq[-1]=dict_argmax(V[-1])
    for t in range(N-2,-1,-1):
        tag =output_seq[t+1]
        state=back[t+1][tag]
        output_seq[t]=state
    #print "\nOutput sequence is : \n",tuple(output_seq)
    return output_seq ## Changed the output for the exhaustive coding also to return tuple instead of list.

def randomized_test(N=3, V=5):
    # This creates a random model and checks if the exhaustive and viterbi
    # decoders agree.6n 
    import random
    A = { (a,b): random.random() for a in range(V) for b in range(V) }
    Bs = [ [random.random() for k in range(V)] for i in range(N)]
             
             
    print "output_vocab=", range(V)
    print "A=",A
    print "Bs=",Bs


    fromex  = exhaustive(A,Bs, range(V))
    print fromex
    fromvit = viterbi(A,Bs,range(V))
    print fromvit
    assert fromex==fromvit
    print "Worked!"

if __name__=='__main__':
    A = {(0,0):3, (0,1):0, (1,0):0, (1,1):3}
    Bs= [ [0,1], [0,1], [30,0] ]
    # that's equivalent to: [ {0:0,1:1}, {0:0,1:1}, {0:30,1:0} ]

    yex = exhaustive(A, Bs, set([0,1]))
    print "Exhaustive decoding:", yex
    print "score:", goodness_score(yex, A, Bs)
    yvi = viterbi(A, Bs, set([0,1]))
    print "Viterbi decoding:", yvi
    print "score:", goodness_score(yvi, A, Bs)
    randomized_test(N=3, V=5)