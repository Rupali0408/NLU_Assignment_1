import nltk
import string
import random
import math
import numpy as np
from nltk.help import brown_tagset
from numpy import linalg as LA
import matplotlib
# from nltk.book import *
from nltk.tokenize import *
from sklearn.model_selection import train_test_split
from collections import defaultdict

from nltk.corpus import gutenberg
from nltk.corpus import brown

brown_text = []
gutenberg_text = []
Vocabulary = []


def text_cleaning():
    brown_data = []
    brown_data = brown.sents()
    gutenberg_data = gutenberg.sents()
    punctuations = [',', '.', ':', ';', '?', '"', '!', '--', '(', ')','``']
    punctuations.append("''")
    for sentence in brown_data:
        sentence.insert(0, '<s>')
        sentence.insert(0, '<s>')
        sentence.append('</s>')
        for i in sentence:
            if i in punctuations:
                sentence.remove(i)
        brown_text.append(sentence)

    for sentence in gutenberg_data:
        sentence.insert(0, '<s>')
        sentence.insert(0, '<s>')
        sentence.append('</s>')
        for i in sentence:
            if i in punctuations:
                sentence.remove(i)
        gutenberg_text.append(sentence)
        # print(gutenberg_text[3])


text_cleaning()

def generate_Ngram(text,n):
    gram_n =  {}
    if n == 1:
        for sentence in text:
            for i in range(len(sentence)):
                key = tuple(sentence[i:i + n])
                if key in gram_n:
                    gram_n[key] += 1
                else:
                    gram_n[key] = 1
    else:
        for sentence in text:
            for i in range(len(sentence) - (n - 1)):
                outer_key = tuple(sentence[i:i + (n-1)])
                inner_key = (sentence[i+n-1],)
                #print(outer_key)
                if outer_key in gram_n:
                    if inner_key in gram_n[outer_key]:
                        gram_n[outer_key][inner_key] += 1
                    else:
                        gram_n[outer_key][inner_key] = 1
                else:
                    gram_n[outer_key] = {}
                    gram_n[outer_key][inner_key] = 1

    #gram_n = gram_n.items()
    #gram_n = sorted(gram_n,key = lambda x: x[1],reverse=True)
    #print((gram_n))
    return gram_n
#text_cleaning()
#B_train,B_test = train_test_split(brown_text,test_size=0.1)
#unigram = generate_Ngram(B_train,1)
#print(unigram)

def compute_probabilities(train):
    train_word_count = 0
    unigram_prob_sum = 0.0
    unigram = generate_Ngram(train, 1)
    bigram = generate_Ngram(train, 2)
    trigram = generate_Ngram(train, 3)
    for sentence in train:
        train_word_count += len(sentence)
    for key in unigram:
        unigram[key] = (unigram[key], (float(unigram[key] - 0.75) / train_word_count))
        value = unigram[key]
        unigram_prob_sum += value[1]
    for key in unigram:
        Vocabulary.append(key)

    for outer_key in bigram:
        for inner_key in bigram[outer_key]:
            count_denominator = unigram[outer_key]
            bigram[outer_key][inner_key] = (
            bigram[outer_key][inner_key], float(bigram[outer_key][inner_key] - 0.75) / count_denominator[0])

    for outer_key in trigram:
        for inner_key in trigram[outer_key]:
            count_denominator = bigram[(outer_key[0],)][(outer_key[1],)]
            trigram[outer_key][inner_key] = (
            trigram[outer_key][inner_key], float(trigram[outer_key][inner_key] - 0.75) / count_denominator[0])

    return unigram, bigram, trigram, unigram_prob_sum


def stupid_backoff_probability(w, unigram, bigram, trigram, u_sum):
    p = 0.0
    t_sum = 0.0
    b_sum = 0.0
    alpha = 0.0
    beta = 0.0
    gamma = 0.0
    t_outer_key = (w[0], w[1])
    t_inner_key = (w[2],)
    b_outer_key = (w[1],)
    b_inner_key = (w[2],)
    u_key = (w[2],)

    alpha = (1 - t_sum) / (1 - b_sum)
    beta = (1 - b_sum) / (1 - u_sum)
    gamma = float(1 - u_sum)
    # print(t_sum,b_sum,gamma)
    alpha = 0.4
    beta = 0.4
    if t_outer_key in trigram:
        if t_inner_key in trigram[t_outer_key]:
            value = trigram[t_outer_key][t_inner_key]
            pt = value[1]
            p = pt
    elif b_outer_key in bigram:
        if b_inner_key in bigram[b_outer_key]:
            value = bigram[b_outer_key][b_inner_key]
            pb = value[1]
            p = alpha * pb

    elif u_key in unigram:
        value = unigram[u_key]
        pu = value[1]
        p = alpha * beta * pu
    else:
        p = alpha * beta * gamma
    # p = l1*pt + l2*pb + l3*pu
    return (p)


def perplexity(test_data, unigram, bigram, trigram, unigram_prob_sum):
    sum_p = 0
    count = 0
    test_word_count = 0
    for sentence in test_data:
        test_word_count += len(sentence)
    for sentence in test_data:
        count += 1
        p = 1
        for i in range(len(sentence) - 2):
            w = (sentence[i], sentence[i + 1], sentence[i + 2])
            p = p * stupid_backoff_probability(w, unigram, bigram, trigram, unigram_prob_sum)
        if p == 0.0:
            continue
        #print(count)
        Pw = (math.log(p, 2))

        sum_p += Pw

    H = -1 * (sum_p) / test_word_count

    perplexity = pow(2, H)
    return perplexity


def generate_words_list(prev, current, trigram):
    trigram_words = []
    p = []
    words = []
    outer_key = (prev, current)
    p.clear()
    trigram_words.clear()
    if outer_key in trigram:
        trigram_words = []
        p = []
        for inner_key in trigram[outer_key]:
            next = inner_key[0]
            trigram_words.append(next)
            value = trigram[outer_key][inner_key]
            p.append(value[1])
        p.append(1 - sum(p))
        trigram_words.append('UNK')
    # print(trigram_words)

    p_words = sorted(zip(p, trigram_words), reverse=True)[:50]

    for i in range(len(p_words)):
        words.append(p_words[i][1])
    # print(words)
    return trigram_words, p


def sentence_generator(unigram, bigram, trigram):
    sentence = []
    # bigram_words = []
    word = "<s>"
    sentence.append(word)
    sentence.append(word)
    prev = "<s>"
    current = "<s>"
    while (current != "</s>"):
        trigram_words, prob = generate_words_list(prev, current, trigram)
        index = np.array(range(0, len(trigram_words)))
        prob = np.array(prob)
        next_w = np.random.choice(index, p=prob)
        while trigram_words[next_w] == 'UNK':
            next_w = np.random.choice(index, p=prob)
        print(trigram_words[next_w])
        sentence.append(trigram_words[next_w])
        prev = current
        current = trigram_words[next_w]
    return sentence

B_train,B_test = train_test_split(brown_text,test_size=0.1)
unigram,bigram,trigram,unigram_prob_sum = compute_probabilities(B_train)
#sentence = sentence_generator(unigram,bigram,trigram)
#print(sentence)
#print(len(B_test))
print("D1: Brown Corpus ,D2: Gutenberg Corpus")
perp = perplexity(B_test,unigram,bigram,trigram,unigram_prob_sum)
print("Perplexity for  S1: Train: D1-Train, Test: D1-Test")
print(perp)
print()



G_train,G_test = train_test_split(gutenberg_text,test_size=0.1)
gunigram,gbigram,gtrigram,gunigram_prob_sum  = compute_probabilities(G_train)
#sentence = sentence_generator(gunigram,gbigram,gtrigram)
#print(sentence)
perp = perplexity(G_test,gunigram,gbigram,gtrigram,gunigram_prob_sum)
print("Perplexity for  S2: Train: D2-Train, Test: D2-Test")

print(perp)



bgunigram,bgbigram,bgtrigram,bgunigram_prob_sum  = compute_probabilities((G_train+B_train))
#sentence = sentence_generator(bgunigram,bgbigram,bgtrigram)
#print(sentence)
perp = perplexity(B_test,bgunigram,bgbigram,bgtrigram,bgunigram_prob_sum)
print("Perplexity for S3: Train: D1-Train + D2-Train, Test: D1-Test")
print(perp)

bgunigram,bgbigram,bgtrigram,bgunigram_prob_sum  = compute_probabilities((G_train+B_train))
#sentence = sentence_generator(bgunigram,bgbigram,bgtrigram)
#print(sentence)

perp = perplexity(G_test,bgunigram,bgbigram,bgtrigram,bgunigram_prob_sum)
print("Perplexity for S4: Train: D1-Train + D2-Train, Test: D2-Test")
print(perp)