'''
EECS 595 Homework 1 Problem 2: Collocation Identification

name: Xiechen Wang
uniqname: xiechenw
'''

import numpy as np
import sys

filename = sys.argv[1]
algorithm = sys.argv[2]

if algorithm != 'chi-square' and algorithm != 'PMI':
    sys.exit()

fp = open(filename, 'r')
text_data = fp.read()
fp.close()

words = []
text = ''

for i in range(len(text_data)):

    if text_data[i] == ' ':
        if text[0].islower() or text[0].isupper() or text[0].isdigit():
	    words.append(text)
	elif text[0] == '\n' and len(text) > 1:
	    if text[1].islower() or text[1].isupper() or text[1].isdigit():
                words.append(text[1:len(text)])            
        text = ''

    else:
        text += text_data[i]

raw_bigrams = []
N = len(words)

for i in range(N-1):

    raw_bigrams.append(words[i] + ' ' + words[i+1])

bigrams = []
avoid_duplicate_bigrams = []

for i in range(N-1):

    if raw_bigrams[i] not in avoid_duplicate_bigrams and raw_bigrams.count(raw_bigrams[i]) >= 5:

        bigram = []
        bigram.append(words[i])
        bigram.append(words[i+1])
        bigram.append(raw_bigrams.count(raw_bigrams[i]))
        bigram.append(words.count(words[i]))
        bigram.append(words.count(words[i+1]))

        avoid_duplicate_bigrams.append(raw_bigrams[i])
        bigrams.append(bigram)

scores = []
B = len(bigrams)
score = 0

for i in range(B):

    if algorithm == 'chi-square':

        O11 = bigrams[i][2]
        O12 = bigrams[i][3] - bigrams[i][2]
        O21 = bigrams[i][4] - bigrams[i][2]
        O22 = N + bigrams[i][2] - bigrams[i][3] - bigrams[i][4]
        E11 = (O11+O12)*(O11+O21)/N
        E12 = (O12+O11)*(O12+O22)/N
        E21 = (O11+O21)*(O22+O21)/N
        E22 = (O12+O22)*(O21+O22)/N

        score = np.square(O11-E11)/E11+np.square(O12-E12)/E12+np.square(O21-E21)/E21+np.square(O22-E22)/E22

    else:

        score = np.log(bigrams[i][2]*N*N/bigrams[i][3]/bigrams[i][4]/(N-1))

    bigrams[i].append(score)

score_sorted = sorted(bigrams, key=lambda x:x[5])

for i in range(20):
    print(score_sorted[B-i-1][0] + ' ' + score_sorted[B-i-1][1] + ' ' + str(score_sorted[B-i-1][5]))
