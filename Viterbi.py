'''
EECS 595 Homework 3: Viterbi Part-of-speech Tagger

name: Xiechen Wang
uniqname: xiechenw
'''

import numpy as np
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]

f = open(train_path, 'r', encoding = 'utf-8')

count_tagphi = {}
count_phi = 0

count_wordtag = {}
count_tagtag = {}

count_tag = {}
count_word = {}
tag_idx = {}

i = 0
j = 0

for line in f.readlines():

    line = line.split()
    flag = True
    count_phi += 1
    for wordtag in line:

        if '/' not in wordtag:
            continue

        word = wordtag[:wordtag.find('/')]
        tag = wordtag[wordtag.find('/') + 1:]
        while '/' in tag:
            word = word + '/' + tag[:tag.find('/')]
            tag = tag[tag.find('/') + 1:]

        if '|' in tag:
            tag = tag[:tag.find('|')]

        if word not in count_word:
            count_word[word] = 1
        else:
            count_word[word] += 1

        if tag not in tag_idx:
            tag_idx[tag] = j
            j += 1
            count_tag[tag] = 1
        else:
            count_tag[tag] += 1

        if flag:
            if tag not in count_tagphi:
                count_tagphi[tag] = 1
            else:
                count_tagphi[tag] += 1
            flag = False
        elif tag not in count_tagtag:
            count_tagtag[tag] = {}
            count_tagtag[tag][tag_prev] = 1
        elif tag_prev not in count_tagtag[tag]:
            count_tagtag[tag][tag_prev] = 1
        else:
            count_tagtag[tag][tag_prev] += 1

        if tag not in count_wordtag:
            count_wordtag[tag] = {}
            count_wordtag[tag][word] = 1
        elif word not in count_wordtag[tag]:
            count_wordtag[tag][word] = 1
        else:
            count_wordtag[tag][word] += 1

        tag_prev = tag

f.close()

len_tag = len(tag_idx)
W_s = len(count_word)
tags = list(count_tag.keys())
prob_tagtag = np.zeros((len_tag, len_tag))

i = 0
for tag in tag_idx:

    if tag in count_tagphi:
        count_tagphi[tag] = count_tagphi[tag] / count_phi # count is probability afterwards
    else:
        count_tagphi[tag] = 0

    for tag_prev in count_tagtag[tag]:
        j = tag_idx[tag_prev]
        prob_tagtag[i, j] = count_tagtag[tag][tag_prev] / count_tag[tag_prev]
    i += 1

f = open(test_path, 'r', encoding = 'utf-8')

test_tags = []
test_words = []
predict_tags = []
baseline_tags = []

fw = open(test_path + '.out', 'w')

for line in f.readlines():

    line = line.split()
    W = len(line)
    w = 0
    score = np.zeros((len_tag, W))
    backptr = np.zeros((len_tag, W))
    words = []

    for wordtag in line:

        word = wordtag[:wordtag.find('/')]
        words.append(word)
        tag = wordtag[wordtag.find('/') + 1:]
        test_words.append(word)
        test_tags.append(tag)
        prob_set = []

        for t in range(len_tag):
            if word in count_wordtag[tags[t]]:
                tmp = count_wordtag[tags[t]][word]
            else:
                tmp = 0
            # modified smoothing for better accuracy
            if w == 0:
                score[t, 0] = (tmp + 0.1) / (count_tag[tags[t]] + 0.1 * W_s) * count_tagphi[tags[t]]
            else:
                score[t, w] = (tmp + 0.1) / (count_tag[tags[t]] + 0.1 * W_s) * max(score[:, w-1] * prob_tagtag[t])
                backptr[t, w] = np.argmax(score[:, w-1] * prob_tagtag[t])
            prob_set.append(tmp)
        w += 1
        baseline_tags.append(tags[np.argmax(prob_set)])

    seq = []
    tmp = np.argmax(score[:, W-1])
    seq.append(tags[tmp])
    for w in range(W-2, -1, -1):
        tmp = int(backptr[tmp, w+1])
        seq.append(tags[tmp])
    seq.reverse()
    predict_tags.extend(seq)

    for i in range(len(words)):
        fw.write(words[i] + '/' + seq[i] + ' ')
    fw.write('\n')

fw.close()

f.close()
c_v = 0
c_b = 0
for i in range(len(predict_tags)):
    if predict_tags[i] == test_tags[i]:
        c_v += 1
    if baseline_tags[i] == test_tags[i]:
        c_b += 1
print('Accuracy of Viterbi: ' + str(c_v / len(predict_tags) * 100) + '%')
print('Accuracy of baseline: ' + str(c_b / len(predict_tags) * 100) + '%')
