'''
EECS 595 Homework 4 Problem 2: Predicting Response Time Using Naive Bayes Part 2: Naive Bayes

name: Xiechen Wang
uniqname: xiechenw
'''

import numpy as np
import sys

# stopwords = []
# f = open('stopwords', 'r')
# for line in f.readlines():
#     line = line.strip()
#     stopwords.append(line)
# f.close()

def trainNaiveBayes(train_path):

    i = 0
    window = ''
    wordspeed = {}
    speed_c = [0, 0]

    f = open(train_path, 'r')
    for line in f.readlines():
        line = line.strip()
        if '^' not in line:
            continue
        text = line[:line.find('^')]
        time = line[line.find('^') + 1:]
        while '^' in time:
            text = text + '^' + time[:time.find('^')]
            time = time[time.find('^') + 1:]
        i += 1
        window = window + text + ' '
        if i == 5:
            words = window.split()          
            window = ''
            i = 0
            if int(time) > 60:
                flag = 1
            else:
                flag = 0
            speed_c[flag] += 1
            for word in words:
                # if word in stopwords:
                #     continue
                if word in wordspeed:
                    wordspeed[word][flag] += 1
                else:
                    if flag:
                        wordspeed[word] = [0, 1]
                    else:
                        wordspeed[word] = [1, 0]
    f.close()
    return wordspeed, speed_c

def testNaiveBayes(test_path, wordspeed, speed_c):
    
    test_labels = []
    predicted_labels = []
    f_count = 0
    s_count = 0
    f_k = 0
    s_k = 0
    for word in wordspeed:
        if wordspeed[word][0] > 0:
            f_count += wordspeed[word][0]
            f_k += 1
        if wordspeed[word][1] > 0:
            s_count += wordspeed[word][1]
            s_k += 1
    i = 0
    f = open(test_path, 'r')
    window = ''

    for line in f.readlines():
        line = line.strip()
        if '^' not in line:
            continue
        text = line[:line.find('^')]
        time = line[line.find('^') + 1:]
        while '^' in time:
            text = text + '^' + time[:time.find('^')]
            time = time[time.find('^') + 1:]
        i += 1   
        if i == 5:
            words = window.split()
            window = ''
            i = 0
            pw_s = speed_c[1] / sum(speed_c)
            pw_f = speed_c[0] / sum(speed_c)
            for word in words:
                # if word in stopwords:
                #     continue
                if word in wordspeed:
                    pw_s = pw_s * (wordspeed[word][1] + 1) / (s_count + s_k)
                    pw_f = pw_f * (wordspeed[word][0] + 1) / (f_count + f_k)
                else:
                    pw_s = pw_s * 1 / (s_count + s_k)
                    pw_f = pw_f * 1 / (f_count + f_k)
            if pw_s > pw_f:
                predicted_labels.append('slow')
            else:
                predicted_labels.append('fast')
            if int(time) <= 60:
                test_labels.append('fast')
            else:
                test_labels.append('slow')
        else:
            window = window + text + ' '        

    f.close()
    return test_labels, predicted_labels

if __name__=='__main__':

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    wordspeed, speed_c = trainNaiveBayes(train_path)
    test_labels, predicted_labels = testNaiveBayes(test_path, wordspeed, speed_c)
    acc = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predicted_labels[i]:
            acc += 1 / len(test_labels)
    print('Accuracy = ' + str(acc))
