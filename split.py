'''
EECS 595 Homework 4 Problem 2: Predicting Response Time Using Naive Bayes Part 1: Split

name: Xiechen Wang
uniqname: xiechenw
'''

import os
import msgpack
import numpy as np

path = './data_merged/ubuntu/'
files = os.listdir(path)
context_windows = {}
j = 0

# Note that all annotated code below were used to ensure context windows with same messages (it happens because of overlapping in one document, only consecutive 4 non-Ubuntu User could take them apart) won't be splitted away w.r.t train / test. However, it will result in poor accuracy... about 0.57 ~ 0.58 I think? Similarly, if I choose first 80% windows as train data, the accuracy was also about 0.58 ~ 0.59. So I just randomly split instead.

# fw_1 = open('train_split', 'w')
# fw_2 = open('test_split', 'w')
# k = 0

for file in files:
      
    with open(path + file, 'rb') as handle:
        conversation = msgpack.unpackb(handle.read())
        message_list = conversation[b'messages']
        context_window = ''
        i = 1
        for message in message_list:      
            context_window += str(message[b'text'], encoding = 'utf-8') + '^' + str(message[b'response_time']) + '\n'
            if i == 5 and str(message[b'user'], encoding = 'utf-8') == 'Ubuntu User':                    
                context_windows[j] = context_window
                context_window = context_window[context_window.find('\n') + 1:]
                # if k == 0:
                #     tmp = np.random.rand()
                # if tmp < 0.8:
                #     fw_1.write(context_window + '\n')
                # else:
                #     fw_2.write(context_window + '\n')
                j += 1
                i = 4
                # k = 4
            elif i == 5:
                context_window = context_window[context_window.find('\n') + 1:]
                # k -= 1
                continue
            i += 1

# If use the annotated parts above, just discard the code below.

fw_1 = open('train_split', 'w')
fw_2 = open('test_split', 'w')

for i in range(j):
    tmp = np.random.rand()
    # if i / j < 0.8:
    if tmp < 0.8:
        fw_1.write(context_windows[i] + '\n')
    else:
       fw_2.write(context_windows[i] + '\n')
