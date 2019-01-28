'''
EECS 595 Homework 1 Problem 1: Sentence Boundary Detection

name: Xiechen Wang
uniqname: xiechenw
'''

import sys
from sklearn import tree

filename_1 = sys.argv[1]
filename_2 = sys.argv[2]

fp = open(filename_1, 'r')
text_train = fp.read()
fp.close()
fp = open(filename_2, 'r')
text_test = fp.read()
fp.close()

words_train = []
tokens_train = []
words_test = []
tokens_test = []
word_flag = False
token_flag = False
text = ''

for i in range(len(text_train)):
    
    if text_train[i] == ' ':
        if word_flag:
            if text[0].isupper() or text[0].islower() or text[0].isdigit():
                token_flag = True
                words_train.append(text)
            text = ''
            word_flag = False
        else:
            text = ''
            word_flag = True   
        
    elif text_train[i] == '\n' and token_flag:
        tokens_train.append(text)
        text = ''
        token_flag = False

    else:
        text += text_train[i]

for i in range(len(text_test)):
    
    if text_test[i] == ' ':
        if word_flag:
            if text[0].isupper() or text[0].islower() or text[0].isdigit():
                token_flag = True
                words_test.append(text)
            text = ''
            word_flag = False
        else:
            text = ''
            word_flag = True   
        
    elif text_test[i] == '\n' and token_flag:
        tokens_test.append(text)
        text = ''
        token_flag = False

    else:
        text += text_test[i]

data_train = []
labels_train = []
data_test = []
labels_test = []
dictionary = []

for i in range(len(words_train)):

    if tokens_train[i] == 'EOS' or tokens_train[i] == 'NEOS':

        datum = []
        if tokens_train[i] == 'EOS':
            labels_train.append(1)
        else:
            labels_train.append(0)

        word_left = words_train[i]
        if word_left[len(word_left)-1] == '.':
            word_left = word_left[0:len(word_left)-1]
        if i < len(words_test) - 1:
            word_right = words_train[i+1]
            if word_right[len(word_right)-1] == '.':
                word_right = word_right[0:len(word_right)-1]
	else:
	    word_right = ' '

	# core features
	
	if word_left in dictionary:
	    datum.append(dictionary.index(word_left)+1)
	else:
	    dictionary.append(word_left)
	    datum.append(len(dictionary))

	if word_right in dictionary:
	    datum.append(dictionary.index(word_right)+1)
	else:
	    dictionary.append(word_right)
	    datum.append(len(dictionary))	
        if len(word_left) < 3:
            datum.append(True)
        else:
            datum.append(False)
        if word_left[0].isupper:
            datum.append(True)
        else:
            datum.append(False)
        if word_right[0].isupper:
            datum.append(True)
        else:
            datum.append(False)
	
	# my features
	
	if len(word_right) < 3:
            datum.append(True)
        else:
            datum.append(False)
	if word_right[0].isdigit:
            datum.append(True)
        else:
            datum.append(False)
	if word_left[0].isdigit:
            datum.append(True)
        else:
            datum.append(False)
	
        data_train.append(datum)

for i in range(len(words_test)):

    if tokens_test[i] == 'EOS' or tokens_test[i] == 'NEOS':

        datum = []
        if tokens_test[i] == 'EOS':
            labels_test.append(1)
        else:
            labels_test.append(0)

        word_left = words_test[i]
        if word_left[len(word_left)-1] == '.':
            word_left = word_left[0:len(word_left)-1]
	if i < len(words_test) - 1:
            word_right = words_test[i+1]
	    if word_right[len(word_right)-1] == '.':
                word_right = word_right[0:len(word_right)-1]
	else:
	    word_right = ' '	  

	# core features
	
	if word_left in dictionary:
	    datum.append(dictionary.index(word_left)+1)
	else:
	    datum.append(0)

	if word_right in dictionary:
	    datum.append(dictionary.index(word_right)+1)
	else:
	    datum.append(0)
	
        if len(word_left) < 3:
            datum.append(True)
        else:
            datum.append(False)
        if word_left[0].isupper:
            datum.append(True)
        else:
            datum.append(False)
        if word_right[0].isupper:
            datum.append(True)
        else:
            datum.append(False)
	
	# my features
	
	if len(word_right) < 3:
            datum.append(True)
        else:
            datum.append(False)
	if word_right[0].isdigit:
            datum.append(True)
        else:
            datum.append(False)
	if word_left[0].isdigit:
            datum.append(True)
        else:
            datum.append(False)
	
        data_test.append(datum)

data_train_final = []
data_test_final = []
D = len(dictionary)

for i in range(len(data_train)):
    datum = []
    one_hot = [0]
    for d in range(D):
	if data_train[i][0] == d+1:
	    one_hot.append(1)
	else:
    	    one_hot.append(0)
    datum.extend(one_hot)
    one_hot = [0]
    for d in range(D):
	if data_train[i][1] == d+1:
	    one_hot.append(1)
	else:
    	    one_hot.append(0)
    datum.extend(one_hot)
    datum.extend(data_train[i][2:len(data_train[i])])
    data_train_final.append(datum)

for i in range(len(data_test)):
    datum = []
    if data_test[i][0] == 0:
	one_hot = [1]
    else:
	one_hot = [0]
    for d in range(D):
	if data_test[i][0] == d+1:
	    one_hot.append(1)
	else:
    	    one_hot.append(0)
    datum.extend(one_hot)
    if data_test[i][1] == 0:
	one_hot = [1]
    else:
	one_hot = [0]
    for d in range(D):
	if data_test[i][1] == d+1:
	    one_hot.append(1)
	else:
    	    one_hot.append(0)
    datum.extend(one_hot)
    datum.extend(data_test[i][2:len(data_test[i])])
    data_test_final.append(datum)

clf = tree.DecisionTreeClassifier()
clf.fit(data_train_final, labels_train)
labels_train_predicted = clf.predict(data_train_final)
labels_test_predicted = clf.predict(data_test_final)
print('Train Accuracy = ' + str(100*(1-float(sum(abs(labels_train_predicted-labels_train)))/len(labels_train))) + '%')
print('Test Accuracy = ' + str(100*(1-float(sum(abs(labels_test_predicted-labels_test)))/len(labels_test))) + '%')

fp = open(filename_2 + '.out', 'w')
j = 0

for i in range(len(text_test)):
    
    if text_test[i] == ' ':
	
        if word_flag:
	
	    thisword = text
            word_flag = False
            if (text[0].isupper() or text[0].islower() or text[0].isdigit()) and text_test[i+1] != 'T':
		if labels_test_predicted[j] == 1:
		    thistoken = 'EOS'
		else:
		    thistoken = 'NEOS'
	    	j += 1
            else:
		thistoken = 'TOK'
	    fp.write(thisnumber + ' ' + thisword + ' ' + thistoken +'\n')
            text = ''

        else:

            thisnumber = text
	    text = ''
            word_flag = True
	    text = ''   

    else:
	text += text_test[i]
	if text_test[i] == '\n':
		text = ''

fp.close()
