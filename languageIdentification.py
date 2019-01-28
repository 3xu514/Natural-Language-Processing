import numpy as np
import sys
import matplotlib.pyplot as plt

d = 100
learning_rate = 0.1
epoch = 3

def softmax(input):

    output = np.zeros((3,1))
    t = np.max(input)
    for i in range(3):
        output[i] = np.exp(input[i]-t)/(np.exp(input[0]-t)+np.exp(input[1]-t)+np.exp(input[2]-t))
    return output

def forward(parameters, x):

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1, x) + b1
    a1 = 1/(1+np.exp(-z1))
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    cache = {'a1': a1, 'a2':a2}

    return cache

def backward(parameters, cache, x):

    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']
    dL = a2 - y
    dz2 = np.zeros((3,1))
    for j in range(3):
        for i in range(3):
            if(i == j):
                dz2[j] += dL[i]*a2[i]*(1-a2[j])
            else:
                dz2[j] += -dL[i]*a2[i]*a2[j]
    dw2 = np.dot(dz2,a1.T)
    db2 = dz2
    da1 = np.dot(w2.T, dz2)
    dz1 = np.multiply(da1, np.multiply(a1,(1-a1)))
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    
    grad = {'dw1':dw1, 'dw2':dw2, 'db1':db1, 'db2':db2}
    
    return grad

def predict(parameters, X):

    N = np.size(X,1)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1, X) + np.dot(b1, np.ones([1,N]))
    a1 = 1/(1+np.exp(-z1))
    z2 = np.dot(w2, a1) + np.dot(b2, np.ones([1,N]))
    a2 = np.zeros([3, N])
    prediction = np.zeros([3, N])
    for i in range(N):
        a2[:,i] = softmax(z2[:,i]).squeeze()
        prediction[np.argmax(a2[:,i]),i] = 1
    return prediction.T

def getData(fileName):

    f = open(fileName, 'r', encoding='UTF-8')
    Data = []
    Label = []
    sentence = []
    s = 0
    for line in f.readlines():
        if len(line) > 5: 
            line = line.strip()
            label = line[:line.find(' ')]
            data_raw = line[line.find(' ') + 1:]
            if label == 'ENGLISH':
                label_onehot = [1,0,0]
            elif label == 'ITALIAN':
                label_onehot = [0,1,0]
            else:
                label_onehot = [0,0,1]
            
            data = ''
            for i in range(len(data_raw)):
                ch = data_raw[i]
                if ch != '\n':
                    data += ch
            if len(data) >= 5:
                for i in range(len(data) - 4):
                    token = data[i:i+5]
                    Data.append(token)
                    Label.append(label_onehot)
                    sentence.append(s)
                s += 1
    f.close()
    Data = np.array(Data)
    Label = np.array(Label)
    sentence = np.array(sentence)
    return Data, Label, sentence

def getTest(testPath, labelPath):

    f_test = open(testPath, 'r', encoding = 'ISO-8859-1')
    f_label = open(labelPath, 'r', encoding = 'UTF-8')
    test = []
    label = []
    sentence = []
    s = 0
    for line_test in f_test.readlines():
        line_label = f_label.readline()
        line_label = line_label.strip()
        line_label = line_label[line_label.find(' ') + 1:]
        data = line_test.strip()
        line_label = line_label.strip()
        if line_label == 'English':
            label_onehot = [1, 0, 0]
        elif line_label == 'Italian':
            label_onehot = [0, 1, 0]
        else:
            label_onehot = [0, 0, 1]
        for i in range(len(data) - 4):
            test.append(data[i:i+5])
            label.append(label_onehot)
            sentence.append(s)
        s += 1
    f_test.close()
    f_label.close()
    test = np.array(test)
    label = np.array(label)
    sentence = np.array(sentence)
    return test, label, sentence

def accuracy(label, prediction, sentence, show_lang = False):

    num = max(sentence) + 1
    sen_prediction = np.zeros([num, 3])
    sen_label = np.zeros([num, 3])
    if show_lang:
        f = open('languageIdentificationPart1.output', 'w')
    for i in range(num):
        temp = prediction[np.where(sentence == i)]
        sen_label[i] = label[np.where(sentence == i)][0]
        vote = np.sum(temp, axis = 0)
        sen_prediction[i, np.argmax(vote)] = 1
        if show_lang:
            if np.argmax(vote) == 0:
                f.write("Line" + str(i+1) + ' ENGLISH\n')
            elif np.argmax(vote) == 1:
                f.write("Line" + str(i+1) + ' ITALIAN\n')
            else:
                f.write("Line" + str(i+1) + ' FRENCH\n')
    if show_lang:
        f.close()
    dis = np.abs(sen_label - sen_prediction)
    dis = np.sum(dis,axis = 1)
    acc = 1 - np.count_nonzero(dis)/len(sen_label)
    return acc

def get_matrix(data, c):

    n = len(data)
    M = np.zeros([n, 5*c])
    for i in range(n):
        data_str = data[i]
        datum = [0] * (5 * c)
        for k in range(len(data_str)):
            ch = data_str[k]
            if ch in dic:
                datum[c * k + dic[ch]] = 1
        datum = np.array(datum).reshape(5 * c, 1)
        M[i, :] = datum.T
    
    return M

if __name__=='__main__':
    
    train_path = sys.argv[1]
    dev_path = sys.argv[2]
    test_path = sys.argv[3]
    test_solutions_path = test_path + "_solutions"

    fp = open(train_path, 'r', encoding='utf-8')
    text_train = fp.read()
    fp.close()
    dic = {}
    idx = 0

    for ch in text_train:
        if ch not in dic and ch != '\n':
            dic[ch] = idx
            idx += 1
    c = len(dic)
    
    train, train_label, train_sentence = getData(train_path)
    dev, dev_label, dev_sentence = getData(dev_path)
    test, test_label, test_sentence = getTest(test_path, test_solutions_path)

    parameters = {}
    parameters['w1'] = np.random.rand(d, 5*c)*0.01
    parameters['b1'] = np.zeros((d, 1))
    parameters['w2'] = np.random.rand(3, d)*0.01
    parameters['b2'] = np.zeros((3, 1))
    
    X = get_matrix(train, c)
    D = get_matrix(dev, c)
    T = get_matrix(test, c)

    cost = []
    acc_train = []
    acc_dev = []

    prediction_train = predict(parameters, X.T)
    prediction_dev = predict(parameters, D.T)
    acc_train.append(accuracy(train_label, prediction_train, train_sentence))
    acc_dev.append(accuracy(dev_label, prediction_dev, dev_sentence))

    for i in range(epoch):
    
        permutation = list(np.random.permutation(len(train)))
        shuffled_train = train[permutation]
        shuffled_train_label = train_label[permutation]
        shuffled_train_sentence = train_sentence[permutation]    

        for j in range(len(train)):
    
            x_str = shuffled_train[j]
            y = shuffled_train_label[j].reshape(3,1)
    
            x = [0] * (5*c)
            for k in range(len(x_str)):
                ch = x_str[k]
                if ch in dic:
                    x[c*k + dic[ch]] = 1
            x = np.array(x).reshape(5*c,1)
            X[j,:] = x.T
    
            cache = forward(parameters, x)
            grad = backward(parameters, cache, x)
    
            parameters['w1'] = parameters['w1'] - learning_rate*grad['dw1']
            parameters['b1'] = parameters['b1'] - learning_rate*grad['db1']
            parameters['w2'] = parameters['w2'] - learning_rate*grad['dw2']
            parameters['b2'] = parameters['b2'] - learning_rate*grad['db2']
    
        prediction_train = predict(parameters, X.T)
        prediction_dev = predict(parameters, D.T)
        acc_train.append(accuracy(shuffled_train_label, prediction_train, shuffled_train_sentence))
        acc_dev.append(accuracy(dev_label, prediction_dev, dev_sentence))

    # print(accuracy(shuffled_train_label, prediction_train, shuffled_train_sentence))
    # print(accuracy(dev_label, prediction_dev, dev_sentence))
    prediction_test = predict(parameters, T.T)
    acc_test = accuracy(test_label, prediction_test, test_sentence, True)
    # print('Test Arruracy Is: ' + str(acc_test))    
    plt.plot(acc_train, color = 'b', label = 'Train Set')
    plt.plot(acc_dev, color = 'r', label = 'Dev Set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = 2)
    plt.xticks(range(len(acc_train)))
    plt.show()
