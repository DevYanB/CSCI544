import json
import os
import numpy as np
import re
import time
import sys

stop_words = ['a', 'about', 'after', 'all', 'also', 'always', 'am', 'an', 'and', 'any', 'are', 'at', 'be', 'been', 'being', 'but', 'by', 'came', 'can', 'cant', 'come', 'could', 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'else', 'for', 'from', 'get', 'give', 'goes', 'going', 'had', 'happen', 'has', 'have', 'having', 'how', 'i', 'if', 'ill', "i'm", 'in', 'into', 'is', "isn't", 'it', 'its', "i've", 'just', 'keep', 'let', 'like', 'made', 'make', 'many', 'may', 'me', 'mean', 'more', 'most', 'much', 'no', 'not', 'now', 'of', 'only', 'or', 'our', 'really', 'say', 'see', 'some', 'something', 'take', 'tell', 'than', 'that', 'the', 'their', 'them', 'then', 'they', 'thing', 'this', 'to', 'try', 'up', 'us', 'use', 'used', 'uses', 'very', 'want', 'was', 'way', 'we', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'without', 'wont', 'you', 'your', 'youre']
#NOTE: BIAS TERM WILL 10000% REMOVE NEED FOR 0 MAPPING
tf_label_map = {
    1: "True",
    -1: "Fake",
    0: "Fake"
}
pn_label_map = {
    1: "Pos",
    -1: "Neg",
    0: "Neg"
}

def bow_init(sentences):
    '''
    Standard Bag of Words implementation: dict
    :param sentences:
    :return:
    '''
    bag = dict()
    for line in sentences:
        for word in line:
            if word not in stop_words:
                if word in bag:
                    bag[word] +=1
                else:
                    bag[word] = 1
    return bag

def construct_vector(bow, sentence):
    vect = dict.fromkeys(bow, 0)
    for word in sentence:
        if word not in stop_words and word in bow:
            vect[word] += 1
    return vect

def readinput(filename1):
    file1=open(filename1, 'r', encoding='utf-8')
    output1=file1.readlines()
    return output1

def data_proc(bow, input_sentences):
    count=0
    sentences = list()
    id_list = list()
    for line in input_sentences:
        new_sent = list()
        count += 1
        split_line = line.strip().split(' ')
        line_metadata = split_line[:1]
        id_list.append(line_metadata)
        line_text = split_line[1:]
        for word in line_text:
            proc_word = re.sub(r'[^\w\s]', '', word)
            proc_word = proc_word.lower()
            new_sent.append(proc_word)
        sentences.append(new_sent)

    sentences_postproc = np.empty([1, len(bow)], dtype=np.int8)
    for idx, line in enumerate(sentences):
        ret_dict = construct_vector(bow, line).values()
        temp_vec = np.asarray(list(ret_dict))
        sentences_postproc = np.vstack([sentences_postproc, temp_vec])

    #REMEMBER: FIRST ROW OF SENTENCES_POSTPROC IS A TRASH VALUE

    return sentences_postproc, id_list

def mult(X, w):
    return np.dot(X,w)

def pred(X, w_tf, w_pn):
    tf_activation_array = mult(X, w_tf)
    pn_activation_array = mult(X, w_pn)

    tf_preds = np.sign(tf_activation_array)
    pn_preds = np.sign(pn_activation_array)

    tf_preds = np.vectorize(tf_label_map.get)(tf_preds)
    pn_preds = np.vectorize(pn_label_map.get)(pn_preds)

    return tf_preds, pn_preds

if __name__ == "__main__":
    input_sentences = readinput(sys.argv[2]) #readinput("/Users/devyanbiswas/Desktop/CSCI544/Homeworks/HW4/perceptron-training-data/dev-text.txt")
    model_info = readinput(sys.argv[1]) #readinput("/Users/devyanbiswas/Desktop/CSCI544/Homeworks/HW4/vanillamodel.txt")

    tf_weights = model_info[0].split(',')
    tf_weights = [float(x) for x in tf_weights]
    tf_weights = np.asarray(tf_weights)

    pn_weights = model_info[1].split(',')
    pn_weights = [float(x) for x in pn_weights]
    pn_weights = np.asarray(pn_weights)

    feature_list =  model_info[2].split(',')

    sent_nparray, id_list = data_proc(feature_list, input_sentences)
    #REMOVES GARBAGE VALUE FIRST ROW LELL
    sent_nparray = np.delete(sent_nparray, (0), axis=0)

    tf_preds, pn_preds = pred(sent_nparray, tf_weights, pn_weights)
    tf_preds = list(tf_preds)
    pn_preds = list(pn_preds)

    with open('percepoutput.txt', 'w', encoding='utf-8') as f:
        for idx, id  in enumerate(id_list):
            write_str = str(id[0]) + " " + str(tf_preds[idx]) + " " + str(pn_preds[idx])
            f.write(write_str)
            f.write("\n")

