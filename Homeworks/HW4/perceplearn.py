import json
import os
import numpy as np
import re
import time
import sys

stop_words = ['a', 'about', 'after', 'all', 'also', 'always', 'am', 'an', 'and', 'any', 'are', 'at', 'be', 'been', 'being', 'but', 'by', 'came', 'can', 'cant', 'come', 'could', 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'else', 'for', 'from', 'get', 'give', 'goes', 'going', 'had', 'happen', 'has', 'have', 'having', 'how', 'i', 'if', 'ill', "i'm", 'in', 'into', 'is', "isn't", 'it', 'its', "i've", 'just', 'keep', 'let', 'like', 'made', 'make', 'many', 'may', 'me', 'mean', 'more', 'most', 'much', 'no', 'not', 'now', 'of', 'only', 'or', 'our', 'really', 'say', 'see', 'some', 'something', 'take', 'tell', 'than', 'that', 'the', 'their', 'them', 'then', 'they', 'thing', 'this', 'to', 'try', 'up', 'us', 'use', 'used', 'uses', 'very', 'want', 'was', 'way', 'we', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'without', 'wont', 'you', 'your', 'youre']
tf_label_map = {
    "True": 1,
    "Fake": -1
}
pn_label_map = {
    "Pos": 1,
    "Neg": -1
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

def readinput(filename):
    file=open(filename, 'r', encoding='utf-8')
    output=file.readlines()
    return output

def mult(X, w):
    return np.dot(X,w)

# http://www.ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf
# https://rasbt.github.io/mlxtend/user_guide/classifier/Perceptron/
# https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/#:~:text=The%20Perceptron%20algorithm%20is%20a,and%20predicts%20a%20class%20label.
# https://vitalflux.com/perceptron-explained-using-python-example/#:~:text=Perceptron%20is%20termed%20as%20machine,weights%20of%20the%20input%20signals
def perceptron_training(label_info, sentences_array, feature_len, itr=10):
    '''
    This will basically be like training two classifiers in parallel.
    One to classify pos or neg, the other for true or fake lell
    :param label_info:
    :param sentences_array:
    :param feature_len:
    :param itr:
    :return:
    '''
    vanilla_weights_tf = np.ones([feature_len, 1])
    averaged_weights_tf = np.ones([feature_len, 1])

    vanilla_weights_pn = np.ones([feature_len, 1])
    averaged_weights_pn = np.ones([feature_len, 1])

    avg_counter = 1

    for it in range(0, itr):
        print("EPOCH: ", it)
        internal_counter = 0
        tf_num_bad = 0
        pn_num_bad = 0
        for label_data, data_entry in zip(label_info, sentences_array):
            internal_counter += 1
            if internal_counter == 1:
                continue
            id = label_data[0]
            tf_label = tf_label_map[label_data[1]]
            pn_label = pn_label_map[label_data[2]]

            #TODO: Implement the averaged version of all of these

            #activation for one set of values
            vanilla_tf_activation = mult(data_entry, vanilla_weights_tf)[0]# + bias_tf
            vanilla_pn_activation = mult(data_entry, vanilla_weights_pn)[0]# + bias_pn

            # checking if signs are NOT aligned basically lel
            # updates the weights if not
            if tf_label*vanilla_tf_activation <= 0:
                tf_num_bad += 1
                vanilla_tf_update_amt = (data_entry*tf_label).reshape((feature_len, 1))
                vanilla_weights_tf = np.add(vanilla_weights_tf,vanilla_tf_update_amt)

                averaged_tf_update_amt = (data_entry*tf_label).reshape((feature_len, 1))
                averaged_tf_update_amt = (averaged_tf_update_amt * avg_counter).reshape((feature_len, 1))
                averaged_weights_tf = np.add(averaged_weights_tf, averaged_tf_update_amt )

            if pn_label*vanilla_pn_activation <= 0:
                pn_num_bad += 1
                vanilla_pn_update_amt = (data_entry*pn_label).reshape((feature_len, 1))
                vanilla_weights_pn = np.add(vanilla_weights_pn,vanilla_pn_update_amt)

                averaged_pn_update_amt = (data_entry * pn_label).reshape((feature_len, 1))
                averaged_pn_update_amt = (averaged_pn_update_amt * avg_counter).reshape((feature_len, 1))
                averaged_weights_pn = np.add(averaged_weights_pn, averaged_pn_update_amt)

            avg_counter += 1
        print("NUMBER TF MISMATCH: ", tf_num_bad)
        print("NUMBER PN MISMATCH: ", pn_num_bad)
    return vanilla_weights_tf, averaged_weights_tf, vanilla_weights_pn, averaged_weights_pn, avg_counter

def data_proc(lines):
    count = 0
    sentences = list()
    metadata_postproc = list()
    metadata_postproc.append([0,0,0])   # Garbage way to init, doing b/c
                                        # first row of sentences_postproc
                                        # is also, well, garbage
    for line in lines:
        new_sent = list()
        count += 1
        split_line = line.strip().split(' ')
        line_metadata = split_line[:3]
        line_text = split_line[3:]
        metadata_postproc.append(line_metadata)
        for word in line_text:
            proc_word = re.sub(r'[^\w\s]', '', word)
            proc_word = proc_word.lower()
            new_sent.append(proc_word)
        sentences.append(new_sent)

    #TODO: Frequency cutoff here since sorted returns list
    #
    grand_dict = sorted(bow_init(sentences))
    grand_dict = [x for x in grand_dict if not any(c.isdigit() for c in x)]
    grand_dict = grand_dict[1:] #gets rid of empty string @ front of the sorted list
    np.set_printoptions(suppress=True)
    sentences_postproc = np.empty([1,len(grand_dict)],dtype=np.int8)

    for idx, line in enumerate(sentences):
        ret_dict = construct_vector(grand_dict, line).values()
        temp_vec = np.asarray(list(ret_dict))
        sentences_postproc = np.vstack([sentences_postproc, temp_vec])
    # np.savetxt("training_sentences.csv", sentences_postproc, fmt='%i', delimiter=",")

    return metadata_postproc, sentences_postproc, len(grand_dict), grand_dict


if __name__ == "__main__":
    temp = readinput(sys.argv[1]) #readinput("/Users/devyanbiswas/Desktop/CSCI544/Homeworks/HW4/perceptron-training-data/train-labeled.txt")
    # start = time.time()
    metadata_list, sentences_nparray, feature_len, feat_words = data_proc(temp)
    # print(metadata_list)
    # print(sentences_nparray)
    # print(time.time() - start)
    # print(len(metadata_list), len(sentences_nparray))
    weights_tf, averaged_tf, weights_pn, averaged_pn, avg_counter = perceptron_training(metadata_list, sentences_nparray, feature_len)
    weights_tf = weights_tf.reshape(feature_len)
    averaged_tf = (averaged_tf * (1/avg_counter)).reshape(feature_len)
    weights_pn = weights_pn.reshape(feature_len)
    averaged_pn = (averaged_pn * (1/avg_counter)).reshape(feature_len)
    # print(weights_pn - averaged_pn)
    # print("=====")
    # print(weights_tf - averaged_tf)
    np.savetxt('vanillamodel.txt', (weights_tf, weights_pn, feat_words), fmt='%s', delimiter=',')
    np.savetxt('averagedmodel.txt', (weights_tf - averaged_tf, weights_pn - averaged_pn, feat_words), fmt='%s', delimiter=',')
