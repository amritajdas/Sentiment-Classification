import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join

import numpy as np
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

## ---------------------------------------------------Importing DATA-----------------------------------------------##

from os import listdir
from os.path import isfile, join
positiveFiles = ['positive_how_r_u/' + f for f in listdir('positive_how_r_u/') if isfile(join('positive_how_r_u/', f))]
negativeFiles = ['negative_how_r_u/' + f for f in listdir('negative_how_r_u/') if isfile(join('negative_how_r_u/', f))]
neutralFiles = ['neutral/' + f for f in listdir('neutral/') if isfile(join('neutral/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
print('Negative files finished')

for ne in neutralFiles:
    with open(ne, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
print('Neutral files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

##----------------------------------------------------------------------------------------------------------------##

##------------------------------------------------- Data Preprossing----------------------------------------------##


POS_FILE = []  ## Positive File
val = ""
for i in positiveFiles:
    val = ""
    with open(i) as tata:
        for j in tata:
            val+=j
    POS_FILE.append(val)  

NEG_FILE = []  ## Negative File
val = ""
for i in negativeFiles:
    val = ""
    with open(i) as tata:
        for j in tata:
            val+=j
    NEG_FILE.append(val)        
    
NEUT_FILE = []  ## neutral File
val = ""
for i in neutralFiles:
    val = ""
    with open(i) as tata:
        for j in tata:
            val+=j
    NEUT_FILE.append(val)        

# Doing a first cleaning of the texts
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text    

pos_file_dic = {}  ## Postive Word Dictionary
for i in POS_FILE:
    for word in clean_text(i).split(" "):
        if word in pos_file_dic:
            pos_file_dic[word] += 1
        else :
            pos_file_dic[word] = 1          

neg_file_dic = {}  ## Negative Word Dictionary
for i in NEG_FILE:
    for word in clean_text(i).split(" "):
        if word in neg_file_dic:
            neg_file_dic[word] += 1
        else :
            neg_file_dic[word] = 1    

neut_file_dic = {}  ## Neutral Word Dictionary
for i in NEUT_FILE:
    for word in clean_text(i).split(" "):
        if word in neut_file_dic:
            neut_file_dic[word] += 1
        else :
            neut_file_dic[word] = 1  

## ------------------------------------ Data Cleaning With Certain Threshold----------------------------##

pos_thresh = 105 ## Positive Threshold For the Frequency

clean_pos = []
jus_a_var = []
for i in positiveFiles:
    jus_a_var = []
    with open(i) as tata:
        for j in tata:
            jus_a_var.append(j)
        clean_pos.append(jus_a_var)                      

final_clean_pos = []
final_clean_str = ''
for i in clean_pos:
    final_clean_str=''
    for word in i:
        for j in word.split(" "):
            if j in pos_file_dic:
                if pos_file_dic[j]>=pos_thresh:
                    final_clean_str =final_clean_str+j+" "
                    
    if len(final_clean_str)==0:
        continue
    else:
        final_clean_pos.append(final_clean_str)

neg_thresh= 350  ## Negative Threshold For The Frequency

clean_neg = []
jus_a_var = []
for i in negativeFiles:
    jus_a_var = []
    with open(i) as tata:
        for j in tata:
            jus_a_var.append(j)
        clean_neg.append(jus_a_var)

final_clean_neg = []
final_clean_str = ''
for i in clean_neg:
    final_clean_str=''
    for word in i:
        for j in word.split(" "):
            if j in neg_file_dic:
                if neg_file_dic[j]>=neg_thresh:
                    final_clean_str =final_clean_str+j+" "
                    
    if len(final_clean_str)==0:
        continue
    else:
        final_clean_neg.append(final_clean_str)
        
neut_thresh = 350  ## Neutral Threshold For The Frequency
                                       
clean_neut = []
jus_a_var = []
for i in neutralFiles:
    jus_a_var = []
    with open(i) as tata:
        for j in tata:
            jus_a_var.append(j)
        clean_neut.append(jus_a_var)

final_clean_neut = []
final_clean_str = ''
neut_thresh = 350 
for i in clean_neut:
    final_clean_str=''
    for word in i:
        for j in word.split(" "):
            if j in neut_file_dic:
                if neut_file_dic[j]>=neut_thresh:
                    final_clean_str =final_clean_str+j+" "
                    
    if len(final_clean_str)==0:
        continue
    else:
        final_clean_neut.append(final_clean_str)

pos_neg_neut = final_clean_pos + final_clean_neg + final_clean_neut  ## Total length (postive + Negative + Neutral)

## ------------------------------------------- Generate Emmbeddings------------------------------------------##
maxSeqLength = 50

ids = np.zeros((len(pos_neg_neut), maxSeqLength), dtype='int32')
fileCounter = 0
for pf in final_clean_pos:
    split = pf.split(" ")
    indexCounter = 0
    for word in split:
        try:
            ids[fileCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
            break
    fileCounter = fileCounter + 1 

for pf in final_clean_neg:
    split = pf.split(" ")
    indexCounter = 0
    for word in split:
        try:
            ids[fileCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
            break
    fileCounter = fileCounter + 1 
for pf in final_clean_neut:
    split = pf.split(" ")
    indexCounter = 0
    for word in split:
        try:
            ids[fileCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
            break
    fileCounter = fileCounter + 1 


#Pass into embedding function and see if it evaluates. 

np.save('idsMatrixnew', ids)
ids = np.load('idsMatrixnew.npy')
## --------------------------------------------------- Get the Train Test Split Batches -----------------------------##

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 3 == 0): 
            num = randint(1,len(final_clean_pos))
           # print("{} postivite".format(num))
            labels.append([1,0,0])
        elif (i%3 == 1):
            num = randint(len(final_clean_pos)+2,len(final_clean_pos)+len(final_clean_neg))
           #print("{} Negative".format(num))
            labels.append([0,1,0])
        else :
            num = randint(len(final_clean_pos)+len(final_clean_neg)+2,len(pos_neg_neut))
           #print("{} Negative".format(num))
            labels.append([0,0,1])
        arr[i] = ids[num-1:num]
        
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(1,len(pos_neg_neut))
        if (num <= len(final_clean_pos)):
            labels.append([1,0,0])
        elif num > len(final_clean_pos) and num < len(final_clean_pos)+len(final_clean_neg):
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

## ------------------------------------------------------- Model Traning -----------------------------------------------##

batchSize = len(ids[:])//100
lstmUnits = 64
numClasses = 3   #  3 Classes (Postive , Neutral , Negative)
iterations = 100000


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)
multilstm = tf.contrib.rnn.MultiRNNCell([lstmCell]*3)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime
with tf.Session() as sess:
    #print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
## --------------------------------------Tranning ----------------------------------------##
for i in range(iterations):
##Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   
    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

#    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "modelsnew/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()     

##----------------------------------- Accuracy ------------------------------------------##

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('modelsnew'))    

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
    
##--------------------------------------xxxxx---------------------------------------------##           

