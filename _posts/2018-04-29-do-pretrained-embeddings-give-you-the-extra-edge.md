---
title: "Do Pretrained Embeddings Give You The Extra Edge?"
date: "2018-04-29"
categories: 
  - "deep-learning"
  - "kaggle"
coverImage: "/post_images/Screen-Shot-2018-04-28-at-10.44.38-PM.png"
---

This is a repost from [my kernel](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge) at Kaggle, which has received several positive responses from the community that it's helpful to them. This is one of my kernels that tackles the interesting [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) at Kaggle, which aims to identify and classify toxic online comments.

In this kernel, we shall see if pretrained embeddings like Word2Vec, GLOVE and Fasttext, which are pretrained using billions of words could improve our accuracy score as compared to training our own embedding. We will compare the performance of models using these pretrained embeddings against the baseline model that doesn't use any pretrained embeddings in my previous kernel [here](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras).

![](https://qph.fs.quoracdn.net/main-qimg-3e812fd164a08f5e4f195000fecf988f)

Perhaps it's a good idea to briefly step in the world of word embeddings and see what's the difference between Word2Vec, GLOVE and Fasttext.

Embeddings generally represent geometrical encodings of words based on how frequently appear together in a text corpus. Various implementations of word embeddings described below differs in the way as how they are constructed.

**Word2Vec**

The main idea behind it is that you train a model on the context on each word, so similar words will have similar numerical representations.

Just like a normal feed-forward densely connected neural network(NN) where you have a set of independent variables and a target dependent variable that you are trying to predict, you first break your sentence into words(tokenize) and create a number of pairs of words, depending on the window size. So one of the combination could be a pair of words such as ('cat','purr'), where cat is the independent variable(X) and 'purr' is the target dependent variable(Y) we are aiming to predict.

We feed the 'cat' into the NN through an embedding layer initialized with random weights, and pass it through the softmax layer with ultimate aim of predicting 'purr'. The optimization method such as SGD minimize the loss function "(target word | context words)" which seeks to minimize the loss of predicting the target words given the context words. If we do this with enough epochs, the weights in the embedding layer would eventually represent the vocabulary of word vectors, which is the "coordinates" of the words in this geometric vector space.

![](/post_images/R8VLFs2.png)

The above example assumes the skip-gram model. For the Continuous bag of words(CBOW), we would basically be predicting a word given the context.

**GLOVE**

GLOVE works similarly as Word2Vec. While you can see above that Word2Vec is a "predictive" model that predicts context given word, GLOVE learns by constructing a co-occurrence matrix (words X context) that basically count how frequently a word appears in a context. Since it's going to be a gigantic matrix, we factorize this matrix to achieve a lower-dimension representation. There's a lot of details that goes in GLOVE but that's the rough idea.

**FastText**

FastText is quite different from the above 2 embeddings. While Word2Vec and GLOVE treats each word as the smallest unit to train on, FastText uses n-gram characters as the smallest unit. For example, the word vector ,"apple", could be broken down into separate word vectors units as "ap","app","ple". The biggest benefit of using FastText is that it generate better word embeddings for rare words, or even words not seen during training because the n-gram character vectors are shared with other words. This is something that Word2Vec and GLOVE cannot achieve.

Let's start off with the usual importing pandas, etc

In \[1\]:

import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad\_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import matplotlib.pyplot as plt
%matplotlib inline
import gensim.models.keyedvectors as word2vec
import gc

/opt/conda/lib/python3.6/site-packages/h5py/\_\_init\_\_.py:36: FutureWarning: Conversion of the second argument of issubdtype from \`float\` to \`np.floating\` is deprecated. In future, it will be treated as \`np.float64 == np.dtype(float).type\`.
  from .\_conv import register\_converters as \_register\_converters
Using TensorFlow backend.

Some preprocessing steps that we have taken in my earlier kernel.

In \[2\]:

train \= pd.read\_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test \= pd.read\_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
embed\_size\=0

In \[3\]:

list\_classes \= \["toxic", "severe\_toxic", "obscene", "threat", "insult", "identity\_hate"\]
y \= train\[list\_classes\].values
list\_sentences\_train \= train\["comment\_text"\]
list\_sentences\_test \= test\["comment\_text"\]

In \[4\]:

max\_features \= 20000
tokenizer \= Tokenizer(num\_words\=max\_features)
tokenizer.fit\_on\_texts(list(list\_sentences\_train))
list\_tokenized\_train \= tokenizer.texts\_to\_sequences(list\_sentences\_train)
list\_tokenized\_test \= tokenizer.texts\_to\_sequences(list\_sentences\_test)

In \[5\]:

maxlen \= 200
X\_t \= pad\_sequences(list\_tokenized\_train, maxlen\=maxlen)
X\_te \= pad\_sequences(list\_tokenized\_test, maxlen\=maxlen)

Since we are going to evaluate a few word embeddings, let's define a function so that we can run our experiment properly. I'm going to put some comments in this function below for better intuitions.

Note that there are quite a few GLOVE embeddings in Kaggle datasets, and I feel that it would be more applicable to use the one that was trained based on Twitter text. Since the comments in our dataset consists of casual, user-generated short message, the semantics used might be very similar. Hence, we might be able to capture the essence and use it to produce a good accurate score.

Similarly, I have used the Word2Vec embeddings which has been trained using Google Negative News text corpus, hoping that it's negative words can work better in our "toxic" context.

In \[6\]:

def loadEmbeddingMatrix(typeToLoad):
        #load different embedding file from Kaggle depending on which embedding 
        #matrix we are going to experiment with
        if(typeToLoad\=="glove"):
            EMBEDDING\_FILE\='../input/glove-twitter/glove.twitter.27B.25d.txt'
            embed\_size \= 25
        elif(typeToLoad\=="word2vec"):
            word2vecDict \= word2vec.KeyedVectors.load\_word2vec\_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary\=True)
            embed\_size \= 300
        elif(typeToLoad\=="fasttext"):
            EMBEDDING\_FILE\='../input/fasttext/wiki.simple.vec'
            embed\_size \= 300

        if(typeToLoad\=="glove" or typeToLoad\=="fasttext" ):
            embeddings\_index \= dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f \= open(EMBEDDING\_FILE)
            for line in f:
                #split up line into an indexed array
                values \= line.split()
                #first index is word
                word \= values\[0\]
                #store the rest of the values in the array as a new array
                coefs \= np.asarray(values\[1:\], dtype\='float32')
                embeddings\_index\[word\] \= coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings\_index))
        else:
            embeddings\_index \= dict()
            for word in word2vecDict.wv.vocab:
                embeddings\_index\[word\] \= word2vecDict.word\_vec(word)
            print('Loaded %s word vectors.' % len(embeddings\_index))
            
        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the 
        #same statistics for the rest of our own random generated weights. 
        all\_embs \= np.stack(list(embeddings\_index.values()))
        emb\_mean,emb\_std \= all\_embs.mean(), all\_embs.std()
        
        nb\_words \= len(tokenizer.word\_index)
        #We are going to set the embedding size to the pretrained dimension as we are replicating it.
        #the size will be Number of Words in Vocab X Embedding Size
        embedding\_matrix \= np.random.normal(emb\_mean, emb\_std, (nb\_words, embed\_size))
        gc.collect()

        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
        #our own dictionary and loaded pretrained embedding. 
        embeddedCount \= 0
        for word, i in tokenizer.word\_index.items():
            i\-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding\_vector \= embeddings\_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding\_vector is not None: 
                embedding\_matrix\[i\] \= embedding\_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
        
        del(embeddings\_index)
        gc.collect()
        
        #finally, return the embedding matrix
        return embedding\_matrix

The function would return a new embedding matrix that has the loaded weights from the pretrained embeddings for the common words we have, and randomly initialized numbers that has the same mean and standard deviation for the rest of the weights in this matrix.

Let's move on and load our first embeddings from Word2Vec.

In \[7\]:

embedding\_matrix \= loadEmbeddingMatrix('word2vec')

/opt/conda/lib/python3.6/site-packages/ipykernel\_launcher.py:30: DeprecationWarning: Call to deprecated \`wv\` (Attribute will be removed in 4.0.0, use self instead).

Loaded 3000000 word vectors.
total embedded: 66078 common words

In \[8\]:

embedding\_matrix.shape

Out\[8\]:

(210337, 300)

With the embedding weights, we can proceed to build a LSTM layer. The whole architecture is pretty much the same as the previous one I have done in the earlier kernel here, except that I have turned the LSTM into a bidirectional one, and added a dropout factor to it.

We start off with defining our input layer. By indicating an empty space after comma, we are telling Keras to infer the number automatically.

In \[9\]:

inp \= Input(shape\=(maxlen, )) #maxlen=200 as defined earlier

Next, we pass it to our Embedding layer, where we use the "weights" parameter to indicate the use of the pretrained embedding weights we have loaded and the "trainable" parameter to tell Keras **not to retrain** the embedding layer.

In \[10\]:

x \= Embedding(len(tokenizer.word\_index), embedding\_matrix.shape\[1\],weights\=\[embedding\_matrix\],trainable\=False)(inp)

Next, we pass it to a LSTM unit. But this time round, we will be using a Bidirectional LSTM instead because there are several kernels which shows a decent gain in accuracy by using Bidirectional LSTM.

How does Bidirectional LSTM work?

![](/post_images/jaKiP0S.png)

Imagine that the LSTM is split between 2 hidden states for each time step. As the sequence of words is being feed into the LSTM in a forward fashion, there's another reverse sequence that is feeding to the different hidden state at the same time. You might noticed later at the model summary that the output dimension of LSTM layer has doubled to 120 because 60 dimensions are used for forward, and another 60 are used for reverse.

The greatest advantage in using Bidirectional LSTM is that when it runs backwards you preserve information from the future and using the two hidden states combined, you are able in any point in time to preserve information from both past and future.

We are also introducing 2 more new mechanisms in this notebook: **LSTM Drop out and recurrent drop out.**

Why are we using dropout? You might have noticed that it's easy for LSTM to overfit, and in my previous notebook, overfitting problem starts to surface in just 2 epochs! Drop out is not something new to most of us, and these mechanisms applies the same dropout principles in a LSTM context.

![](/post_images/ksSyArD.png) LSTM Dropout is a probabilistic drop out layer on the inputs in each time step, as depict on the left diagram(arrows pointing upwards). On the other hand, recurrent drop out is something like a dropout mask that applies drop out between the hidden states throughout the recursion of the whole LSTM network, which is depicted on the right diagram(arrows pointing to the right).

These mechanisms could be set via the "dropout" and "recurrent\_dropout" parameters respectively. Please ignore the colors in the picture.

In \[11\]:

x \= Bidirectional(LSTM(60, return\_sequences\=True,name\='lstm\_layer',dropout\=0.1,recurrent\_dropout\=0.1))(x)

Okay! With the LSTM behind us, we'll feed the output into the rest of the layers which we have done so in the previous kernel.

In \[12\]:

x \= GlobalMaxPool1D()(x)

In \[13\]:

x \= Dropout(0.1)(x)

In \[14\]:

x \= Dense(50, activation\="relu")(x)

In \[15\]:

x \= Dropout(0.1)(x)

In \[16\]:

x \= Dense(6, activation\="sigmoid")(x)

In \[17\]:

model \= Model(inputs\=inp, outputs\=x)
model.compile(loss\='binary\_crossentropy',
                  optimizer\='adam',
                  metrics\=\['accuracy'\])

It's a good idea to see the whole architecture of the network before training as you wouldn't want to waste your precious time training on the wrong set-up.

In \[18\]:

model.summary()

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
Layer (type)                 Output Shape              Param #   
=================================================================
input\_1 (InputLayer)         (None, 200)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
embedding\_1 (Embedding)      (None, 200, 300)          63101100  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
bidirectional\_1 (Bidirection (None, 200, 120)          173280    
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
global\_max\_pooling1d\_1 (Glob (None, 120)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_1 (Dropout)          (None, 120)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_1 (Dense)              (None, 50)                6050      
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_2 (Dropout)          (None, 50)                0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_2 (Dense)              (None, 6)                 306       
=================================================================
Total params: 63,280,736
Trainable params: 179,636
Non-trainable params: 63,101,100
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Finally, we fire off the training process by aiming to run for 4 epochs with a batch size of 32. We save the training and validation loss in a variable so we can take a look and see if there's overfitting.

In \[19\]:

#batch\_size = 32
#epochs = 4
#hist = model.fit(X\_t,y, batch\_size=batch\_size, epochs=epochs, validation\_split=0.1)

The training of the model will take longer than what Kaggle kenel allows. I have pre-run it, and this is the result that you should roughly see

Train on 143613 samples, validate on 15958 samples

Epoch 1/4 143613/143613 \[==============================\] - 2938s 20ms/step - loss: 0.0843 - acc: 0.9739 - val\_loss: 0.0630 - val\_acc: 0.9786

Epoch 2/4 143613/143613 \[==============================\] - 3332s 23ms/step - loss: 0.0573 - acc: 0.9805 - val\_loss: 0.0573 - val\_acc: 0.9803

Epoch 3/4 143613/143613 \[==============================\] - 3119s 22ms/step - loss: 0.0513 - acc: 0.9819 - val\_loss: 0.0511 - val\_acc: 0.9817

Epoch 4/4 143613/143613 \[==============================\] - 3137s 22ms/step - loss: 0.0477 - acc: 0.9827 - val\_loss: 0.0498 - val\_acc: 0.9820

The result isn't too shabby but it's about the same as the baseline model which we train our own embedding. What about the other pretrained embeddings such as GLOVE and FastText? Let's try them out.

Over here, we are not going to repeat the whole process again. If you are running the notebook yourself, simply replace

In \[20\]:

#loadEmbeddingMatrix('word2vec')

with

In \[21\]:

#loadEmbeddingMatrix('glove') #for GLOVE or
#loadEmbeddingMatrix('fasttext') #for fasttext

to load the pretrained embedding from the different sources. For the sake of our benchmarking. I have pre-run it and collected all the results.

**GLOVE:**

Train on 143613 samples, validate on 15958 samples

Epoch 1/4 143613/143613 \[==============================\] - 2470s 17ms/step - loss: 0.1160 - acc: 0.9656 - val\_loss: 0.0935 - val\_acc: 0.9703

Epoch 2/4 143613/143613 \[==============================\] - 2448s 17ms/step - loss: 0.0887 - acc: 0.9721 - val\_loss: 0.0800 - val\_acc: 0.9737

Epoch 3/4 143613/143613 \[==============================\] - 2410s 17ms/step - loss: 0.0799 - acc: 0.9745 - val\_loss: 0.0753 - val\_acc: 0.9757

Epoch 4/4 143613/143613 \[==============================\] - 2398s 17ms/step - loss: 0.0753 - acc: 0.9760 - val\_loss: 0.0724 - val\_acc: 0.9768

**Fasttext:**

Train on 143613 samples, validate on 15958 samples

Epoch 1/4 143613/143613 \[==============================\] - 2800s 19ms/step - loss: 0.0797 - acc: 0.9757 - val\_loss: 0.0589 - val\_acc: 0.9795

Epoch 2/4 143613/143613 \[==============================\] - 2756s 19ms/step - loss: 0.0561 - acc: 0.9808 - val\_loss: 0.0549 - val\_acc: 0.9804

Epoch 3/4 143613/143613 \[==============================\] - 2772s 19ms/step - loss: 0.0507 - acc: 0.9819 - val\_loss: 0.0548 - val\_acc: 0.9811

Epoch 4/4 143613/143613 \[==============================\] - 2819s 20ms/step - loss: 0.0474 - acc: 0.9828 - val\_loss: 0.0507 - val\_acc: 0.9817

And of course, the same **baseline** model which doesn't use any pretrained embeddings, taken straight from the previous kenel except that we ran for 4 epochs:

Train on 143613 samples, validate on 15958 samples

Epoch 1/4 143613/143613 \[==============================\] - 5597s 39ms/step - loss: 0.0633 - acc: 0.9788 - val\_loss: 0.0480 - val\_acc: 0.9825

Epoch 2/4 143613/143613 \[==============================\] - 5360s 37ms/step - loss: 0.0448 - acc: 0.9832 - val\_loss: 0.0464 - val\_acc: 0.9828

Epoch 3/4 143613/143613 \[==============================\] - 5352s 37ms/step - loss: 0.0390 - acc: 0.9848 - val\_loss: 0.0470 - val\_acc: 0.9829

Epoch 4/4 129984/143613 \[==============================\] - 5050s 37ms/step - loss: 0.0386 - acc: 0.9858 - val\_loss: 0.0478 - val\_acc: 0.9830

It's easier if we plot the losses into graphs.

In \[22\]:

all\_losses \= {
'word2vec\_loss': \[0.084318213647104789,
  0.057314205012433353,
  0.051338302593577821,
  0.047672802178572039\],
 'word2vec\_val\_loss': \[0.063002561892695971,
  0.057253835496480658,
  0.051085027624451551,
  0.049801279793734249\],
'glove\_loss': \[0.11598931579683543,
  0.088738223480436862,
  0.079895263566000005,
  0.075343037429358703\],
 'glove\_val\_loss': \[0.093467933030432285,
  0.080007083813922117,
  0.075349041991106688,
  0.072366507668134517\],
 'fasttext\_loss': \[0.079714499498945865,
  0.056074704045674786,
  0.050703874653286324,
  0.047420131195761134\],
 'fasttext\_val\_loss': \[0.058888281775148932,
  0.054906051694414926,
  0.054768857866843601,
  0.050697043558286421\],
 'baseline\_loss': \[0.063304489498915865,
  0.044864004045674786,
  0.039013874651286124,
  0.038630130175761134\],
 'baseline\_val\_loss': \[0.048044281075148932,
  0.046414051594414926,
  0.047058757860843601,
  0.047886043558285421\]
}

In \[23\]:

#f, ax = plt.subplots(1)
epochRange \= np.arange(1,5,1)
plt.plot(epochRange,all\_losses\['word2vec\_loss'\])
plt.plot(epochRange,all\_losses\['glove\_loss'\])
plt.plot(epochRange,all\_losses\['fasttext\_loss'\])
plt.plot(epochRange,all\_losses\['baseline\_loss'\])
plt.title('Training loss for different embeddings')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(\['Word2Vec', 'GLOVE','FastText','Baseline'\], loc\='upper left')
plt.show()

![](data:image/png;base64,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
TkSuQmCC
)

Well, it certainly looks like the baseline has the minimum training loss. But before we close this case and pick the baseline model as the winner, this plot does not tell the full story as there seems to be some overfitting in the baseline model. It appears that from the 2nd epoch, overfitting has started to slip in as the validation loss has become higher than training loss.

In \[24\]:

epochRange \= np.arange(1,5,1)
plt.plot(epochRange,all\_losses\['baseline\_loss'\])
plt.plot(epochRange,all\_losses\['baseline\_val\_loss'\])
plt.title('Training Vs Validation loss for baseline model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(\['Training', 'Validation'\], loc\='upper left')
plt.show()

![](data:image/png;base64,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
)

What about the rest? Let's plot all the training/validation loss plots out to compare side by side.

In \[25\]:

f, ((ax1, ax2), (ax3, ax4)) \= plt.subplots(2, 2, sharex\='col', sharey\='row',figsize\=(20, 20))

plt.title('Training Vs Validation loss for all embeddings')
ax1.plot(epochRange,all\_losses\['baseline\_loss'\])
ax1.plot(epochRange,all\_losses\['baseline\_val\_loss'\])
ax1.set\_title('Baseline')
ax1.set\_ylim(0.03, 0.12)

ax2.plot(epochRange,all\_losses\['word2vec\_loss'\])
ax2.plot(epochRange,all\_losses\['word2vec\_val\_loss'\])
ax2.set\_title('Word2Vec')
ax2.set\_ylim(0.03, 0.12)

ax3.plot(epochRange,all\_losses\['glove\_loss'\])
ax3.plot(epochRange,all\_losses\['glove\_val\_loss'\])
ax3.set\_title('GLOVE')
ax3.set\_ylim(0.03, 0.12)

ax4.plot(epochRange,all\_losses\['fasttext\_loss'\])
ax4.plot(epochRange,all\_losses\['fasttext\_val\_loss'\])
ax4.set\_title('FastText')
ax4.set\_ylim(0.03, 0.12)

plt.show()

![](data:image/png;base64,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
kiRJkjAsUowOaZfP45eNpHXLHM6963VmfLwy7pIkSZIkSWr2DIsUqy5tWvL4ZSMpLszlvLvfYOr8
FXGXJEmSJElSs2ZYpNh1Ksrj8fEjOah1Hhfe+wavzl0ed0mSJEmSJDVbhkVKCyWtWvDY+JEc2i6f
i+6fxj/mLIu7JEmSJEmSmiXDIqWN9gW5PHrpSHqWFHDZAzN4flZZ3CVJkiRJktTsGBYprbTJz+GR
S0bS56BWfPuhGTz73pK4S5IkSZIkqVkxLFLaKWqZzUMXD2fQwa25+tG3+NPbn8ZdkiRJkiRJzYZh
kdJSYYtsHrhoOEMPacP3Hn+bJ2csirskSZIkSZKaBcMipa383Czu+9ZwRndvzz8/+Q6PvvFJ3CVJ
kiRJkpTxDIuU1vJyktx1wVC+cHgx//qH93hgykdxlyRJkiRJUkYzLFLaa5GdZMJ5Qzi+bwd++qeZ
3PXK/LhLkiRJkiQpYxkWqUnIzUpy2zmD+acBHfmvv8zm1n+Uxl2SJEmSJEkZKSvuAqTGyk4muGns
kWQn3+HXk+ZQVVPLd4/rSQgh7tIkSZIkScoYhkVqUrKSCX571hFkJxPc8MJcKqtr+ecTehkYSZIk
SZK0jxgWqclJJgL/e/pAspMJbvu/eVRW1/KTr/YxMJIkSZIkaR8wLFKTlEgEfvmN/uQkA3e9uoCq
mlr+8+v9SCQMjCRJkiRJ2huGRWqyQgj87OR+5GQluPOVBVTW1PKLUwcYGEmSJEmStBcMi9SkhRD4
t3/qQ05Wglv/MY/K6oj/PWMgSQMjSZIkSZI+F8MiNXkhBH44phc5ySTXv/Ah1bW1/ObMQWQlE3GX
JkmSJElSk2NYpIwQQuC7X+lJdlbgf/82h6qaWm4ceyTZBkaSJEmSJO0RwyJllCu+2IOcZIL/+sts
Kqvf5NZzjiQ3Kxl3WZIkSZIkNRluu1DGueSYw7j2lH68MLuMyx6cwaaqmrhLkiRJkiSpyTAsUkY6
f9Sh/PdpA3jpw3IuuX86GysNjCRJkiRJaozmExZVbYS//wLWlcVdiQ6QccO78uszBvHavOVceO8b
rN9cHXdJkiRJkiSlveYTFn08GV65Dm4cCM/+M6xeGHdFOgDOGNKF688+gukfr+L8e95g7aaquEuS
JEmSJCmtNZ+wqMdX4KrpMOBMmH4v3HQE/OlKWDEv7sq0n51yRGduGXck7yxczXl3vc6aDQZGkiRJ
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
FgEAAAAAACBAWAQAAAAAAIAAYREAAAAAAAAChEUAAAAAAAAIEBYBAAAAAAAgQFgEAAAAAACAAGER
AAAAAAAAAoRFAAAAAAAACBAWAQAAAAAAIEBYBAAAAAAAgABhEQAAAAAAAAKERQAAAAAAAAgQFgEA
AAAAACBAWAQAAAAAAIAAYREAAAAAAAAChEUAAAAAAAAIEBYBAAAAAAAgQFgEAAAAAACAAGERAAAA
AAAAAoRFAAAAAAAACBAWAQAAAAAAIEBYBAAAAAAAgABhEQAAAAAAAAKERQAAAAAAAAgQFgEAAAAA
ACCQVlhkZlea2XYz22lmXxzneIGZPeQfX29mC1KOzzOzHjP708mZNgAAAAAAAKbCCcMiMwtLukPS
VZKWSPqEmS1JGXaTpCPOuRZJ35R0e8rxb0p68tSnCwAAAAAAgKmUTmXRxZJ2OudanXMDkh6UtCpl
zCpJ9/mvH5F0hZmZJJnZRyW1StoyOVMGAAAAAADAVEknLGqUtCfh673+vnHHOOeGJHVKqjKzYkl/
Lumrpz5VAAAAAAAATLV0wiIbZ59Lc8xXJX3TOdfzvj/A7GYz22hmG9vb29OYEgAAAE4VazAAADCe
dMKivZLmJnzdJKltojFmlicpJqlD0iWSvm5muyV9XtJfmNltqT/AOXeXc26Fc25FTU1Nxh8CAAAA
mWMNBgAAxpOXxpgNkhaZWbOkfZJWS/r9lDFrJd0g6QVJ10p6xjnnJH1wZICZfUVSj3Pu25MwbwAA
AAAAAEyBE4ZFzrkhvxroKUlhSfc457aY2dckbXTOrZV0t6Q1ZrZTXkXR6qmcNAAAAAAAAKZGOpVF
cs6tk7QuZd+XE173S7ruBN/jKycxPwAAAAAAAEyjdHoWAQAAAAAAYJYgLAIAAAAAAECAsAgAAAAA
AAABwiIAAAAAAAAECIsAAAAAAAAQICwCAAAAAABAgLAIAAAAAAAAAcIiAAAAAAAABAiLAAAAAAAA
ECAsAgAAAAAAQICwCAAAAAAAAAHCIgAAAAAAAAQIiwAAAAAAABAgLAIAAAAAAECAsAgAAAAAAAAB
wiIAAAAAAAAECIsAAAAAAAAQICwCAAAAAABAgLAIAAAAAAAAAcIiAAAAAAAABAiLAAAAAAAAECAs
AgAAAAAAQICwCAAAAAAAAAHCIgAAAAAAAAQIiwAAAAAAABAgLAIAAAAAAECAsAgAAAAAAAABwiIA
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
)

With all the losses laid out, it's easy to see which the best option is. While it appears that GLOVE still some room to go before it overfits, the loss is high compared to the rest. On the other hand, Word2Vec and FastText starts to overfit at the 4rd and 3rd epochs respectively. So which one would you pick as the winner? In my opinion, **still the baseline model.**

So what went wrong? Aren't pretrained embeddings supposed to improve because it's trained with billions of words from tons of feature-rich corpus?

One probability is that these pretrained embeddings are not trained against text in the same context so the number of common words between our text and text that these pretrained embeddings were trained would be low. Let's plot the number of words we are using in the embedding layer.

In \[26\]:

wordCount \= {'word2vec':66078,'glove':81610,'fasttext':59613,'baseline':210337}

In \[27\]:

ind \= np.arange(0,4,1)  \# the x locations for the groups
width \= 0.35       \# the width of the bars

plt.title('Number of common words used in different embeddings')
embNames \= list(wordCount.keys())
embVals \= list(wordCount.values())
plt.barh(ind,embVals,align\='center', height\=0.5, color\='m',tick\_label\=embNames)
plt.show()

![](data:image/png;base64,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
)

From the above bar chart, it's obvious that the baseline would have the most words since the embedding layer is trained using the words in the dataset. The important takeaway is that the pretrained embeddings only contains about 60,000 words in common(less than half of baseline) and the embedding layer that is built from these pretrained weights couldn't represent the training data well enough.

Although building your own embedding takes a longer time, it might be worthwhile because it builds specifically for your context.

And that finally wraps up this kernel! I hope someone learnt something in this kernel. If you spot an error, feel free to let me know by commenting below.

Thanks for reading and good luck in the competition!

**TODO:**

1. There are many pretrained embeddings in Kaggle, and they are trained in different contexts of text corpus. You could try out other pretrained embeddings that is more suitable to the dataset in our competition.
2. Introduce LSTM drop out and recurrent drop out in baseline model, and tune the dropout rate to decrease overfitting.

In \[ \]:
