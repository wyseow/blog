---
title: "Go Even Deeper with Char-Gram+CNN"
date: "2018-04-29"
categories: 
  - "deep-learning"
  - "kaggle"
coverImage: "/post_images/Screen-Shot-2018-04-28-at-10.44.38-PM.png"
---

This is a repost from [my kernel](https://www.kaggle.com/sbongo/for-beginners-go-even-deeper-with-char-gram-cnn) at Kaggle, which has received several positive responses from the community that it's helpful to them. This is one of my kernels that tackles the interesting [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) at Kaggle, which aims to identify and classify toxic online comments.

**In this notebook, we are going to tackle the same toxic classification problem just like my previous notebooks but this time round, we are going deeper with the use of Character-level features and Convolutional Neural Network (CNNs).**

**_Updated with saved model and submission below_**

![](/post_images/okCCLAU.jpg)

**Why do we consider the idea of using char-gram features?**

You might noticed that there are a lot of sparse misspellings due to the nature of the dataset. When we train our model using the word vectors from our training set, we might be missing out some genuine words and mispellings that are not present in the training set but yet present in our prediction set. Sometimes that wouldn't affect the model's capability to make good judgement, but most of the time, it's unable to correctly classify because the misspelt words are not in the model's "dictionary".

Hence, if we could "go deeper" by splitting the sentence into a list of characters instead of words, the chances that the same characters that are present in both training and prediction set are much higher. You could imagine that this approach introduce another problem: an explosion of dimensions. One of the ways to tackle this problem is to use CNN as it's designed to solve high-dimensional dataset like images. Traditionally, CNN is used to solve computer vision problems but there's an increased trend of using CNN not just in Kaggle competitions but also in papers written by researchers too. Therefore, I believe it deserve a writeup and without much ado, let's see how we can apply CNN to our competition at hand.

I have skipped some elaboration of some concepts like embeddings which I have went through in my previous notebooks, so take a look at these if you are interested in learning more:

- [Do Pretrained Embeddings Give You The Extra Edge?](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
- [\[For Beginners\] Tackling Toxic Using Keras](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras)

**A brief glance at Convolutional Neural Network (CNNs)**

CNN is basically a feed-forward neural network that consists of several layers such as the convolution, pooling and some densely connected layers that we are familiar with.

![](/post_images/aa46tRe.png)

Firstly, as seen in the above picture, we feed the data(image in this case) into the convolution layer. The convolution layer works by sliding a window across the input data and as it slides, the window(filter) applies some matrix operations with the underlying data that falls in the window. And when you eventually collect all the result of the matrix operations, you will have a condensed output in another matrix(we call it a feature map).

![](/post_images/wSbiLCi.gif)

With the resulting matrix at hand, you do a max pooling that basically down-samples or in another words decrease the number of dimensions without losing the essence.

![](/post_images/Cphci9k.png)

Consider this simplified image of max pooling operation above. In the above example, we slide a 2 X 2 filter window across our dataset in strides of 2. As it's sliding, it grabs the maximum value and put it into a smaller-sized matrix.

There are different ways to down-sample the data such as min-pooling, average-pooling and in max-pooling, you simply take the maximum value of the matrix. Imagine that you have a list: \[1,4,0,8,5\]. When you do max-pooling on this list, you will only retain the value "8". Indirectly, we are only concerned about the existence of 8, and not the location of it. Despite it's simplicity, it's works quite well and it's a pretty niffy way to reduce the data size.

Again, with the down-sized "after-pooled" matrix, you could feed it to a densely connected layer which eventually leads to prediction.

**How does this apply to NLP in our case?**

Now, forget about real pixels about a minute and imagine using each tokenized character as a form of pixel in our input matrix. Just like word vectors, we could also have character vectors that gives a lower-dimension representation. So for a list of 10 sentences that consists of 50 characters each, using a 30-dimensional embedding will allow us to feed in a 10x50x30 matrix into our convolution layer. ![](/post_images/g59nKYc.jpg) Looking at the above picture, let's just focus(for now) on 1 sentence instead of a list. Each character is represented in a row (8 characters), and each embedding dimension is represented in a column (5 dimensions) in this starting matrix.

You would begin the convolution process by using filters of different dimensions to "slide" across your initial matrix to get a lower-dimension feature map. There's something I deliberately missed out earlier: filters.

![](/post_images/Lwa7wBG.gif) The sliding window that I mention earlier are actually filters that are designed to capture different distinctive features in the input data. By defining the dimension of the filter, you can control the window of infomation you want to "summarize". To translate back in the picture, each of the feature maps could contain 1 high level representation of the embeddings for each character.

Next, we would apply a max pooling to get the maximum value in each feature map. In our context, some characters in each filter would be selected through this max pooling process based on their values. As usual, we would then feed into a normal densely connected layer that outputs to a softmax function which gives the probabilities of each class.

Note that my explanation hides some technical details to facilitate understanding. There's a whole load of things that you could tweak with CNN. For instance, the stride size which determine how often the filter will be applied, narrow VS wide CNN, etc.

**Okay! Let's see how we could implement CNN in our competition.**

As always, we start off with the importing of relevant libraries and dataset:

In \[1\]:

import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad\_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU,Conv1D,MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from sklearn.model\_selection import train\_test\_split
from keras.models import load\_model

/opt/conda/lib/python3.6/site-packages/h5py/\_\_init\_\_.py:36: FutureWarning: Conversion of the second argument of issubdtype from \`float\` to \`np.floating\` is deprecated. In future, it will be treated as \`np.float64 == np.dtype(float).type\`.
  from .\_conv import register\_converters as \_register\_converters
Using TensorFlow backend.

In \[2\]:

train \= pd.read\_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
submit \= pd.read\_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
submit\_template \= pd.read\_csv('../input/jigsaw-toxic-comment-classification-challenge/sample\_submission.csv', header \= 0)

Split into training and test set:

In \[3\]:

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(train, train\[\["toxic", "severe\_toxic", "obscene", "threat", "insult", "identity\_hate"\]\], test\_size \= 0.10, random\_state \= 42)

Store the comments as seperate variables for further processing.

In \[4\]:

list\_sentences\_train \= X\_train\["comment\_text"\]
list\_sentences\_test \= X\_test\["comment\_text"\]
list\_sentences\_submit \= submit\["comment\_text"\]

In our previous notebook, we have began using Kera's helpful Tokenizer class to help us do the gritty text processing work. We are going to use it again to help us split the text into characters by setting the "char\_level" parameter to true.

In \[5\]:

max\_features \= 20000
tokenizer \= Tokenizer(num\_words\=max\_features,char\_level\=True)

This function allows Tokenizer to create an index of the tokenized unique characters. Eg. a=1, b=2, etc

In \[6\]:

tokenizer.fit\_on\_texts(list(list\_sentences\_train))

Then we get back a list of sentences with the sequence of indexes which represent each character.

In \[7\]:

list\_tokenized\_train \= tokenizer.texts\_to\_sequences(list\_sentences\_train)
list\_sentences\_test \= tokenizer.texts\_to\_sequences(list\_sentences\_test)
list\_tokenized\_submit \= tokenizer.texts\_to\_sequences(list\_sentences\_submit)

Since there are sentences with varying length of characters, we have to get them on a constant size. Let's put them to a length of 500 characters for each sentence:

In \[8\]:

maxlen \= 500
X\_t \= pad\_sequences(list\_tokenized\_train, maxlen\=maxlen)
X\_te \= pad\_sequences(list\_sentences\_test, maxlen\=maxlen)
X\_sub \= pad\_sequences(list\_tokenized\_submit, maxlen\=maxlen)

Just in case you are wondering, the reason why I used 500 is because most of the number of characters in a sentence falls within 0 to 500:

In \[9\]:

totalNumWords \= \[len(one\_comment) for one\_comment in list\_tokenized\_train\]
plt.hist(totalNumWords)
plt.show()

![](data:image/png;base64,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
kqRu/wTfRPaTMRa8JQAAAABJRU5ErkJggg==
)

Finally, we can start buliding our model.

First, we set up our input layer. As mentioned in the Keras documentation, we have to include the shape for the very first layer and Keras will automatically derive the shape for the rest of the layers.

In \[10\]:

inp \= Input(shape\=(maxlen, ))

We use an embedding size of 240. That also means that we are projecting characters on a 240-dimension vector space. It will output a (num of sentences X 500 X 240) matrix. We have talked about embedding layer in my previous notebooks, so feel free to check them out.

In \[11\]:

embed\_size \= 240
x \= Embedding(len(tokenizer.word\_index)+1, embed\_size)(inp)

Here's the meat of our notebook. With the output of embedding layer, we feed it into a convolution layer. We use a window size of 4 (remember it's 5 in the earlier picture above) and 100 filters (it's 6 in the earlier picture above) to extract the features in our data. That also means we slides a window across the 240 dimensions of embeddings for each of the 500 characters and it will result in a (num of sentences X 500 X 100) matrix. Notice that we have set padding to "same". What does this padding means? ![](/post_images/hITQent.png) For simplicity sake, let's imagine we have a 32 x 32 x 3 input matrix and a 5 x 5 x 3 filter, if you apply the filter on the matrix with 1 stride, you will end up with a 28 x 28 x 3 matrix. In the early stages, you would want to preserve as much information as possible, so you will want to have a 32 x 32 x 3 matrix back. If we add(padding) some zeros around the original input matrix, we will be sure that the result output matrix dimension will be the same. But if you really want to have the resulting matrix to be reduced, you can set the padding parameter to "valid".

In \[12\]:

x \= Conv1D(filters\=100,kernel\_size\=4,padding\='same', activation\='relu')(x)

Then we pass it to the max pooling layer that applies the max pool operation on a window of every 4 characters. And that is why we get an output of (num of sentences X 125 X 100) matrix.

In \[13\]:

x\=MaxPooling1D(pool\_size\=4)(x)

Next, we pass it to the Bidriectional LSTM that we are famliar with, since the previous notebook.

In \[14\]:

x \= Bidirectional(GRU(60, return\_sequences\=True,name\='lstm\_layer',dropout\=0.2,recurrent\_dropout\=0.2))(x)

Afterwhich, we apply a max pooling again but this time round, it's a global max pooling. What's the difference between this and the previous max pooling attempt?

In the previous max pooling attempt, we merely down-sampled a single 2nd dimension, which contains the number of characters. From a matrix of: (num of sentences X 500 X 100) it becomes: (num of sentences X 125 X 100) which is still a 3d matrix.

But in global max pooling, we perform pooling operation across several dimensions(2nd and 3rd dimension) into a single dimension. So it outputs a: (num of sentences X 120) 2D matrix.

In \[15\]:

x \= GlobalMaxPool1D()(x)

Now that we have a 2D matrix, it's convenient to plug it into the densely connected layer, followed by a relu activation function.

In \[16\]:

x \= Dense(50, activation\="relu")(x)

We'll pass it through a dropout layer and a densely connected layer that eventually passes to a sigmoid function.

In \[17\]:

x \= Dropout(0.2)(x)
x \= Dense(6, activation\="sigmoid")(x)

You could experiment with the dropout rate and size of the dense connected layer to see it could decrease overfitting.

Finally, we move on to train the model with 6 epochs and the results seems pretty decent. The training loss decreases steadily along with validation loss until at the 5th or 6th epoch where traces of overfitting starts to emerge.

In \[18\]:

model \= Model(inputs\=inp, outputs\=x)
model.compile(loss\='binary\_crossentropy',
                  optimizer\='adam',
                 metrics\=\['accuracy'\])

In \[19\]:

model.summary()

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
Layer (type)                 Output Shape              Param #   
=================================================================
input\_1 (InputLayer)         (None, 500)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
embedding\_1 (Embedding)      (None, 500, 240)          545280    
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
conv1d\_1 (Conv1D)            (None, 500, 100)          96100     
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
max\_pooling1d\_1 (MaxPooling1 (None, 125, 100)          0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
bidirectional\_1 (Bidirection (None, 125, 120)          57960     
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
global\_max\_pooling1d\_1 (Glob (None, 120)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_1 (Dense)              (None, 50)                6050      
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_1 (Dropout)          (None, 50)                0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_2 (Dense)              (None, 6)                 306       
=================================================================
Total params: 705,696
Trainable params: 705,696
Non-trainable params: 0
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Due to Kaggle kernel time limit, I have pasted the training output of these 6 epochs.

In \[20\]:

batch\_size \= 32
epochs \= 6
#uncomment below to train in your local machine
#hist = model.fit(X\_t,y\_train, batch\_size=batch\_size, epochs=epochs,validation\_data=(X\_te,y\_test),callbacks=callbacks\_list)

Train on 143613 samples, validate on 15958 samples

Epoch 1/6 143613/143613 \[==============================\] - 2580s 18ms/step - loss: 0.0786 - acc: 0.9763 - val\_loss: 0.0585 - val\_acc: 0.9806

Epoch 2/6 143613/143613 \[==============================\] - 2426s 17ms/step - loss: 0.0582 - acc: 0.9804 - val\_loss: 0.0519 - val\_acc: 0.9816

Epoch 3/6 143613/143613 \[==============================\] - 2471s 17ms/step - loss: 0.0531 - acc: 0.9816 - val\_loss: 0.0489 - val\_acc: 0.9823

Epoch 4/6 143613/143613 \[==============================\] - 2991s 21ms/step - loss: 0.0505 - acc: 0.9821 - val\_loss: 0.0484 - val\_acc: 0.9829

Epoch 5/6 143613/143613 \[==============================\] - 3023s 21ms/step - loss: 0.0487 - acc: 0.9826 - val\_loss: 0.0463 - val\_acc: 0.9829

Epoch 6/6 143613/143613 \[==============================\] - 2961s 21ms/step - loss: 0.0474 - acc: 0.9830 - val\_loss: 0.0463 - val\_acc: 0.9831

**UPDATE**

I have uploaded the saved model in this notebook so that you could even continue the training process. To load the model and do a prediction, you could do this:

In \[21\]:

model \= load\_model('../input/epoch-6-model/model-e6.hdf5')

batch\_size \= 32
y\_submit \= model.predict(X\_sub,batch\_size\=batch\_size,verbose\=1)

153164/153164 \[==============================\] - 408s 3ms/step

Getting the prediction data in a format ready for competition submission:

In \[22\]:

y\_submit\[np.isnan(y\_submit)\]\=0
sample\_submission \= submit\_template
sample\_submission\[\["toxic", "severe\_toxic", "obscene", "threat", "insult", "identity\_hate"\]\] \= y\_submit
sample\_submission.to\_csv('submission.csv', index\=False)

I hope this notebook serves as a good start for beginners who are interested in tackling NLP problems using the CNN angle. There are some ideas which you could use to push the performance further, such as :

1. Tweak CNN parameters such as number of strides, different padding settings, window size.
2. Hyper-parameter tunings
3. Experiment with different architecture layers

Thank you for your time in reading and if you like what I wrote, support me by upvoting my notebook..

With the toxic competition coming to an end in a month, I wish everyone godspeed!
