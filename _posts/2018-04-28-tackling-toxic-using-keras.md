---
title: "Tackling Toxic Using Keras"
date: "2018-04-28"
categories: 
  - "deep-learning"
  - "kaggle"
  - "machine-learning"
tags: 
  - "deep-learning"
coverImage: "/post_images/Screen-Shot-2018-04-28-at-10.44.38-PM.png"
---

This is a repost from [my kernel](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras) at Kaggle, which has received several positive responses from the community that it's helpful to them. This is one of my kernels that tackles the interesting [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) at Kaggle, which aims to identify and classify toxic online comments.

<script src="https://c328740.ssl.cf1.rackcdn.com/mathjax/latest/MathJax.js?config=TeX-AMS_HTML" type="text/javascript"></script>

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

<script type="text/x-mathjax-config">MathJax.Hub.Config({ tex2jax: { inlineMath: [ ['$','$'], ["\\(","\\)"] ], displayMath: [ ['$$','$$'], ["\\[","\\]"] ], processEscapes: true, processEnvironments: true }, // Center justify equations in code and markdown cells. Elsewhere // we use CSS to left justify single line equations in code cells. displayAlign: 'center', "HTML-CSS": { styles: {'.MathJax_Display': {"margin": 0}}, linebreaks: { automatic: true } } });</script>

![](/post_images/not_santa_detector_dl_logos.jpg)

**This notebook attempts to tackle this classification problem by using Keras LSTM. While there are many notebook out there that are already tackling using this approach, I feel that there isn't enough explanation to what is going on each step. As someone who has been using vanilla Tensorflow, and recently embraced the wonderful world of Keras, I hope to share with fellow beginners the intuition that I gained from my research and study.**

**Join me as we walk through it.**

We import the standard Keras library

In \[1\]:

import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad\_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

/opt/conda/lib/python3.6/site-packages/h5py/\_\_init\_\_.py:36: FutureWarning: Conversion of the second argument of issubdtype from \`float\` to \`np.floating\` is deprecated. In future, it will be treated as \`np.float64 == np.dtype(float).type\`.
  from .\_conv import register\_converters as \_register\_converters
Using TensorFlow backend.

Loading the train and test files, as usual

In \[2\]:

train \= pd.read\_csv('../input/train.csv')
test \= pd.read\_csv('../input/test.csv')

A sneak peek at the training and testing dataset

In \[3\]:

train.head()

Out\[3\]:

.dataframe tbody tr th:only-of-type { vertical-align: middle; } <div></div> .dataframe tbody tr th { vertical-align: top; } <div></div> .dataframe thead th { text-align: right; }

|  | id | comment\_text | toxic | severe\_toxic | obscene | threat | insult | identity\_hate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0000997932d777bf | Explanation\\nWhy the edits made under my usern... | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 000103f0d9cfb60f | D'aww! He matches this background colour I'm s... | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 000113f07ec002fd | Hey man, I'm really not trying to edit war. It... | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0001b41b1c6bb37e | "\\nMore\\nI can't make any real suggestions on ... | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0001d958c54c6e35 | You, sir, are my hero. Any chance you remember... | 0 | 0 | 0 | 0 | 0 | 0 |

A common preprocessing step is to check for nulls, and fill the null values with something before proceeding to the next steps. If you leave the null values intact, it will trip you up at the modelling stage later

In \[4\]:

train.isnull().any(),test.isnull().any()

Out\[4\]:

(id               False
 comment\_text     False
 toxic            False
 severe\_toxic     False
 obscene          False
 threat           False
 insult           False
 identity\_hate    False
 dtype: bool, id              False
 comment\_text    False
 dtype: bool)

Looks like we don't need to deal with the null values after all!

Note that: There are tons of preprocessing and feature engineering steps you could do for the dataset, but our focus today is not about the preprocessing task so what we are doing here is the minimal that could get the rest of the steps work well.

Movng on, as you can see from the sneak peek, the dependent variables are in the training set itself so we need to split them up, into X and Y sets.

In \[5\]:

list\_classes \= \["toxic", "severe\_toxic", "obscene", "threat", "insult", "identity\_hate"\]
y \= train\[list\_classes\].values
list\_sentences\_train \= train\["comment\_text"\]
list\_sentences\_test \= test\["comment\_text"\]

The approach that we are taking is to feed the comments into the LSTM as part of the neural network but we can't just feed the words as it is.

So this is what we are going to do:

1. Tokenization - We need to break down the sentence into unique words. For eg, "I love cats and love dogs" will become \["I","love","cats","and","dogs"\]
2. Indexing - We put the words in a dictionary-like structure and give them an index each For eg, {1:"I",2:"love",3:"cats",4:"and",5:"dogs"}
3. Index Representation- We could represent the sequence of words in the comments in the form of index, and feed this chain of index into our LSTM. For eg, \[1,2,3,4,2,5\]

Fortunately, Keras has made our lives so much easier. If you are using the vanilla Tensorflow, you probably need to implement your own dictionary structure and handle the indexing yourself. In Keras, all the above steps can be done in 4 lines of code. Note that we have to define the number of unique words in our dictionary when tokenizing the sentences.

In \[6\]:

max\_features \= 20000
tokenizer \= Tokenizer(num\_words\=max\_features)
tokenizer.fit\_on\_texts(list(list\_sentences\_train))
list\_tokenized\_train \= tokenizer.texts\_to\_sequences(list\_sentences\_train)
list\_tokenized\_test \= tokenizer.texts\_to\_sequences(list\_sentences\_test)

You could even look up the occurrence and the index of each words in the dictionary:

In \[7\]:

#commented it due to long output
#for occurence of words
#tokenizer.word\_counts
#for index of words
#tokenizer.word\_index

Now, if you look at "list\_tokenized\_train", you will see that Keras has turned our words into index representation for us

In \[8\]:

list\_tokenized\_train\[:1\]

Out\[8\]:

\[\[688,
  75,
  1,
  126,
  130,
  177,
  29,
  672,
  4511,
  12052,
  1116,
  86,
  331,
  51,
  2278,
  11448,
  50,
  6864,
  15,
  60,
  2756,
  148,
  7,
  2937,
  34,
  117,
  1221,
  15190,
  2825,
  4,
  45,
  59,
  244,
  1,
  365,
  31,
  1,
  38,
  27,
  143,
  73,
  3462,
  89,
  3085,
  4583,
  2273,
  985\]\]

But there's still 1 problem! What if some comments are terribly long, while some are just 1 word? Wouldn't our indexed-sentence look like this:

Comment #1: \[8,9,3,7,3,6,3,6,3,6,2,3,4,9\]

Comment #2: \[1,2\]

And we have to feed a stream of data that has a consistent length(fixed number of features) isn't it?

And this is why we use "padding"! We could make the shorter sentences as long as the others by filling the shortfall by zeros.But on the other hand, we also have to trim the longer ones to the same length(maxlen) as the short ones. In this case, we have set the max length to be 200.

In \[9\]:

maxlen \= 200
X\_t \= pad\_sequences(list\_tokenized\_train, maxlen\=maxlen)
X\_te \= pad\_sequences(list\_tokenized\_test, maxlen\=maxlen)

How do you know what is the best "maxlen" to set? If you put it too short, you might lose some useful feature that could cost you some accuracy points down the path.If you put it too long, your LSTM cell will have to be larger to store the possible values or states.

One of the ways to go about it is to see the distribution of the number of words in sentences.

In \[10\]:

totalNumWords \= \[len(one\_comment) for one\_comment in list\_tokenized\_train\]

In \[11\]:

plt.hist(totalNumWords,bins \= np.arange(0,410,10))#\[0,50,100,150,200,250,300,350,400\])#,450,500,550,600,650,700,750,800,850,900\])
plt.show()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYcAAAD8CAYAAACcjGjIAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBo
dHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAEw9JREFUeJzt3X+sXOWd3/H3Z82PjTa7BYJBCJua
RJYaNmod1iVIVKs0acFAVROJSESrxYqQvEpBStStGrMrlTRZKqdSkhYpy4psXEybjUPzQ1jBWdYi
VNFKG8AkDuCwrG+JGxxb2KkJYRUpKcm3f8xzk5Gf8f3pO3PNfb+k0Zz5nnNmvvPY1x8/55yZm6pC
kqRhvzbpBiRJy4/hIEnqGA6SpI7hIEnqGA6SpI7hIEnqGA6SpI7hIEnqGA6SpM5Zk25goS688MJa
t27dpNuQpDPKU0899cOqWj3bdmdsOKxbt459+/ZNug1JOqMk+T9z2c7DSpKkjuEgSeoYDpKkjuEg
SeoYDpKkjuEgSeoYDpKkjuEgSeoYDpKkzhn7CemltG7bw6dcd2j7jWPsRJImw5mDJKljOEiSOoaD
JKljOEiSOoaDJKljOEiSOoaDJKljOEiSOoaDJKljOEiSOoaDJKkzazgkWZvksSTPJTmQ5IOt/pEk
P0iyv91uGNrnziRTSZ5Pct1QfVOrTSXZNlS/PMnjSQ4m+UKSc073G5Ukzd1cZg6vAX9YVW8FrgZu
T3JFW/epqtrQbnsA2rpbgN8GNgF/mmRVklXAp4HrgSuA9w09z8fbc60HXgZuO03vT5K0ALOGQ1Ud
rapvteVXgeeAS2fYZTOwq6p+WlXfA6aAq9ptqqpeqKqfAbuAzUkCvAv4Ytt/J3DTQt+QJGnx5nXO
Ick64O3A4610R5Knk+xIcn6rXQq8OLTb4VY7Vf1NwI+q6rWT6pKkCZlzOCR5I/Al4ENV9WPgXuAt
wAbgKPCJ6U1H7F4LqI/qYWuSfUn2HT9+fK6tS5LmaU7hkORsBsHwuar6MkBVvVRVP6+qXwCfYXDY
CAb/8187tPsa4MgM9R8C5yU566R6p6ruq6qNVbVx9erVc2ldkrQAc7laKcBngeeq6pND9UuGNnsP
8Gxb3g3ckuTcJJcD64EngCeB9e3KpHMYnLTeXVUFPAbc3PbfAjy0uLclSVqMufya0GuA3weeSbK/
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
B0lS5/8DL/b9JFWJ7woAAAAASUVORK5CYII=
)

As we can see, most of the sentence length is about 30+. We could set the "maxlen" to about 50, but I'm being paranoid so I have set to 200. Then again, it sounds like something you could experiment and see what is the magic number.

**Finally the start of building our model!**

This is the architecture of the model we are trying to build. It's always to good idea to list out the dimensions of each layer in the model to think visually and help you to debug later on. ![](/post_images/txJomEa.png)

As mentioned earlier, the inputs into our networks are our list of encoded sentences. We begin our defining an Input layer that accepts a list of sentences that has a dimension of 200. ![](/post_images/uSjU4J7.png) By indicating an empty space after comma, we are telling Keras to infer the number automatically.

In \[12\]:

inp \= Input(shape\=(maxlen, )) #maxlen=200 as defined earlier

Next, we pass it to our Embedding layer, where we project the words to a defined vector space depending on the distance of the surrounding words in a sentence. Embedding allows us to reduce model size and most importantly the huge dimensions we have to deal with, in the case of using one-hot encoding to represent the words in our sentence. ![](/post_images/embedding-custom-projection.png) The output of the Embedding layer is just a list of the coordinates of the words in this vector space. For eg. (-81.012) for "cat" and (-80.012) for "dog". We could also use the distance of these coordinates to detect relevance and context. Embedding is a pretty deep topic, and if you are interested, this is a comprehensive guide: [https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

We need to define the size of the "vector space" we have mentioned above, and the number of unique words(max\_features) we are using. Again, the embedding size is a parameter that you can tune and experiment.

In \[13\]:

embed\_size \= 128
x \= Embedding(max\_features, embed\_size)(inp)

The embedding layer outputs a 3-D tensor of (None, 200, 128). Which is an array of sentence(None means that it's size is inferred), and for each words(200), there is an array of 128 coordinates in the vector space of embedding.

Next, we feed this Tensor into the LSTM layer. We set the LSTM to produce an output that has a dimension of 60 and want it to return the whole unrolled sequence of results. As you probably know, LSTM or RNN works by recursively feeding the output of a previous network into the input of the current network, and you would take the final output after X number of recursion. But depending on use cases, you might want to take the unrolled, or the outputs of each recursion as the result to pass to the next layer. And this is the case.

![](/post_images/RNN-unrolled.png)

From the above picture, the unrolled LSTM would give us a set of h0,h1,h2 until the last h.

From the short line of code that defines the LSTM layer, it's easy to miss the required input dimensions. LSTM takes in a tensor of \[Batch Size, Time Steps, Number of Inputs\]. Batch size is the number of samples in a batch, time steps is the number of recursion it runs for each input, or it could be pictured as the number of "A"s in the above picture. Lastly, number of inputs is the number of variables(number of words in each sentence in our case) you pass into LSTM as pictured in "x" above.

We can make use of the output from the previous embedding layer which outputs a 3-D tensor of (None, 200, 128) into the LSTM layer. What it does is going through the samples, recursively run the LSTM model for 200 times, passing in the coordinates of the words each time. And because we want the unrolled version, we will receive a Tensor shape of (None, 200, 60), where 60 is the output dimension we have defined.

In \[14\]:

x \= LSTM(60, return\_sequences\=True,name\='lstm\_layer')(x)

Before we could pass the output to a normal layer, we need to reshape the 3D tensor into a 2D one. We reshape carefully to avoid throwing away data that is important to us, and ideally we want the resulting data to be a good representative of the original data.

Therefore, we use a Global Max Pooling layer which is traditionally used in CNN problems to reduce the dimensionality of image data. In simple terms, we go through each patch of data, and we take the maximum values of each patch. These collection of maximum values will be a new set of down-sized data we can use.

As you can see from other Kaggle kernels, different variants (Average,Max,etc) of pooling layers are used for dimensionality reduction and they could yield different results so do try them out.

If you are interested in finding out the technical details of pooling, read up here: [https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/](https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/)

In \[15\]:

x \= GlobalMaxPool1D()(x)

With a 2D Tensor in our hands, we pass it to a Dropout layer which indiscriminately "disable" some nodes so that the nodes in the next layer is forced to handle the representation of the missing data and the whole network could result in better generalization. ![](/post_images/dropout.jpeg)

We set the dropout layer to drop out 10%(0.1) of the nodes.

In \[16\]:

x \= Dropout(0.1)(x)

After a drop out layer, we connect the output of drop out layer to a densely connected layer and the output passes through a RELU function. In short, this is what it does:

**Activation( (Input X Weights) + Bias)**

all in 1 line, with the weights, bias and activation layer all set up for you! We have defined the Dense layer to produce a output dimension of 50.

In \[17\]:

x \= Dense(50, activation\="relu")(x)

We feed the output into a Dropout layer again.

In \[18\]:

x \= Dropout(0.1)(x)

Finally, we feed the output into a Sigmoid layer. The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.

In \[19\]:

x \= Dense(6, activation\="sigmoid")(x)

We are almost done! All is left is to define the inputs, outputs and configure the learning process. We have set our model to optimize our loss function using Adam optimizer, define the loss function to be "binary\_crossentropy" since we are tackling a binary classification. In case you are looking for the learning rate, the default is set at 0.001.

In \[20\]:

model \= Model(inputs\=inp, outputs\=x)
model.compile(loss\='binary\_crossentropy',
                  optimizer\='adam',
                  metrics\=\['accuracy'\])

The moment that we have been waiting for as arrived! It's finally time to put our model to the test. We'll feed in a list of 32 padded, indexed sentence for each batch and split 10% of the data as a validation set. This validation set will be used to assess whether the model has overfitted, for each batch. The model will also run for 2 epochs. These are some of the tunable parameters that you can experiment with, to see if you can push the accurate to the next level without crashing your machine(hence the batch size).

In \[21\]:

batch\_size \= 32
epochs \= 2
model.fit(X\_t,y, batch\_size\=batch\_size, epochs\=epochs, validation\_split\=0.1)

Train on 143613 samples, validate on 15958 samples
Epoch 1/2
143613/143613 \[==============================\] - 1594s 11ms/step - loss: 0.0722 - acc: 0.9770 - val\_loss: 0.0494 - val\_acc: 0.9818
Epoch 2/2
143613/143613 \[==============================\] - 1581s 11ms/step - loss: 0.0452 - acc: 0.9832 - val\_loss: 0.0464 - val\_acc: 0.9830

Out\[21\]:

<keras.callbacks.History at 0x7f8113df9a58>

Seems that the accuracy is pretty decent for a basic attempt! There's a lot that you could do (see TODO below) to further improve the accuracy so feel free to fork the kernel and experiment for yourself!

**Additional tips and tricks**

1) If you have hit some roadblocks, especially when it starts returning dimension related errors, a good idea is to run "model.summary()" because it lists out all your layer outputs, which is pretty useful for diagnosis.

In \[22\]:

model.summary()

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
Layer (type)                 Output Shape              Param #   
=================================================================
input\_1 (InputLayer)         (None, 200)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
embedding\_1 (Embedding)      (None, 200, 128)          2560000   
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
lstm\_layer (LSTM)            (None, 200, 60)           45360     
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
global\_max\_pooling1d\_1 (Glob (None, 60)                0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_1 (Dropout)          (None, 60)                0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_1 (Dense)              (None, 50)                3050      
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_2 (Dropout)          (None, 50)                0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_2 (Dense)              (None, 6)                 306       
=================================================================
Total params: 2,608,716
Trainable params: 2,608,716
Non-trainable params: 0
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

2) While adding more layers, and doing more fancy transformations, it's a good idea to check if the outputs are performing as you have expected. You can reveal the output of a particular layer by :

In \[23\]:

from keras import backend as K

\# with a Sequential model
get\_3rd\_layer\_output \= K.function(\[model.layers\[0\].input\],
                                  \[model.layers\[2\].output\])
layer\_output \= get\_3rd\_layer\_output(\[X\_t\[:1\]\])\[0\]
layer\_output.shape
#print layer\_output to see the actual data

Out\[23\]:

(1, 200, 60)

In \[ \]:

Personally I find Keras cuts down a lot of time and saves you the agony of dealing with grunt work of defining the right dimensions for matrices. The time saved could have spent on fruitful tasks like experimenting with different variations of model, etc. However, I find that many variables and processes have been initialized automatically in a way that beginners to deep learning might not realize what is going on under the hood. There's a lot of intricate details so I encourage newbies to open up this black box and you will be rewarded with a wealth of knowledge in deep learning.

I hope someone will find this short guide useful. If you like to see more of such guides, support me by upvoting this kernel. Thanks for reading and best of luck for the competition!

**TODO:**

1. Using Pre-trained models to boost accuracy and take advantage of existing efforts
2. Hyper-parameter tuning of bells and whistles
3. Introduce early stopping during training of model
4. Experiment with different architecture.
