---
title: "Reality sucks - dealing with imbalanced data"
date: "2017-06-21"
categories: 
  - "data-preprocessing"
  - "machine-learning"
coverImage: "/post_images/4f0d80d0b83a3b652c8cc0b99570c3c5.jpg"
---

You stumble upon some intriguing patient cancer dataset that seems to be the last remaining puzzle towards solving the human war against cancer that will make this world a better place for everyone and you excitedly download the dataset.

Your data analysis usually go through these standard processes: 1) Load data 2) Do some pre-processing of data (cleaning, converting variables to categorical variables, etc) 4) Use visualisation library like pyplot (Okay, maybe more advanced stuff like Seaborn) to discover early insights and low hanging fruits 3) Load a machine learning model from scikit-learn library 4) Fit the model with your prepared data 5) Finally, predict the y values of your test x values. 6) Use some metrics to assess model accuracy

**_Voila!_** You have just generated your gold standard prediction in a proven pipeline that could apply to all dataset in Kaggle!

**BUT wait!** You realise that all, if not most of the predicted values return the same value! And upon deeper look, you might discover that most of your sample data belong to one, same class!

This is a good example of imbalanced data, and the fact is that not every dataset that comes to you will be nicely distributed, labelled and all that stuff. Only if real world dataset is like Kaggle dataset huh?

This is when rubber hits the road and that 6-figure data science job doesn't sound so easy anymore. But hey, this is valuable skills to add to your resume and if you set to be a data scientist, you are bound to hit this problem so let’s see how do we tackle this nasty issue.

Here’s a very simplified example to bring the problem across. Consider this dataset: \[Product\]\[Cost\]\[Category\] iPad 340 1 iPhone 200 2 iMac 1900 3 iPhone6 250 2 iPhone7 390 2 iPhone8 490 2

When you train the model using the dataset above to predict the category of the product given the product name and cost, you are as good as teaching the model to recognise the features that it needs to predict category value of 2 because they are the “majority class” - samples from a class that represent most of the data.

There’s simply not enough samples from the other classes to teach the model what it's like to predict other things than category value of 2. In this case, the other classes which only has a few samples(relatively) are called “minority classes”

Ideally, you should hit a good balanced, equal distribution of all your classes to ensure that the model can be properly trained for all classes and you should be able to spot this problem in your “visualisation step” where you inspect the data’s distribution using histogram and stuff like that.

\[caption id="attachment\_74" align="alignnone" width="626"\]![](/post_images/seaborn-countplot-1-2.png) Look at that mountain of imbalanced data!\[/caption\]

So how do you go about solving this problem of imbalanced data?

Fortunately, all is not lost - there are 3 ways to go about it.

**1) Resample differently**

If you have a large dataset to play with, do an under-sample. Under-sample means removing the additional majority class samples to make the whole dataset balanced again.

For example, reduce the samples that belong to category 2. After under-sampling: \[Product\]\[Cost\]\[Category\] iPad 340 1 iPhone 200 2 iMac 1900 3

I know this might look strange because there are only 3 samples left! So that’s why I said you really need to have a large dataset to even consider under-sampling.

Technically, this is one of the easiest way to go about it. Choose a X samples from each class and to ensure that the distribution of each class is equal:

\[python\] #Number of samples per class num\_choose = 100

#Get 100 samples from class 1 indices\_1 = trainDF\[trainDF.Score == 1\].index random\_indices = np.random.choice(indices\_1, num\_choose, replace=False) sample\_1 = trainDF.loc\[random\_indices\]

#Get 100 samples from class 2 indices\_2 = trainDF\[trainDF.Score == 2\].index random\_indices = np.random.choice(indices\_2, num\_choose, replace=False) sample\_2 = trainDF.loc\[random\_indices\]

dataset = pd.concat(\[sample\_1,sample\_2\]) \[/python\]

On the other hand, if you don’t have a large dataset to play with, do an over-sample. Over-sample means increasing the number of minority class samples to make the whole dataset balanced again.

For example. reduce the samples that belong to category 1 and 3. After: \[Product\]\[Cost\]\[Category\] iPad 340 1 iPhone 200 2 iMac 1900 3 iPhone6 250 2 iPhone7 390 2 iPhone8 490 2 iMac2 1800 3 <-synthetically created samples iMac3 2000 3 <-synthetically created samples iPad2 370 1 <-synthetically created samples iPad3 390 1 <-synthetically created samples

That’s kind of neat huh? A good demonstration of “fake it till you make it”.

There are many techniques and methods to create artificial samples and the most easy way is to use SMOTE(Synthetic Minority Over-sampling Technique). It’s currently implemented by this Python library over [here](https://github.com/scikit-learn-contrib/imbalanced-learn) and you just need to import the python module to start using it. Here's an example:

\[python\] from collections import Counter from sklearn.datasets import make\_classification from imblearn.over\_sampling import SMOTE import pandas as pd import numpy as np

#Create an imbalanced dataset X, y = make\_classification(n\_classes=2, class\_sep=2,weights=\[0.1, 0.9\], n\_informative=3, n\_redundant=1, flip\_y=0,n\_features=10, n\_clusters\_per\_class=1, n\_samples=100, random\_state=10)

print('Original dataset shape {}'.format(Counter(y))) #prints Original dataset shape Counter({1: 90, 0: 10})

combined = pd.DataFrame(np.column\_stack(\[X,y\])) print(combined.shape) #prints (100, 11)

sm = SMOTE(random\_state=42) X\_res, y\_res = sm.fit\_sample(X, y)

print('Resampled dataset shape {}'.format(Counter(y\_res))) #prints Resampled dataset shape Counter({1: 90, 0: 90}) #We can see that SMOTE has created the lacking 80 samples of the minority class!

combined2 = pd.DataFrame(np.column\_stack(\[X\_res,y\_res\])) print(combined2.shape) #prints (180, 11)

\[/python\]

**2) Accept the imbalance, and define accuracy as sensitivity and specificity rather than a complete correct VS wrong prediction.**

Perhaps you could use a ROC(Receiver Operating Characteristic) curve to see the rate of true positive prediction over the rate of false positive prediction based on the different threshold, and accept the fact that the model will perform at an accuracy of X% with the probability of Y% of them a wrong prediction.

Consider this ROC curve:![](/post_images/sphx_glr_plot_roc_002.png) It represents the trade-off(price to pay) of positive accuracy for negative accuracy. The large the area of the curve, the more accurate it is.

I’ll talk about this-in depth-in a later post. There’s much more details to it.

**3) Use weights in models to offset the imbalance.**

You can use the "**class\_weight"** parameter in many scikit-learn models to give additional weight to the minority classes during training and predicting. This is an example of using class\_weight in SVM in an attempt to  have the model place more emphasis on minority class:

\[python\] # set the class "1" to have weight of 10 wclf = svm.SVC(kernel='linear', class\_weight={1: 10}) wclf.fit(X, y) \[/python\]

Visualisation: ![](/post_images/sphx_glr_plot_separating_hyperplane_unbalanced_001.png)

These are some of the ways that I have personally used, and there are tons of other sophisticated ways to tackle this issue. If you have a good idea, feel free to share it with us below.

Thanks for reading and see you at the next post!
