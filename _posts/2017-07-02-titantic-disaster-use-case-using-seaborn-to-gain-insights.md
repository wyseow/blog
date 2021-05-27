---
title: "Titantic Disaster Use Case: Using Seaborn to gain insights"
date: "2017-07-02"
categories: 
  - "data-preprocessing"
  - "data-visualization"
coverImage: "/post_images/cover_large-1.jpg"
---

There are tons of Python-based visualisation tools out there but my favourite one has to be [Seaborn](http://seaborn.pydata.org). Some would say using Seaborn is a form of cheating. Well, after all Seaborn is just a wrapper of matplotlib and instead of saying Seaborn VS matplotlib, we should look at it as a upgraded, flashy version of the old trusty matplotlib library. The part I like about Seaborn is that it comes with a ready set of color palettes that not only makes your data visualisation looks tasty, it also shouts out professionalism in just a liner or two.

\[caption id="attachment\_103" align="alignnone" width="497"\]![](/post_images/f1-score-graph.png) An example colourful barplot o\_\_O\[/caption\]

We'll come to the colouring of the visualisation in another post as it has a lot of depth to it, and besides, that's not what we are here today for. We are here for, Titanic!

Today, we are going use the Seaborn library to help us discover some interesting insights of the most popular dataset from Kaggle, the [Titanic Disaster](titanic: Machine Learning from Disaster | Kaggle) dataset. We are going to find out what passenger profile leads to a higher survival rate. Are you likely to survive because you are a female (where's the gender equality in that?!). Or perhaps being a rich passenger could buy you a ticket to the lifeboat? Join me as we find out.

Let's start things off with importing our pandas library, and on top of that, we also import the Seaborn library:

\[python\] import pandas as pd import seaborn as sns

from sklearn import preprocessing import matplotlib.pyplot as plt

#this allows plots to appear directly in the notebook %matplotlib inline \[/python\]

Next, we'll import the Titanic dataset and my personal practice is to always use the head function and shape attributes after importing to take a look at a small sample of the dataset:

\[python\] #import the titanic dataset trainDF = pd.read\_csv('train.csv')

print(trainDF.head()) print(trainDF.shape) \[/python\]

![](/post_images/Screen-Shot-2017-07-03-at-11.29.54-PM-1024x510.png)

Let's use the describe function to see some basic stats of each features:

\[python\] print(trainDF.describe().transpose()) \[/python\]

![](/post_images/Screen-Shot-2017-07-03-at-11.33.15-PM-1024x398.png)

It reveals some of the interesting facts.

For example, we can see the wide variety of the age of the passengers(as young as 0.42 to as old as 80 years old). The fare goes as low as 0 to as high as 512! Wow, some people actually the ship for FREE?!

There might be more fine details hidden within it but without good visualization its easy to miss them.

Next, we'll do some quick pre-processing to the data to get it ready for visualisation. We would not be explaining much as today's post focuses on the visualization aspects. I will be touching on this pre-processing part in another blog post, so stay tuned! I have put some comments on the code, but feel free to ask away if you have questions!

\[python\] #Main goal of this data preprocessing is to clean up the data, and check for any NA values. If there's any, fill in the blanks.

#We use .isnull() to see which variables have empty data. print(trainDF.isnull().any()) #Can see that Age, Cabin and Embarked have null values.

#Let's work on Age first. #print(trainDF\['Age'\].mean()) print(trainDF\['Age'\].std()) #fill empty age by getting the mean of the age group trainDF\['Age'\].fillna(trainDF.groupby("Sex")\["Age"\].transform("mean"),inplace=True)

#Forget about Cabin as it's just a random Cabin ID #trainDF = trainDF.drop('Cabin', 1)

#Let's work on Embarked as place of Embarked might affect survival rates #Fill up the NA values with the highest embarkment spot print(trainDF\["Embarked"\].value\_counts()) #S seems to be the most frequently one trainDF\['Embarked'\].fillna("S",inplace=True)

#check for null values again, should not have any now.. print(trainDF.isnull().any())

#Replace the categorical values with integers to prepare them #for analysis and model fitting later one lb = preprocessing.LabelEncoder()

#converts male to 1 and female to 0 trainDF\['Sex'\] = lb.fit\_transform(trainDF\['Sex'\]) #alternatively, we can do this as well: dataset\['Sex'\] = dataset\['Sex'\].map( {'female': 1, 'male': 0} ).astype(int)

#converts Embarked info to numeric as well trainDF\['Embarked'\] = lb.fit\_transform(trainDF\['Embarked'\]) #trainDF\['Embarked'\] = lb.fit\_transform(trainDF\['Embarked'\]) #print(trainDF.iloc\[:10,:\])

#okay, all set! data pre-processing all done! print(trainDF.head()) \[/python\]

![](/post_images/Screen-Shot-2017-07-03-at-11.38.53-PM-1024x479.png)

OKAY! Finally here comes the juicy part!

The whole goal of exploring the Titanic dataset is to discover some interesting facts and derive some early insights using the Seaborn package without even getting to the machine learning stage.

First, let's find out what are the contributing factors of survival. I have a few hypothesis. Eg. Would male passenger be better suvivors? Perhaps men are stronger and be more resilient towards disater? Let's see...

We could use panda's corr() function to gain a big picture of what are the few features that we should focus our energy on, by analysing at the bivariate corelation between the features. The core() function generates a matrix, and we'll pass the matrix to Seaborn's heatmap (one of my favourite tool) function for visualization:

\[python\] corr = trainDF.corr() plt.figure(figsize=(14,14)) sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='viridis') \[/python\]

\[caption id="attachment\_118" align="alignnone" width="678"\]![](/post_images/Screen-Shot-2017-07-03-at-11.44.33-PM-993x1024.png) Look at this colourful piece of art...\[/caption\]

There's a lot to see here. But I mainly use it to find out which features matters to the dependent variable(survived).

The heatmap is pretty helpful when you need to present the stats to someone not as geeky as us, as it's easy to see that the darker it is, the stronger are the correlation.

First off, we can see that **Fare** has a positive corelation, and Sex and PClass has a negative corelation with Survival rate.

As Fare rate increase(by 0.26), Survival rate increase(by 1) Does that means that if you paid a higher fare, your survival rate would increase? We'll find out soon enough.

Secondly, as Passenger class(**PClass**) decrease(by -0.34), Survival rate increase(by 1). By definition, High class = 1, Low class= 3 Does that means that if you are a "better" class passenger, your survival rate increase as well? We'll find out soon enough as well.

As **Sex** decrease(by -0.5), Survival rate increase(by 1). By definition, Sex 0=Female, 1=Male The highest corelation! It seems that if you are female, your survival rate increase! Okay, enough of these stirring of mysteries! Let's get our hands dirty and delve in these 3 features to find out more.

Before delving in the bivariate correlation between the suspicious independent and the dependent variable, I would like to do a thorough inspection of each of the variables by doing a univariate analysis. That includes checking out the distribution of the variables, and to know more about their characteristics:

\[python\] fig, axs = plt.subplots(8, 4, figsize = (15,12)) colorToUse = "green"

ax1 = plt.subplot2grid((8, 4), (0, 0), rowspan=2, colspan=2) ax2 = plt.subplot2grid((8, 4), (0, 2), rowspan=2, colspan=2) ax3 = plt.subplot2grid((8, 4), (2, 0), rowspan=2, colspan=2) ax4 = plt.subplot2grid((8, 4), (2, 2), rowspan=2, colspan=2) ax5 = plt.subplot2grid((8, 4), (4, 0), rowspan=2, colspan=2) ax6 = plt.subplot2grid((8, 4), (4, 2), rowspan=2, colspan=2)

ax7 = plt.subplot2grid((8, 4), (6, 0), rowspan=2, colspan=2) ax8 = plt.subplot2grid((8, 4), (6, 2), rowspan=2, colspan=2)

fig.tight\_layout(pad = 0.4, w\_pad = 3.0, h\_pad = 4.0)

ax1.set\_title("Plot 1: Fare", fontsize =16) ax2.set\_title("Plot 2: Pclass", fontsize =16) ax3.set\_title("Plot 3: Sex", fontsize =16) ax4.set\_title("Plot 4: Survived", fontsize =16) ax5.set\_title("Plot 5: Age", fontsize =16) ax6.set\_title("Plot 6: Embarked", fontsize =16) ax7.set\_title("Plot 5: SibSp(Num of siblings/spouse)", fontsize =16)# of siblings / spouses aboard the Titanic ax8.set\_title("Plot 6: Parch(Parent/children)", fontsize =16)# of parents / children aboard the Titanic

#sns.kdeplot(trainDF.Fare, ax=ax1) #KDE Plot Suitable for plotting densities of continous variable sns.distplot(trainDF\["Fare"\],color = colorToUse,ax=ax1) #distplot more Suitable for plotting densities of categorical variables sns.distplot(trainDF\["Pclass"\],color = colorToUse,ax=ax2) sns.distplot(trainDF\["Sex"\],color = colorToUse,ax=ax3) sns.distplot(trainDF\["Survived"\],color = colorToUse,ax=ax4) #Suitable for plotting continous var sns.kdeplot(trainDF\["Age"\],shade = True,color = colorToUse,ax=ax5) #distplot more Suitable for plotting categorical variables sns.distplot(trainDF\["Embarked"\],color = colorToUse,ax=ax6) #KDE Plot Suitable for plotting densities of continous variable sns.kdeplot(trainDF\["SibSp"\],shade = True,color = colorToUse,ax=ax7) sns.kdeplot(trainDF\["Parch"\],shade = True,color = colorToUse,ax=ax8)

\[/python\]

![](/post_images/Screen-Shot-2017-07-03-at-11.56.49-PM-1024x843.png)

Okay, I know that's a lot to digest but let's go through this one-by-one.

In the first plot, we have used a KDE plot (kdeplot) to illustrate the densities of the Fare variable. And we can see that most passengers paid less than $50 for the ride and there are a few (see the little bumps towards the end?) of them who paid as high as $500 in fare!

In the second plot, we have used a distribution plot (distplot) to illustrate the frequency of the PClass variable. Seaborn's distplot combines density into a histogram diagram and it's helpful when you need to compare the different class for categorical variables.

We can see that most of the passengers comes from the lower class. We know that most don't survived this mishap, but among those survived, what class are they from? is there a way we can look at it in a illustrative way? We will soon find out.

Note that we have label encoded the Sex categorical variable into a 1 or 0. Sex 0=Female, 1=Male. Hence, we can see that there are more male passenger than female passenger.

Unfortunately, we can see that most people don't survive.

When it comes to Age, most passengers are in their 30s+.

Most passenger embarked at the #2 point. I wonder is this correlated to survival rate?

Most passenger don't have siblings or spouse on board. However, most passengers brought 1 parent or kid. I wonder do escaping together as a family actually increases your survival rate?

Seems that we have tons of questions to be answered. But let's just focus on the 3 highest correlated variables: Fare, PClass, and Sex.

We are finally inspecting the bivariate correlation between our focused variables and survived variable! I'll pull out some useful plotting tools in the Seaborn library to show their relationship.

First, let's do a bivariate correlation analysis for PClass against Survived.

\[python\] fig, axs = plt.subplots(2, 6, figsize = (12,4)) colorToUse = "hls"

ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2) ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=2, colspan=2) ax3 = plt.subplot2grid((2, 6), (0, 4), rowspan=2, colspan=2)

fig.tight\_layout(pad = 0.4, w\_pad = 3.0, h\_pad = 4.0) # To understand how this works see point and link 3

ax1.set\_title("Co-relation of fare and survived(Pointplot)", fontsize =12) ax2.set\_title("Co-relation of fare and survived(Barplot)", fontsize =12) ax3.set\_title("Co-relation of fare and survived(ViolinPlot)", fontsize =12)

#For categorical VS numerical corelation, we can use the pointplot to see the corelation in the big picture #due to a bug in pointplot, we have to draw a pointplot just with line :( sns.pointplot(x="Survived", y="Fare", data=trainDF, markers="",join=True, ci=None, color="grey",ax=ax1) sns.pointplot(x="Survived", y="Fare",palette=colorToUse,join=True, data=trainDF,ax=ax1)

#and use the barplot to see amount of co-relation involved sns.barplot(trainDF\['Survived'\],trainDF\["Fare"\],palette = colorToUse,ax=ax2)

#But the problem is that it doesnt tell the mean+density, so we have to use the ViolinPlot to see them! sns.violinplot(trainDF\['Survived'\],trainDF\["Fare"\],palette = colorToUse,ax=ax3)

\[/python\]

![](/post_images/Screen-Shot-2017-07-04-at-8.46.29-PM-1024x367.png) We have used Seaborn's PointPlot in the first plot. Point plots can be more useful than bar plots for focusing comparisons between different levels of one or more categorical variables as it draws a line(slope) between the classes, and our eyes can judge the differences much easily.

Anyway, from the line, we can see that generally, higher fare does leads to higher survival.

In the second plot, we use the good old trusty Bar plot to see the amount of differences between the 2 classes.

But the problem is that it doesnt tell the mean and density, so we have to use the Violin Plot to see them!

A Violin Plot plays a similar role as a box and whisker plot in the sense that it shows the distribution of quantitative data across several classes of categorical variables and also pack in a kernel density estimation of the underlying distribution. It's probably the most all-in-one tool in Seaborn's arsenal!

In short, a big area size represents the high density and small thin area represents the low density. The white dot shows the median and the small vertical black box shows the interquartile range. The line stretches as far as the 95% confidence interval.

Therefore, from the Violin Plot we can see that although higher fare leads to higher survival rates in general, there are only a few people (see the thin line going to the 500 range) in these cases and it's these few people who are "dragging" the mean up. This is something that we probably would have missed if we stopped at Point plots and bar plots.

Moving on, let's do a similar analysis for PClass against Survived.

\[python\] fig, axs = plt.subplots(2, 6, figsize = (12,4)) colorToUse = "hls"

ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2) ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=2, colspan=2) ax3 = plt.subplot2grid((2, 6), (0, 4), rowspan=2, colspan=2)

fig.tight\_layout(pad = 0.4, w\_pad = 3.0, h\_pad = 4.0) # To understand how this works see point and link 3

ax1.set\_title("Co-relation of Pclass and survived(Pointplot)", fontsize =12) ax2.set\_title("Co-relation of Pclass and survived(Barplot)", fontsize =12) ax3.set\_title("Co-relation of Pclass and survived(ViolinPlot)", fontsize =12)

#For categorical VS numerical corelation, we can use the pointplot to see the corelation in the big picture #due to a bug in pointplot, we have to draw a pointplot just with line :( sns.pointplot(x="Survived", y="Pclass", data=trainDF, markers="",join=True, ci=None, color="grey",ax=ax1) sns.pointplot(x="Survived", y="Pclass",palette=colorToUse,join=True, data=trainDF,ax=ax1)

#and use the barplot to see amount of co-relation involved sns.barplot(trainDF\['Survived'\],trainDF\["Pclass"\],palette = colorToUse,ax=ax2)

#But the problem is that it doesnt tell the mean+density, so we have to use the ViolinPlot to see them! sns.violinplot(trainDF\['Survived'\],trainDF\["Pclass"\],palette = colorToUse,ax=ax3) \[/python\]

![](/post_images/Screen-Shot-2017-07-04-at-10.10.55-PM-1024x385.png)

Okay, what do we see? From the first plot(Pointplot), we can see that generally, lower Pclass(actually means high class and expensive tickets) does leads to higher survival.

From the second plot(Barplot), we see the amount of difference between the 2 classes.

Against, the problem is that it doesnt tell the mean and density, so we have to use the ViolinPlot to see them.

So in the third plot(Violinplot), we can see that among the survivors, there's a pretty balanced distributions of Pclass; it's not dominated by just high class passengers, which is very different from what we thought in the 1st plot.

Perhaps we can say that the reason why it looks like most passengers belong to Pclass of 3 didn't survived is because the overall population of Pclass = 3 is much higher than Pclass=1, so-let's say-there's a good chance that when you throw a stone into the population the likelihood of hitting a passenger of PClass 3 is much higher of PClass 1.

Lastly, let's do a similar analysis for PClass against Survived.

\[python\] fig, axs = plt.subplots(2, 6, figsize = (12,4)) colorToUse = "hls"

ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2) ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=2, colspan=2) ax3 = plt.subplot2grid((2, 6), (0, 4), rowspan=2, colspan=2)

fig.tight\_layout(pad = 0.4, w\_pad = 3.0, h\_pad = 4.0)

ax1.set\_title("Co-relation of Sex and survived(Pointplot)", fontsize =12) ax2.set\_title("Co-relation of Sex and survived(Barplot)", fontsize =12) ax3.set\_title("Co-relation of Sex and survived(ViolinPlot)", fontsize =12)

#For categorical VS numerical corelation, we can use the pointplot to see the corelation in the big picture #due to a bug in pointplot, we have to draw a pointplot just with line :( sns.pointplot(x="Survived", y="Sex", data=trainDF, markers="",join=True, ci=None, color="grey",ax=ax1) sns.pointplot(x="Survived", y="Sex",palette=colorToUse,join=True, data=trainDF,ax=ax1)

#and use the barplot to see amount of co-relation involved sns.barplot(trainDF\['Survived'\],trainDF\["Sex"\],palette = colorToUse,ax=ax2)

#countplot is helpful to delve through classes(how-many-among-questions) sns.countplot(trainDF\["Survived"\], hue="Sex", data=trainDF) \[/python\]

![](/post_images/Screen-Shot-2017-07-05-at-11.47.01-PM-1024x377.png) In case you have forgotten, Sex 0=Female,and 1=Male,

From the first plot(Pointplot), we can see that generally, Female(0) are more likely to survive.

From the second plot(Barplot), we could see that although there are more male than female in this ship, a lot of survivors are female.

In third plot, we have used Seaborn's Countplot and it's hue parameter to further separate the Survived into each Sex class. We can easily see that among the survivors, most are female and it's 2 times of male counterpart!

No big shocker for this correlation.

Just to add on a few more Seaborn's incredible suite of plots, we could use Seaborn's Jointplot to plot a graph to see the bivarite relationship between 2 continuous numerical variables.

For instance, let's do a Fare VS Age plot:

\[python\] #5 available parameters for display: “scatter” | “reg” | “resid” | “kde” | “hex” #Specify regression to show regression line so that you can see the pattern better seaplot = sns.jointplot(x="Fare",y="Age",data=trainDF,kind="reg") \[/python\]

![](/post_images/Screen-Shot-2017-07-05-at-11.55.42-PM-300x281.png) As you can see, Jointplot shows the correlation (strength and direction) of the X variable against Y using Pearson's coefficient, which is the same method as the corelation table above. By specifying "reg" for the kind parameter, we are telling the function to fit a regression line. You could try different ones to find out.

Anyway, we can see that there's some small(weak) corelation between Fare and Age, although not much. I love the extra kde plot at the top and right for you to see the density of each variable.

It could also show categorical variables correlation as well but it's not as good to look at. Let's give it a try with the additional parameter "logistic" and use a little "jitter". Using back the same Fare VS Survived plot as above:

\[python\] seaplot = sns.jointplot(x="Fare",y="Survived",data=trainDF,kind="reg",logistic=True,y\_jitter=0.1) \[/python\]

![](/post_images/Screen-Shot-2017-07-05-at-11.58.31-PM-300x280.png) Again, just like the previous plot above, it shows that there's a good corelation that as fare increases, survival rate increases as well. If you notice closely, you can see that there's a couple of outliers that gives weight to the "high fare" survivors.

That's pretty much all for today! There's a lot to learn and nothing beats getting your hands dirty. Do try out exploring your favourite dataset using the different Seaborn plots, and let me know if you stumble upon something useful or cool.

The important take away is that by using different kinds of plots, it shows the corelation of variables in a different light, exposing potential insights and leads to even more discovery! Very often, you might realise that we don't need any sophisticated tools or machine learning to uncover the next course of action.
