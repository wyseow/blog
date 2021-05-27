---
title: "Let's say we want to build a model to predict booking prices for a hotel booking company. Between linear regression and random forest regression, which model would perform better and why?"
date: "2020-12-28"
categories: 
  - "machine-learning"
  - "statistics"
  - "test-your-knowledge"
coverImage: "/post_images/187790135.jpg"
---

Before we quickly answer "Random Forest", let's take a step back and put on our structured thinking cap to ask ourselves why and perhaps in real life, companies might take the other choice.

Questions to consider:  
**1) What are the features available for us to model?**  
If the features are mostly categorical, especially with large number of distinct values, then linear regression might not be a good choice because it can't handle cardinality as well as random forest.

**2) What are the underlying statistics of the features?**  
If extreme outliers are present and retained for valid reasons, then linear regression can be affected.

There are assumptions of linear regression that needs to be satisifed:  
1) Linear relationship between indepdent and dependent variables  
2) Residuals exhibits normality(mean of 0, unit variance)  
3) Homoscedasticity: Variance of residuals is constant for all observations.  
4) No multi-colinearity between indepedent variables.

**2) What is the definition of "better"?**

We assume the definition refers to accuracy in prediction. Other definition could be processing speed, etc.

Generally, random forest yields higher accuracy than linear regression but it cannot extrapolate values: unable to predict values beyond the range that it has observed during training time. In our context, this means that it cannot predict a sudden, extremely high increase in booking price that it hasn't seen before.

Solutions:  
1) Extension of random forests: trees are grown where the terminal leaves contain linear regression models (eg.Cubist)  
2) Neural networks  
2) Use linear methods like linear regression

**Business interpretability**

Either models provide means for feature importance, which is helpful for business to understand the behavoir of the models or identify drivers behind the booking price.

What would you choose? Comment below!
