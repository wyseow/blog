---
title: "Let's say we have 1 million app rider journey trips. We want to build a model to predict ETA after a rider makes a ride request..."
date: "2020-12-29"
categories: 
  - "data-science"
  - "machine-learning"
  - "test-your-knowledge"
coverImage: "/post_images/akrales_160314_0978_A_0165.0.0.png.jpeg"
---

..how would we know if we have enough data to create an accurate enough model?

Questions to consider:  
1) **What is the definition of "accurate enough"?** How much error is acceptable?  
Before thinking about modelling metrics, from a business perspective, we might have an idea of the level of accuracy we need to hit a product goal, and it might not be reasonable from an implementation perspective. For example, if goal is to have an accuracy of RMSE of 1-min and in reality the data is sparse(0 for most data points), oddly distributed(only have data for certain groups) or noisy(values stored inaccurately) then it might be diffcult to achieve the goal.

2) **Are there any existing, simple models or hieristics driven rule-based systems that could serve as baseline?**  
If yes, we could observe the relative improvement by the new model, and determine if that is accurate enough for business users.

Other possible solutions:

**Learning curves:** Use Learning curves to observe accuracy when training data is progressively increased. If we fit our model on 20%..50%..80% of our data size and then cross-validate to determine model accuracy, we can then determine how much more data we need to achieve a certain accuracy level.

For example. If we reach 75% accuracy with 500K datapoints but then only 77% accuracy with 1 million datapoints, then adding more data will only yield marginal results. Also, we’ll realize that our model is not predicting well enough with its existing features since doubling the training data size did not significantly increase the accuracy rate. This would inform us that we need to re-evaluate our features rather than collect more data.

**Cross validation (CV):** We could use cross validation to see how well the model would perform in pratice, and generalize to unseen data.

CV partitions a sample of data into complementary subsets, performing the modelling on one subset(training set), and validating the model on the other subset (validation or testing set). Multiple rounds of cross-validation are performed using different partitions, and the validation results are averaged over the rounds to give an estimate of the model's predictive performance.

**Statistical approach**

We could also use the Hoeffding Inequality to estimate sample size given confidence level.

[Here](https://malishoaib.wordpress.com/2017/09/08/sample-size-estimation-for-machine-learning-models-using-hoeffdings-inequality/)'s a good post on more information on Hoeffding Inequality
