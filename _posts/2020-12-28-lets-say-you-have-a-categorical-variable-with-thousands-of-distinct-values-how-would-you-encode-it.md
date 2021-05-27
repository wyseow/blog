---
title: "Let's say you have a categorical variable with thousands of distinct values, how would you encode it?"
date: "2020-12-28"
categories: 
  - "machine-learning"
  - "test-your-knowledge"
coverImage: "/post_images/photo-1507842217343-583bb7270b66.jpeg"
---

One-hot encoding is out of the question since a large number of distinct values will result in large dimensionality problems(Curse of Dimensionality) in modeling stage.

**One-hot encoding:** Suitable for catgorical variables where no ordinal relationship exists. Map an unique value to a binary vector that is all 0 values except the index of the encoded label, which is marked with 1.

> Side track: why high dimensionality is an issue:  
> Explanation:  
> 1) If we have more features than observations, we run the risk of overfitting the model; bad out of sample performance.  
> 2) Clustering is harder: every observation appear equidistant(same distance) from each other and this makes the distance metric(eg. euclidean) which quantity similarity, thinks that all observations are equally alike, and hence no meaningful clusters can be formed.

The next choice is label or ordinal encoding depending on whether they have a directional relationship.

**Label encoding**: Map an unique value to a integer  
**Ordinal encoding:** Map an unique value to an integer in a specific order that express directional relationship. Eg.(cold, warm, hot->1,2,3)

An increasingly popualar choice in the deep learning space is to use embeddings.

**Embeddings:** Learn an vector representation of the values where the properties of the vector are learned while training a neural network model. 2nd benefit: The learned vector space can be used to infer the similarity and relationship between categorcial values as close values cluster together in the process of training. 3rd benefit: The vectors can be re-used for future models or applications without retraining.

Do you know other great or better ways of encoding categorical variables with massive number of values? Comment below!
