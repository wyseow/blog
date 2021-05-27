---
title: "Transfer learning and beyond"
date: "2019-05-30"
categories: 
  - "deep-learning"
  - "nlp"
tags: 
  - "bert"
  - "gpt"
  - "nlp"
  - "transfer-learning"
  - "transformer"
coverImage: "/post_images/Transfer_Learning-512.png"
---

Transfer learning has proven to be useful in NLP in the recent years. As many called the "Imagenet moment" when the likes of large pretrained language models such as BERT, GPT, GPT2 have sprung out from the big research labs, they have been extended in various methods to achieve further state of the art results on a wide range of NLP tasks.

The main idea of transfer learning is fairly simple; one would first pre-train a model to let it learn a common vector space representation of text so that it will be in an optimally intitalized state regardless whatever the downstream NLP task is, and then finally a fine-tuning stage that enhance its performance on a particular NLP task that we are concern with.

When it comes to pre-training, there are 2 popular approaches: language model(LM) pre-training and multi-task learning (MTL). While LM pre-training has attracted most of the attention in the area of transfer learning, there is a growing interest in applying MTL.

MTL is useful for multiple related tasks to be learned jointly so that the knowledge learned in one task can benefit other tasks. In contrast to LM pre-training, MTL requires labeled data trained in a supervised learning setting but it provides an effective way of leveraging supervised data from many related tasks. Most importantly, the use of multi-task learning profits from a regularization effect via alleviating overfitting to a specific task, thus making the learned representations universal across tasks.

### **State of the art with MTL**

There's a recent model, Multi-Task Deep Neural Network (MT-DNN) 1 that combines these 2 approaches together and achieve state of the art results on the GLUE benchmark. It incorporate representations from BERT at the lower layer which is shared across all tasks, while the top task-specific layers are designed for different NLU tasks such as text similarity and classification.

![](/post_images/Screen-Shot-2019-06-24-at-12.25.27-PM-1024x749.png)

### **Pretraining + Pretraining = ?**

With a fully pre-trained model at hand, the next logical step is to do fine-tuning like what we have said. But what happens if you do even more pre-training instead?

The authors from an interesting paper2 attempts to investigate this question by conducting some serious experiments on BERT and it results in some findings and recommendations to get more juice out from a fine-tuning process.

Three further pre-training approaches are performed:  
1) Within-task pre-training, in which BERT is further pre-trained on the training data of a target task.  
2) In-domain pre-training, in which the pre-training data is obtained from the same domain of a target task. For example, there are several different sentiment classification tasks, which have a similar data distribution. We can further pre-train BERT on the combined training data from these tasks.  
3) Cross-domain pre-training, in which the pre-training data is obtained from both the same and other different domains to a target task.

![](/post_images/Screen-Shot-2019-06-24-at-12.27.43-PM-1024x606.png)

It has shown that within-task and in-domain further pre-training can significantly boost its performance but it might hurt performance when you pre-train with small dataset for within-task. Cross-domain approach does not perform well.

### **Pretraining + Fine-tuning = ?**

The authors also investigated different fine-tuning strategies.

**Input text length**

Since BERT accept a maximum sequence length of 512, what's the best methods for dealing with long text? They explore truncation methods such as:

1. head-only: keep the first 510 tokens;
2. tail-only: keep the last 510 tokens;
3. head+tail: empirically select the first 128 and the last 382 tokens.

And also hierarchical methods where text is firstly divided into k = L/510 fractions, which is fed into BERT to obtain the representation of the k text fractions. The representation of each fraction is the hidden state of the \[CLS\] tokens of the last layer, just like how we use BERT for classification task. Then they explore using mean pooling, max pooling and self-attention to combine the representations of all the fractions.

![](/post_images/Screen-Shot-2019-06-24-at-12.29.41-PM.png)

It shows that the truncation method of head+tail achieves the best performance.

**Features from Different layers**

Since BERT has 12 layers, and each layer captures the different features of the input text. Which layer is the most effective when you use it for fine-tuning?

![](/post_images/Screen-Shot-2019-06-24-at-12.30.23-PM-873x1024.png)

Interestingly, "Last 4 Layers + max" is as good as the intuitive last layer.

**Catastrophic Forgetting**

Catastrophic forgetting is a common problem in transfer learning, where the pre-trained knowledge is erased during learning of new knowledge. The general solution is to use a lower learning rate, such as 2e-5, to overcome the catastrophic forgetting problem. With an aggressive learn rate of 4e-4, the training set fails to converge.

![](/post_images/Screen-Shot-2019-06-24-at-12.32.40-PM-1024x279.png)

Besides a lowered learning rate, it's crucial to assign a lower learning rate to the lower layer. The idea is to be **extremely careful** when re-adjusting the weights of the intialized layers.

Besides this paper, there is a growing interest to extensively investigate the linguistic ability of these pre-trained LM models. In fact, an interesting research3 was done to evaluate them across 16 tasks.

The future of transfer learning burns brightly and evolving swiftly; it's both exciting and intimidating for a NLP practitioner.

Thank you for reading and I hope everyone has learnt something today.

**References**  
\[1\] https://arxiv.org/abs/1901.11504  
\[2\] https://arxiv.org/abs/1905.05583  
\[3\] https://arxiv.org/abs/1903.08855
