---
title: "Recent Advances in Abstractive Summarization Using Deep Learning Part 2"
date: "2019-03-13"
categories: 
  - "deep-learning"
  - "nlp"
coverImage: "/post_images/0K8eg3bUVu4AG-4FB.jpeg"
---

This post is a continuation to the previous post [here](http://gator4205.temp.domains/~datageeko/recent-advances-in-abstractive-summarization-using-deep-learning/).

We continue to track the recent progress and trend in abstractive summarization in 2018.

The earlier efforts in abstractive summarisation focuses on problems that are related to natural language generation rather than the summarization task itself. Some problems that were tackled:

1. Unfactual information (copy mechanism)
2. OOV words (copy mechanism)
3. Word and sentence level repetition (coverage mechanism)

All these efforts are only sufficient to ensure that the generated summaries are **fluent and natural.** Most importantly they didn't address fundamental qualities of a good text summarization such as:

1. Non-redundancy
2. Referential clarity
3. Focus (Saliency)
4. Structure and Coherence
5. Coverage

Fortunately, we have started to see some recent works in this area and interestingly, the solutions involved incorporating ideas from the extractive summariser side of the world.

## "Learning to extract coherent summary via deep reinforcement learning" (Wu et al., 2018)

I'm playing cheat; why am I talking about extractive summarization where this post is about abstractive summarization? Just hang on with me and you will see. While the abstractive summarizers are trying very hard to surpass benchmarks, the extractive summarisers are having an easier time due to the ROUGE-2 calculation.

In the recent years, using neural networks to perform extractive summarization is becoming attractive. Although this method1 is nothing new, most of them ignore the **coherence** factor when extracting sentences. And this is where the paper comes in.

There are 2 parts in this approach. The first part (Neural Extractive Summarization Model) is a simple Bi-GRU sequence model that focuses on picking the right sentences from the document of sentences. The extractor looks similar to:

\[caption id="attachment\_415" align="aligncenter" width="163"\]![](/post_images/Screen-Shot-2019-03-12-at-11.24.02-AM-163x300.png) Trained by supervised learning, minimizing the negative  
log-likelihood of the ground truth extraction labels. Nothing fancy.\[/caption\]

After the **NES** have been pre-trained, they **further train it with reinforcement learning** with the aim to extract coherent and informative summaries by maximising coherent and ROUGE score. This model is called **RNES**.

We have seen the inception of RL in the recent developments of abstractive summariser, hugely because ROUGE metric is non-differentiable with respect to the model parameters and that is the perfect use case for RL.

The coherent score measure the level of entailment between 2 sentences, and this score is used as one of the rewards in RL.

**How do we compute the coherent score?**

We could throw the problem to a neural network; they propose a **neural coherence model** which is a series of convolution and max-pooling layers performed on the sentence embeddings to capture the syntactic and semantic coherence patterns between sentences.

![](/post_images/Screen-Shot-2019-03-12-at-11.25.41-AM.png)

They then used a pair-wise training strategy which is similar to the siamese network where we  train with triples of sentences (eg. A, B, C, where A and B are similar and C is the dissimilar to A).

\[caption id="attachment\_416" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-03-12-at-5.04.45-PM-300x46.png) objective is to minimise the ranking-based loss(similar to the constrative loss in siamese network)\[/caption\]

Once again, the objective of using RL in this context is to find out the optimal policy (sequence of sentences) where coherence scores are exploited as immediate rewards and ROUGE as the final reward.

**Qualitative results**

\[caption id="attachment\_418" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-03-12-at-5.18.07-PM-300x117.png) Results have surpassed benchmarks\[/caption\]

The authors find that although the variant without coherence perform better, the coherence objective and ROUGE score do not always agree with each other so they believe that the model with coherence reward is a better one.

**Quantitative results**

\[caption id="attachment\_419" align="aligncenter" width="282"\]![](/post_images/Screen-Shot-2019-03-12-at-5.23.50-PM-282x300.png) We could see sentences flowing naturally, as sentences start with "that" should not be the first.\[/caption\]

Their result in ROUGE-2 is pretty good. At this time of writing, they achieve the best score on the anonymized version of the CNN/Dailymail dataset.

Life is not fair; extractive summarizers just need to select the right sentences, but abstractive summarisers need to select the right sentences + generate new sentences fluently and correctly. And that comes to the point that recent abstractive summarizers have slowly moved to hybrid approach which combines extractive and abstractive methods.

## "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting", (Chen et al., 2018)

One of the top-5 models in ROUGE-2 score and inherited a lot of ideas from the paradigm of "extract-then-compress", this paper2 proposed a hybrid extractive-abstractive architecture which first selects/extract salient sentences and then rewrites them abstractively to generate a concise summary.

![](/post_images/Screen-Shot-2019-03-12-at-5.56.15-PM-1024x527.png)

**What is a CNN doing there?**

This is inspired by numerous convolutional neural network (CNN) based encoder-decoder models3,4,5 which many benefits such as shorter paths between pairs of input and output tokens, so that it can propagate gradient signals more efficiently. In another words, it could capture longer range dependency for long sentences with lesser computation time.

A Pointer Network6 is trained to extract sentences recurrently. This model is essentially classifying all sentences of the document at each extraction step.

Over at the abstractor side, they used a simple encoder-aligner-decoder7 with the copy mechanism8 which have seen already, to prevent OOV words problem.

While there is nothing fancy going on in the abstractor model, the interesting part of this paper is about using RL to jointly train the extractor and abstractor models in a end-to-end manner. But before we could do that, starting from a randomly initialized network would take ages to converge. Intuitivetly, when randomly initialized, the extractor would often select sentences that are not relevant, so it would be difficult for the abstractor to learn to abstractively rewrite.

Therefore, they pre-trained each model separately first.

**How to pretrain extractor to pick up salient sentences?**

While we do not have the saliency labels, we could provide a 'proxy' target label which can be derived by  picking the most similar sentences to ground-truth summary sentence which is similar to an earlier extractive model9. Given these proxy training labels, the extractor is then trained to minimize the cross-entropy loss. Simple as that.

**How to pretrain abstractor to rewrite?**

It's simple as taking each summary sentence and pairing it with its extracted sentence. The network is trained as an usual sequence-to-sequence model to minimize the cross-entropy loss.

Finally, both of the pretrained models are further trained end-to-end using RL. Compared to the last paper we reviewed, this paper uses only ROUGE as reward for learn the policy.

Intuitively, the RL training works as follow: If the extractor chooses a good sentence, after the ab- stractor rewrites it the ROUGE score would be high and thus the action is encouraged. If a bad sentence is chosen, though the abstractor still produces a compressed version of it, the summary would not match the ground truth and the low ROUGE score discourages this action.

**Qualitative results**

\[caption id="attachment\_427" align="aligncenter" width="678"\]![](/post_images/Screen-Shot-2019-03-13-at-10.45.45-AM-1024x393.png) At this time of writing, it achieved top 5 for ROUGE-2\[/caption\]

\[caption id="attachment\_428" align="aligncenter" width="500"\]![](/post_images/Screen-Shot-2019-03-13-at-10.53.39-AM.png) Gives a significant speed-up\[/caption\]

The reason why it's much faster is because they could first compute all the extracted sentences for the document, and then abstract every sentence concurrently (in parallel) to generate the overall summary. Since the computation bottleneck is on the abstractor, which has to generate summaries with a large vocabulary from scratch, putting this process to parallel yields the most time saving.

**Quantitative results**

![](/post_images/Screen-Shot-2019-03-13-at-11.21.59-AM-937x1024.png)

## Trends and further direction

We could see that there's a shift of abstractive summarizers towards a 2-stage "extract-then-abstract" process in several recent papers10,11 besides the ones we reviewed. Many feel that a single abstractor unit is not powerful enough to perform both tasks of identifying the salient information and rewriting it abstractively. Therefore, recent efforts involve decoupling these 2 tasks in various ways, and the task of extracting salient information has slowly converge into the realm of neural extractive summarizers and that is why we could see some ideas from the neural extractive summarisers side of world coming into the abstractive side.

While some may feel like it's a step backwards (going back to extractive methods), there are numerous benefits to it:

1. By framing it as separate extractor and abstractive modules, we can test the performance of each task more easily, and we could apply improvements more directly.
2. We could enjoy the works done at the extractive summarisers and apply it together with the abstractor module.

RL will continue to reign future work as the community attempts to tackle fundamental summarization criteria. There will be measurements that track these criteria, like the coherence score we saw earlier and due to the fact that they are not differentiateable, RL will be used to learn a policy that best produce the summaries we expect.

 

References 1 "Learning to extract coherent summary via deep reinforcement learning" (Wu et al., 2018) 2 "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting" (Chen et al., 2018) 3 "Neural machine translation in linear time" (Kalchbrenner et al., 2016) 4 "Quasi-recurrent neural networks" (Bradbury et al., 2016) 5 "Convolutional sequence to sequence learning" (Gehring et al., 2017) 6 "Pointer networks" (Vinyals et al., 2015) 7 "Neural machine translation by jointly learning to align and translate" (Bahdanau et al., 2015) 8 "Get to the point: Summarization with pointer-generator networks" (See et al., 2017) 9 "Summarunner: A recurrent neural network based sequence model for extractive summarization of documents." (Nallapati et al.,2017) 10 "Improving Abstraction in Text Summarization" (Salesforce Research, 2018) 11 "Improving Neural Abstractive Document Summarization with Explicit Information Selection Modeling" (Li et al., 2018a)
