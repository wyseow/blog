---
title: "Crossing the language barrier with NLP"
date: "2019-07-18"
categories: 
  - "machine-learning"
  - "nlp"
coverImage: "/post_images/hello-words-pattern-different-languages_23-2147868000.jpg"
---

One of the biggest open problems in NLP is the unavailability of many non-English dataset. Dealing with low-resource/low-data setting can be quite frustrating when it seems impossible to transfer the same success we saw in various English NLP tasks. In fact, there are voices within the NLP community to advocate research and [focus on low-resource language](http://ruder.io/4-biggest-open-problems-in-nlp/) instead of spending the effort on beating the benchmark.

Fortunately, promising ideas have increasingly appear in the last couple of years, such as Multi-Lingual Language Model, Cross-lingual representations and even a new cross-lingual dataset which would lower the entry barrier for many NLP practitioners. Let's look at some of the interesting ones.

#### XNLI: The Cross-Lingual NLI Corpus

Before we touch on any new models, we should look at this exciting new dataset.

XNLI is an evaluation corpus for cross-lingual **Natural Language Inference (NLI)** task through sentence classification, in 15 languages. There are 5k test, and 2.5k dev pairs of sentences annotated with textual entailment and translated into 14 languages. **Recognising textual entailment (RTE)**, an evaluation method (I call that a mini-task) that supports the task of NLI, we usually have a premise, a corresponding hypothesis and the model predicts whether they agree to each other. Here's an example:

![](https://camo.githubusercontent.com/b897558046365450b4b49fd23f2bc72adbd3b0bd/68747470733a2f2f646c2e666261697075626c696366696c65732e636f6d2f584e4c492f786e6c695f6578616d706c65732e706e67)

Yay or Nay.

Some people might argue that NLI isn't the usual downstream task in many practical applications, but NLI is a decent middle-ground for evaluating the Cross-Lingual Understanding(XLU) - which is the cross-lingual version of Natural Language Understanding (NLU) - capabilities of a cross-lingual NLP model.

That's quite a mouthful of acronyms. To be more precise, Natural Language Processing (NLP) is an umbrella term that involve machines to perform textual tasks through processing. On the other hand, Natural Language Understanding (NLU) could be seen as a smaller subset of NLP that concerns about getting machines to really understand the meanings of text and we use [NLI datasets](https://gluebenchmark.com/tasks) to measure how well the machines could understand. But you might ask whats the definition of understanding, and for now, our definition of understanding stays within the realm of classifying entailment, sentiment, similarity and similar classification task.

There's 1 more: Natural Language Generation (NLG), which concerns about the capability of generating a sequence of words. I believe that a model needs to attain a decent NLU before being able to perform NLG tasks well; the model might be generating fluent sentences but it doesn't make sense because it doesn't understand the underlying meanings of the words. And by the way, that's a separate task of [common sense reasoning](https://nlpprogress.com/english/common_sense.html), which the NLP community has been talking about it recently. I personally think that NLI is a lower level form of "linguistic intelligence" compared to NLU because it's relatively easy for models to know which sentences are contradicting, or the sentiment of the sentence merely by memorising certain words.

I have digressed.

#### MultiLingual BERT

It turns out that the famous BERT model, which has broken the SQuAD leaderboard when it first came out, has a **[Multilingual](https://github.com/google-research/bert/blob/master/multilingual.md)** version (M-BERT) which is pretrained from monolingual corpora in 104 languages! It's a great news to us folks who build NLP models for non-English markets because none of the similar large pretrained language models such as OpenAI GPT, GPT2, [XLM](https://github.com/facebookresearch/XLM), or even the newest [MASS](https://github.com/microsoft/MASS) from Microsoft is trained across so many languages. The closest one is probably [ULMFIT](https://forums.fast.ai/t/language-model-zoo-gorilla/14623) but its not official and community contributed.

🗣But then again you might ask: **How multilingual is M-BERT?**

That's a good question and Google research reveals some interesting answers from their [paper](https://arxiv.org/abs/1906.01502).

The multilingual representation generalised across languages pretty well but performs best when the languages has a **high lexical overlap** (written in same scripts).

🗣Then again you would say: isn't that quite obvious?

If they share a single multilingual vocabulary, word pieces present during the fine tuning also appears in the evaluation, there's probably some "leakage", or memorisation of vocabulary. Is this a superficial form of generalisation? Most importantly, is transfer even possible for languages that don't share the same script (EN-JA)?

![](/post_images/Screen-Shot-2019-07-18-at-2.22.16-PM.png)

Yes, transfer is still possible.

An experiment which uses the Name Entity Recognition(NER) task across 16 languages show that although score is relatively flat for a range of lexical overlap, it could still achieve at least 40% even if there's no lexical overlap. In other words, M-BERT probably learn a deeper representation than simple vocabulary memorisation and still somewhat useful for languages which are written in different scripts, although not as much.

I find that quite impressive because it was trained on separate monolingual corpora so somehow it has learn a common subspace which represents useful linguistic information, in a language agnostic way.

On a similar note, this advantage of lexical overlap echoed an observation from the paper for XLM, where they are able to improve the performance of a LM by adding more data from a language that share the same script.

![](/post_images/Screen-Shot-2019-07-18-at-3.09.38-PM.png)

It is likely to work best for **typologically similar** languages.

It uses the example of EN-JA pair, which are not typologically similar. English words are ordered in Subject-Verb-Object (SVO) while Japanese in SOV. On the other hand, if we test the transferability on a language that is of similar ordering to English, such as Bulgarian, it works great. It shows that M-BERT have trouble generalising between languages of different ordering.

![](/post_images/Screen-Shot-2019-07-18-at-12.08.08-PM.png)

EN-JA is not as effective as EN-BG

🗣**How do you quantitively measure similarity between languages?**

It turns out that by using [WALS](https://wals.info) features, which is a large database of structural (phonological, grammatical, lexical) properties of languages gathered from descriptive materials (such as reference grammars) by a team of 55 authors, you determine the number of overlapping attributes between the languages you want to compare with.

![](/post_images/Screen-Shot-2019-07-18-at-12.17.06-PM.png)

number of WALS features correlates wth accuracy

As expected, an experiment shows that the languages which share a higher number of WALS features yield better accuracy. **It easier for M-BERT to map linguistic structures when they are more similar**.

M-BERT has definitely learn the generalised structure of languages and the transformation needed to accommodate from one to another. But probably **not all kinds of structure**. We could see that when the expected transfer of structure is bigger, such as the case of typological feature (word order), it fails to do well. But if the amount of transfer is small, such as the case of "adjective/noun" order, it can still do well.

![](/post_images/Screen-Shot-2019-07-18-at-1.59.24-PM.png)

Differences in AN/NA tests are much lower compared to SVO/SOV tests

To sum things up, the M-BERT if the pair of languages share the same word order, it should do well. If not, you are out of luck.

These key takeaways from the probing experiments shows promising possibilities of using M-BERT in many NLP tasks where we don't even have any dataset to begin with and I believe by using M-BERT as an encoder to produce multilingual representation for a bigger model or fine tuning to a NLP task straightaway, you would probably have a better chance of achieving results.

I feel that there are challenges for languages which are dissimilar in a way we mentioned and the solutions have not been fully addressed by M-BERT. But let's leave that discussion for a future post...
