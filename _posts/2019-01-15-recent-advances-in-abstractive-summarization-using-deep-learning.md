---
title: "Recent Advances in Abstractive Summarization Using Deep Learning"
date: "2019-01-15"
categories: 
  - "deep-learning"
  - "nlp"
tags: 
  - "deep-learning"
  - "nlp"
coverImage: "/post_images/0K8eg3bUVu4AG-4FB.jpeg"
---

There has been a lot of advances in NLP and abstractive text summarization in these couple of years. While the first method that comes to our mind is deep learning, there are actually a lot more different ways to model the abstract representation of the text.

\[caption id="attachment\_364" align="aligncenter" width="400"\]![](/post_images/Screen-Shot-2019-01-15-at-11.52.03-AM-300x191.png) Abstractive summarization: An overview of the state of the art (S. Gupta, 2018)\[/caption\]

 

In this post, I'll be focusing at the Deep Learning method and I'll be talking about the other methods in a future post.

### "**Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond**" (Nallapati et al., 2016).

One of the most notable model in the early days that serves as the baseline comes from this paper.1

Seq to Seq (Encoder-Decoder) RNNs with attention are traditionally used for Machine Translation (MT) problems at that time, and they applied a similar model for summarisation and turns out it outperformed state of the art systems. They also describe several improvements to address problems that are specific to text summarization.

\[caption id="attachment\_379" align="aligncenter" width="678"\]![](/post_images/Screen-Shot-2019-01-15-at-4.47.31-PM-1024x450.png) Attentional Seq to Seq RNN Model\[/caption\]

**Large Vocabulary Trick (LVT)**

They use the large vocabulary trick (LVT) described in Jean et al. (2014) to reduce the decoder-vocabulary of each mini-batch to the words in the source documents of that batch. The aim of this technique is to reduce the size of the soft-max layer of the decoder which is the main computational bottleneck. Also, it speeds up convergence by focusing the modeling effort only on the words that are essential to a given example. Since a large proportion of the words in the summary come from the source document, it would work quite well.

**Meta-features**

Beside feeding in words features in the form of word embeddings, they also create capture additional linguistic features such as parts-of-speech (POS) tags, named-entity tags, and TF and IDF statistics of the words. They create an embedding for POS tags, similar to word embeddings and for TF and IDF, they convert them into categorical values by discretizing them into a fixed number of bins, and use one-hot representations to indicate the bin they fall into. What is being fed into the encoder at each time step is a long vector which is the concatenation of word embeddings vector, POS embedding vector, NE embedding vector, bin vector.

\[caption id="attachment\_369" align="aligncenter" width="400"\]![](/post_images/Screen-Shot-2019-01-15-at-12.58.41-PM-300x163.png) Feature-rich-encoder: One embedding vector each for POS, NER tags and discretized TF and IDF values, which are concatenated together with word-based embeddings as input to the encoder.\[/caption\]

**Modeling Rare/Unseen Words using Switching Generator-Pointer**

Traditionally, many abstractive summariser's decoder output a 'UNK' for Out of Vocabulary (OOV) words, which are words that the model has not seen during training time. It's a big concern because it's very likely that new or trending words such as celebrities, new product or companies names are not in the training data. And these words could play a central role in describing the topic of the summary.

An intuitive solution is: decoder is equipped with a ‘switch’ that decides between using the generator or a pointer at every time-step. If the switch is turned on, the decoder produces a word from its target vocabulary in the normal fashion. However, if the switch is turned off, the decoder instead generates a pointer to one of the word-positions in the source. The word at the pointer-location is then copied into the summary.

\[caption id="attachment\_371" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-15-at-1.13.54-PM-300x53.png) The prob of switching is modelled as a sigmoid function with switch parameters and decoder's hidden states, context vectors at each time step.\[/caption\]

\[caption id="attachment\_372" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-15-at-1.21.11-PM-300x76.png) If pointer is 'activated', the model probe the attention distributions to sample the pointed word. Note that it uses the encoder’s hidden-state representation to decide which word from the document to point to.\[/caption\]

**Modelling long document with Hierarchical Attention**

Applying attention over long source text is not as effective as short text. In order to identify the key sentences from a long documents, they introduce 2-level attentions, one at the word level and the other at the sentence level.

\[caption id="attachment\_374" align="aligncenter" width="478"\]![](/post_images/Screen-Shot-2019-01-15-at-1.43.21-PM-1024x614.png) The attention weights at the word level, represented by the dashed arrows are re-scaled by the corresponding sentence-level attention weights, represented by the dotted arrows. The dashed boxes at the bottom of the top layer RNN represent sentence-level positional embeddings concatenated to the corresponding hidden states.\[/caption\]

**Temporal Attention**

They noticed that the **same sentence or phrase often gets repeated** in the summary. Therefore, they used the Temporal Attention model of Sankaran et al. (2016) that keeps track of past attentional weights of the decoder and explicitly discourages it from attending to the same parts of the document in future time steps.

\[caption id="attachment\_386" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-16-at-9.17.18-PM-300x115.png) At the t-th timestep, we calculate the attention weights  for t divided by the summation of attention weights for t-1 timestep, which also means that we down weighting the attention weights at t if past attention weights are high on same parts of text.\[/caption\]

After applying this, the authors noticed few repetitions of same words or phases.

**Training specifics**

1. Source vocabulary size is 150K, and the target vocabulary is 60K
2. Source and target lengths to at most 800 and 100 words respectively.
3. 100-dimensional word2vec (Mikolov et al., 2013) embeddings trained on this dataset as input, and unfrozen during training.
4. RNN hidden state dim at 200
5. Used only the **first 1-2 sentences** of the document as the source text. They experience decreased performance once more sentences are used
6. Average number of words in summary sentences are 7-8.

**Results**

We will review the ROUGE-2 scores tested on the CNN/DailyMail dataset, which contain multi-sentence summaries.

\[caption id="attachment\_376" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-15-at-3.43.49-PM-300x73.png) Switching pointer/generator model as well as the hierarchical attention model described fail to outperform the baseline attentional decoder.\[/caption\]

\[caption id="attachment\_377" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-15-at-4.00.12-PM-300x224.png) Arrow indicates that a pointer to the source position was used to generate the corresponding summary word.\[/caption\]

**One of the good quality summary output**

Source: volume of transactions at the nigerian stock exchange has continued its decline since last week , a nse official said thursday . the latest statistics showed that a total of ##.### million shares valued at ###.### million naira -lrb- about #.### million us dollars -rrb- were traded on wednesday in , deals . Target: transactions dip at nigerian stock exchange Generated summary: transactions at nigerian stock exchange down

From this example we could see that although generated summary differs from the target summary, its summaries is still relevant, and this is a phenomenon not captured by word overlapping evaluation metrics, ROUGE-2.

**One of the poor quality summary output**

Source: norway delivered a diplomatic protest to russia on monday after three norwegian fisheries research expeditions were barred from russian waters . the norwegian research ships were to continue an annual program of charting fish resources shared by the two countries in the barents sea region . Target: norway protests russia barring fisheries research ships Generated summary: norway grants diplomatic protest to russia

**Other problems**

We will see how this is fixed in future experiments by other papers.

 

### **"Get To The Point: Summarization with Pointer-Generator Networks"** (See et al., 2017)

A later paper2 attempted to improve the pointer mechanism and to fix the problem of repeating sentence by introducing a **coverage mechanism**.

**Pointer-generator network**

While pointer mechanism is nothing new at the point of writing, they have made some improvements which yield improve in quality of summary.

Compared to Nallapati 2016 model where they have a pointer switch(1 or 0) that decides whether to use words from vocabulary or source text, See 2017 model has a soft switch that serves as a weightage. Next, it creates a extended vocabulary, which is union of the vocabulary, and all words appearing in the source document. Then, the soft switch/weightage is used to compete the probability distribution over the extended vocabulary. It is better explained using the diagram.

\[caption id="attachment\_383" align="aligncenter" width="678"\]![](/post_images/Screen-Shot-2019-01-16-at-5.33.18-PM-1024x564.png) The vocabulary distribution and the attention distribution are weighted and summed to obtain the final distribution, from which we make our prediction. We could see that OOV words such as 2-0 are included in the final distribution.\[/caption\]

**Coverage mechanism**

To fix repetition and redundancy problem which is common in Seq to Seq RNN models (actually to general summarization systems as well), the coverage mechanism of Tu et al. (2016) is used. Notice that Nallapati 2016 model already has a Temporal Attention mechanism to ward against the same issue but the authors of this paper believe that method is too destructive; they believe it's better to inform the attention mechanism to help it make better decisions, than to override its decisions altogether.

In this case, a coverage vector is maintained, which is the sum of **unnormalised** attention distributions over all previous decoder timesteps:

![](/post_images/Screen-Shot-2019-01-16-at-5.32.19-PM.png)At a particular timestep (t), the context vector an accumulation of attention of all the past timesteps and it represents the amount of coverage that each words have received thus far. This coverage vector is then used in the encoder-decoder attention:

![](/post_images/Screen-Shot-2019-01-16-at-5.38.09-PM.png)

This is to ensure that when attention is computed(choosing where to attend next), the coverage of past words is taken into consideration.  This effectively avoid repeatedly attending to the same locations, and thus avoid generating repetitive text.

\[caption id="attachment\_389" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-16-at-9.33.50-PM-300x220.png) Coverage eliminates undesirable repetition.\[/caption\]

**Training specifics**

1. Word embeddings are not from pre-trained state and completely learnt from scratch, unlike Nallapati 2016 approach.
2. Due to the pointer mechanism ability to handle OOV words, only vocabulary of 50k words for both source and target. Nallapati et al.’s (2016) used 150k source and 60k target vocabularies.
3. Truncated the article to 400 tokens and limit the length of the summary to 100 tokens for training and 120 tokens at test time.
4. 230,000 training iterations (12.8 epochs); a total of 3 days and 4 hours.

**Results**

We will review the ROUGE-2 scores tested on the CNN/DailyMail dataset, which contain multi-sentence summaries. ![](/post_images/Screen-Shot-2019-01-17-at-8.39.33-AM-1024x341.png)Notice that it can't surpass the lead-3 baseline (which simply uses the first three sentences of the article as a summary). There are several reasons why.

1. News articles tend to be structured with the most important information at the start; this partially explains the strength of the lead-3 baseline. In fact, they found that using only the first 400 tokens (about 20 sentences) of the article yielded significantly higher ROUGE scores than using the first 800 tokens.
2. ROUGE calculates by number of overlapping words, and since abstractive summariser can produce semantically similar sentences with different words, the score would be low.

We will discuss more about evaluation metrics at a later part.

One of the advantages of abstractive summarisers is that they are able to produce novel words (words that don't appear in source text).

\[caption id="attachment\_394" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-17-at-9.50.52-AM-300x216.png) The % of novel words that different models used, and the amount of copying resulted from pointer mechanism.\[/caption\]

Although the pointer mechanism makes the abstractive system more reliable, it also reduce the % of novel words. While the baseline model has higher novelty, it includes all the incorrectly copied words,UNK tokens and fabrications alongside the good instances of abstraction. Increasing both novelty and reducing erroneous n-grams is a challenge that is partially solved in a later paper.

**Good examples**

Intensity of green shading represents value of the generation probability; lower means model likely to copy from source text, and higher means model likely to use words in vocabulary.  Intensity of yellow shading represents final value of the coverage vector at the end of final model’s summarization process. High means there's a high contention in the attention weights for these words.

[![](/post_images/Screen-Shot-2019-01-17-at-10.11.49-AM-254x300.png)](http://gator4205.temp.domains/~datageeko/wp-content/uploads/2019/01/Screen-Shot-2019-01-17-at-10.11.49-AM.png)Baseline model has issues with OOV words and fabricate details. No-coverage model has repeating issue. Coverage model is about to use back words from source text for OOV words like "saili" and fixes repetition issues. Notice that it's also able to skip over large passages of text to produce shorter sentences.

More examples in the Appendix of the paper.

### A Deep Reinforced Model For Abstractive Summarization (Paulus, 2017)

This paper3 published by Salesforce Research pushes the envelope of abstractive summarization forward further by infusing the concept of Reinforcement Learning (RL) and it has attracted a lot of attention from the non-tech folks due to the large number of forward-looking press releases.

It mainly addresses the issue of repeated sentences and introduces a RL learning objective.

**Intra-decoder attention aka self-attention at decoder**

The authors believed that while Nallapati et al.’s (2016) Temporal Attention ensures that different parts of the encoded input sequence are used, we still need to attend over decoder side of attention, as decoder can still generate repeated phrases based on its own hidden states, especially when generating long sequences.

**Hybrid Learning Objective**

Almost every sequence to sequence models uses minimisation of negative log likelihood as a training objective (teacher forcing). But minimizing it does not produce the best ROUGE scores because of 2 reasons:

1. Exposure bias (Ranzato et al., 2015), comes from the fact that the model has knowledge of the ground truth sequence during training but does not have such supervision when testing (uses predicted word in previous tilmestep), hence accumulating errors as it predicts the sequence.
2. There are many ways to arrange tokens(paraphrase) to produce summary.  The ROUGE metrics take some of this flexibility (overlapping of bigram) into account, but the maximum-likelihood objective does not.

Therefore, the authors suggested to incorporate a maximization of a specific discrete metric (ROUGE) that measures the quality of overall summary instead of minimising NLL and it could be frame as a RL problem as the metric is not differentiable. They proposed a mixed training objective that minimize NLL and maximise ROUGE score at the same time with a scaling factor that plays as the weightage.![](/post_images/Screen-Shot-2019-01-17-at-1.12.33-PM-1-300x55.png)

**Training specifics**

1. 100-dim pre-trained GLOVE Word embeddings, assumed to be frozen during entire training.
2. Source vocabulary size is 150K, and the target vocabulary is 50K. Roughly same as Nallapata's 2016
3. Teacher training enforced at 25% probability to reduce exposure bias.
4. Weightage of 0.9984 for RL component of mixed learning objective.

**Results**

\[caption id="attachment\_399" align="aligncenter" width="678"\]![](/post_images/Screen-Shot-2019-01-17-at-12.50.13-PM-1024x294.png) Notice that while the scores looks lower than See's 2017 model, it is said that See used a non-anonymous version of dataset so numbers couldn't be directly compared. However, See's best model has lower ROUGE scores than their lead-3 extractive baseline, so we could just use the lead-3 extractive baseline to compare.\[/caption\]

![](/post_images/Screen-Shot-2019-01-17-at-1.15.51-PM-1024x728.png)Although the models' ROUGE score didn't improve significantly, the idea of incorporating ROUGE metric as a mixed learning objective is an interesting direction and it paved the way for many similar training objectives in future attempts.

### Current trends in Abstractive Summarization

Recent papers in 2018 focus on applying different reward functions inspired by RL as a mixed objective function to improve generation of novel words.

**Abstractive Reward** (Kryscinski, Salesforce Research, 2018)4 is proposed to encourage the model to parse large chunks of the source document and create concise summaries using phrases not in the source document.

\[caption id="attachment\_405" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-17-at-3.16.01-PM-300x60.png) The fraction of unique n-grams in the summary that are novel, or different from the source.\[/caption\]

**Saliency Reward** (Pasunuru et al., 2018)5 is proposed to gives higher weight to the important, salient words/phrases when calculating the ROUGE score (which by default assumes all words are equally weighted). They trained a saliency predictor on sentence and answer pairs from the SQuAD reading comprehension dataset and use the probabilities of saliency as weights in the ROUGESal score.

![](/post_images/Screen-Shot-2019-01-17-at-3.38.50-PM-300x238.png)

\[caption id="attachment\_407" align="aligncenter" width="300"\]![](/post_images/Screen-Shot-2019-01-17-at-3.38.57-PM-300x88.png) We consider the weight assigned by the saliency predictor of the overlapping Longest Common Subsequence (LCS) tokens between reference summary sentence and generated summary.\[/caption\]

**Entailment Reward** (Pasunuru et al., 2018)5 is proposed to ensure that the generated summary sentence is logically entailed, contain no contradictory or unrelated information. Similarly, a predictor is also trained to calculate the entailment probability score between the ground-truth summary and each sentence of the generated summary and use avg. score as the Entail reward.

 

References cited in this post:

1 "Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond" (Nallapati et al., 2016)

2 "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., 2017)

3 "A Deep Reinforced Model For Abstractive Summarization" (Paulus, 2017)

4 "Improving Abstraction in Text Summarization" (Kryscinski, Salesforce Research, 2018)

5 "Multi-Reward Reinforced Summarization with Saliency and Entailment" (Pasunuru et al., 2018)
