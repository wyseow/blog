---
title: "Fine Tuning OpenAI GPT for Sentence Summarization"
date: "2019-06-23"
categories: 
  - "deep-learning"
  - "nlp"
tags: 
  - "gpt"
  - "transfer-learning"
  - "transformer"
coverImage: "/post_images/Screen-Shot-2019-06-23-at-2.05.32-PM.png"
---

Transfer learning is on the rage for 2018, 2019, and the trend is set to continue as research giants shows no sign of going bigger.

With all the talk about leveraging transfer learning for a task that we ultimately care about; I'm going to put my money where my mouth is, to fine tune the OpenAI GPT model\[1\] for **sentence summarization** task.

Sentence summarization, or headline generation in some literature, is the task of producing a shorter version of a long sentence that preserves most of the meaning.

This task mainly uses the Gigaword summarization dataset has been first used by Rush et al., 2015\[2\] and represents a sentence summarization / headline generation task with very short input documents (31.4 tokens) and summaries (8.3 tokens). It contains 3.8M training, 189k development and 1951 test instances. Sentence summarization models are evaluated with ROUGE-1, ROUGE-2 and ROUGE-L using full-length F1-scores.

You can download the dataset processed by Rush et al., 2015 [here](https://github.com/harvardnlp/sent-summary).

### **Why GPT**

A language model such as OpenAI GPT model which has been pretrained on a very large corpus of text is able to generate long stretches of contiguous coherent text. Such lingustic ability would allievate a sentence summarization model from having to learn a huge task of generating coherent sentence and just focus on learning to extract the salient parts from the source sentence.

This post describes how we could fine-tune this pretrained language model to adapt it to our end-task: sentence summarization. Note that this endeavour deserves a demonstration of complete codes and I'm only showing the most important parts to make reading easy. Please visit my GitHub to get the complete codes.

**Import libraries**

from pytorch\_pretrained\_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIAdam

While the original GPT model is in Tensorflow, a company which specialize in NLP and chatbots called Huggingface has provided a PyTorch version of the pretrained OpenAI GPT model(horray to all Pytorch folks!)

The GPT model uses Byte Pair Encodings (commonly known as BPE)\[3\]\[4\] as the subword units for feeding in so we have to perform BPE tokenization on the input sequence. BPE is motivated by the open-vocabulary problem that NMT or LM tasks would meet. BPE creates a list of merges that are used for splitting out-of-vocabulary words.

**Creating the tokenizer with the vocab used in the pretrained model**

tokenizer = OpenAIGPTTokenizer.from\_pretrained('openai-gpt')

We could see how it's being tokenized:

eg\_text = 'Jim Henson was a puppeteer who invented'
tokenizer.tokenize(eg\_text)
# \['jim</w>','hen','son</w>','was</w>','a</w>','pupp','ete','er</w>','who</w>','invented</w>'\]

**Loading the pretrained GPT model**

model = OpenAIGPTLMHeadModel.from\_pretrained('openai-gpt')

### **Adapting language model to a sentence summarization task**

The GPT model is a auto-regressive LM that predicts the next word so how can we adapt the language model into the task?

![](/post_images/1*YmND0Qj8O6b35J1yU_CPKQ.png)

To be specific, the GPT model is trained on a sequence of words in this example format:

"Jim Henson was a puppeteer who invented"

to predict the next word: "the"

### **Re-framing the problem**

Instead of purely generating the next word, we could let the GPT model learn to start babbling a summarized sentence when a particular token is seen. This is similar to how we could use \[SEP\] token in BERT to signal the model to start generation.

![](/post_images/Screen-Shot-2019-06-23-at-12.19.19-PM.png)

But except the fact that, here's no special tokens such as the \[SEP\] we saw in BERT which was trained together with the sequences so we have add some special tokens that the GPT could learn when to start producing a summarized sentence. These special tokens are newly initialized and we trained their embeddings in the fine-tuning process.

A related question you might ask is why are we not using BERT to fine tune for a sentence generation. The short answer is that BERT is not a auto-regressive language model that could keep predicting the next word; it predicts the masked word within the sentence. Not saying that we can't generate sentence with BERT, after all, it has been pretrained with large corpus of text and it definitely exhibits tons of language ability but it's more tricky. In fact, this sub-area has attracted quite a bit of attention with some research8 working on exploring how we could use BERT for generating sentence.

Anyway, our input sequences with special tokens would look like:

<bos>long\_sentence<summ>summary\_sentence<eos>

Besides using the delimiter to demarcate the start and end of long sentence, we need a stronger signal for the model to learn the boundary between them. 

Therefore, in total we would have three parallel input sequences: word, positional and segment embeddings

![](/post_images/embeedd2.png)

With the pytorch-pretrained-BERT classes, it's easy to add our special tokens to the vocabulary of the tokenizer and create five additional embeddings in the model.

#add the special tokens into the vocabulary
SPECIAL\_TOKENS = \['<bos>','<summ>','<eos>', "<pad>"\]
tokenizer.set\_special\_tokens(SPECIAL\_TOKENS)
model.set\_num\_special\_tokens(len(SPECIAL\_TOKENS))

### **Training Strategy**

Fine tuning a pretrained model requires more care than training an ordinary neurel model. C Sun\[5\] has a great paper that describes the best practices of fine tuning a pretrained model to ensure a successful adaption of task and to prevent catastrophic forgetting. I'm going to talk more about the best practices of fine tuning in a later post.

To prevent that from happening, we linearly warm up the learning rate over about 10% of all steps and slowly decaying it over the rest of the steps. One such example of learning rate schedule:

![](/post_images/warmup_linear_schedule.png)

learning rate schedule

num\_train\_optimization\_steps = len(self.train\_dataset) \* options\['num\_epochs'\] // options\['train\_batchsize'\]

optimizer = OpenAIAdam(optimizer\_grouped\_parameters,lr=options\['model\_lr'\], warmup=options\['warmup\_prop'\],max\_grad\_norm=options\['max\_grad\_norm'\], weight\_decay=options\['opt\_weight\_decay'\],t\_total=num\_train\_optimization\_steps)

### **Training**

model = model.to(device=options\['device'\])

input\_ids, input\_segs, input\_labels = batch\_inputs
lm\_loss = model(input\_ids=input\_ids, position\_ids=None, token\_type\_ids=input\_segs, lm\_labels=input\_labels)
lm\_loss.backward()
optimizer.step()
optimizer.zero\_grad()

I'm skipping a lot of codes here as this is just an excerpt of the training loop

### **Sampling**

The 2 most common decoding strategies are greedy-decoding where you simply select the most likely next token which has the highest probability(argmax) and the beam search where you construct a beam of several hypothesis and you eventually select the best(highest average log-probability) hypothesis among the beams.

Beam search is the most popular option and a beam size of 3-5 would probably generate better than greedy decoding.

![](/post_images/wiazka.gif)

However, a recent work from Ari Holtzman et al.\[6\] showed beam-search and greedy decoding fail to reproduce some distributional aspects of human texts. Instead, he proposed the top-k and nucleus (or top-p) sampling. The basic idea is to sample from the next-token distribution after having filtered this distribution to keep only the top k tokens (top-k) or the top tokens with a cumulative probability just above a threshold (nucleus/top-p).

![](/post_images/1*yEX1poMDsiEBisrJcdpifA.png)

This is how we could do a top-k/top-p sampling for decoding sentence:

def top\_filtering(self, logits, top\_k=0, top\_p=0.0, threshold=-float('Inf'), filter\_value=-float('Inf')):
        assert logits.dim() == 1
        top\_k = min(top\_k, logits.size(-1))
        if top\_k > 0:
            indices\_to\_remove = logits < torch.topk(logits, top\_k)\[0\]\[..., -1, None\]
            logits\[indices\_to\_remove\] = filter\_value

        if top\_p > 0.0:
            sorted\_logits, sorted\_indices = torch.sort(logits, descending=True)
            cumulative\_probabilities = torch.cumsum(F.softmax(sorted\_logits, dim=-1), dim=-1)

            sorted\_indices\_to\_remove = cumulative\_probabilities > top\_p
            sorted\_indices\_to\_remove\[..., 1:\] = sorted\_indices\_to\_remove\[..., :-1\].clone()
            sorted\_indices\_to\_remove\[..., 0\] = 0

            indices\_to\_remove = sorted\_indices\[sorted\_indices\_to\_remove\]
            logits\[indices\_to\_remove\] = filter\_value

        indices\_to\_remove = logits < threshold
        logits\[indices\_to\_remove\] = filter\_value

        return logits

### **Training results**

After fine tuning this model using 100k samples, on an AWS instance with just 1 K80 GPU for 7 hours, it yields a Rouge-2 F1 score of 9.51, which is quite a distance away from the SOTA of 19.03. However, for a training data of such small size, and only a few hours for training, it's not too shabby.

These are some outputs from the model:

**Long sentence:**

1. police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a French antarctic research and supply vessel , a spokesman for the protesters said .
2. the sri lankan government on wednesday announced the closure of government schools with immediate effect as a military campaign against tamil separatists escalated in the north of the country .
3. turnout was heavy for parliamentary elections monday in trinidad and tobago after a month of intensive campaigning throughout the country , one of the most prosperous in the caribbean .

**Ground truth**

1. protesters target French research ship
2. sri lanka closes schools as war escalates
3. trinidad and tobago poll draws heavy turnout by John babb

**Predicted**

1. French antarctic police arrest five anti - nuclear
2. sri lanka announces closure of schools
3. turnout heavy for trinidad elections

We could see that with some fine tuning, GPT is able to learn the signal from delimiter on when to start decoding the summarized sentence.

Thanks to the pretraining of large corpus, it demonstrates its inherent ability to generate syntatically valid sentence, and this is something that even a Seq2Seq model specfically trained for this task with much bigger dataset would not be able to do as well.

While fluency is top-notch, it's not able to pay attend to all salient words, possibly due to a lack of explicit mechanism which many Seq2Seq summarization models possess.

### **Next steps**

The GPT model shows promising results by fine tuning on a small dataset and there are plenty of opportunities where one could perhaps use a much bigger dataset, use a newer pretrained LM such as GPT2 or introducing task specific layer at the top of the model where many such approaches\[7\] have yield good results on the GLUE leaderboard.

I hope you find this post helpful and thank you for reading.

**References**

\[1\] https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf  
\[2\]https://arxiv.org/abs/1509.00685  
\[3\] https://arxiv.org/abs/1508.07909  
\[4\] https://stackoverflow.com/questions/55382596/how-is-wordpiece-tokenization-helpful-to-effectively-deal-with-rare-words-proble  
\[5\] https://arxiv.org/abs/1905.05583  
\[6\] https://arxiv.org/abs/1904.09751  
\[7\] https://arxiv.org/abs/1901.11504  
\[8\] https://arxiv.org/abs/1902.04094
