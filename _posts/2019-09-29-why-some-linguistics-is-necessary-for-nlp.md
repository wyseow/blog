---
title: "Why some linguistics is necessary for NLP"
date: "2019-09-29"
categories: 
  - "nlp"
coverImage: "/post_images/12YwE2J-Tv6J9PI4BCikzKQ.png"
---

Disclaimer: I'm no means an expert in linguistics and below is the opinion of my personal research. Feel free to correct me.

![](/post_images/Hashtags.png)

After sitting in the NLP classes for the last 3 weekends, my classmates exclaimed that it felt like they went through 3 adult English classes. It didn't help that the workshops are designed to mow down every NLP library out there to reduce the NLP problem down to yet another machine learning problem. Seriously, I couldn't blame them; many classes and online tutorials fail to deliver the _cognitive goal_ behind the task and simply supply the technological goal_._

James Allen, an established computational linguist said this in 1987:

There can be two underlying motivations for building a computational theory. The technological goal is simply to build better computers, and any solution that works would be acceptable. The cognitive goal is to build a computational analog of the human-language-processing mechanism; such a theory would be acceptable only after it had been verified by experiment.

We could build a reasonable NLP system by taping together a bunch of technology and computation tricks or we could care whether it understands the language in the same way as a human. 

Many such classes are coming from the technologists' perspective, which isn't wrong. But it's important to keep this other perspective in mind as well. And this could mean returning to our roots in knowledge representations and reasoning for language. 

So today, I believe we have previously talked a lot about different works in NLP but most of the talks are revolving around the computational and mathematical aspects. We havn't look at it much from a linguistic perspective and that brings us to the point of why we should do that and how they could be mapped to actual practical use with real benefits. 

**The short answer to: Why some linguistics is necessary**

It allows us to reason about errors and explore the competency of a model. For example, given the output, is there a syntax(syntactic) or a semantic issue? In syntax, we are considered about the structure of a language, and in semantic, we are concerned about the underlying meaning of the language.

### **Syntax/Syntactic**

Consider this generated sentence:

Basketballs **rolls** across the floor. 

We could say that the model is not sensitive to sentence structure. It fails the subject-verb agreement task (one of the diagnostic methods proposed by Tal 2016\[1\] to assess the linguistic capabilities of LSTM). Or in the case where most of your generated sentences pass the agreements, but why did they falter in certain cases? Perhaps they are structurally complex sentences?

That comes to the point of sentence formation rules and patterns for every language. In English, a simple sentence can be constructed by subject–verb–object (SVO) is a sentence structure where the subject comes first, the verb second, and the object third. Sentence structure is one of the features (phenomena) in linguistic. Another common word order is SOV, which is used by the Japanese language.

![](/post_images/Screenshot-2019-09-28-at-12.11.39-PM.png)

Another way to look at it is that a sentence is constructed by clauses and clauses consists of subjects and predicates:

![](/post_images/Screenshot-2019-09-28-at-12.25.29-PM-1024x476.png)

The image shows the hierarchy of words and their roles in the structure of a longer sentence.

**Again, before we slip into another English class, let's ask ourselves why would we bother knowing all this?**

Consider a problem of text summarization and look at the predicate(diagram above) as the representation of the "main action" of the sentence. Therefore, it's important to be able to represent this information in abstractive settings or retain in extractive settings. In fact, there are approaches(Predicate-argument based approach) which they make use of such structure to identify salient parts and eventually merged them through a NLG process.

I hope by now we have started to build an intuition that language is structured and hierarchical. By leveraging on hierarchy, we could already fulfill a lot of NLP tasks. 

Words in a sentence are not a list of tokens but there are relationships between them. They collectively build up to a bigger unit and form semantics meaning. That kind of link (dependency) between linguistic units gives the motivation of dependency grammar, where we could construct a grammatical structure of a sentence, establishing relationships (depend or modify) between words. 

![](/post_images/Screenshot-2019-09-26-at-12.03.03-PM.png)

corenlp.run

How to read it: The root word here is "moving", and the arrow from the word "moving" to the word "faster" indicates that "faster" modifies "moving", and the label advmod assigned to the arrow describes the exact nature of the dependency.

Often, these annotations are used as features in addition to word embeddings, in an attempt to introduce the notion of linguistic structure into models more explicitly.

On the other end, phrase structure or constituency grammars are based on constituency relation, as opposed to the dependency relations.

![](/post_images/Screenshot-2019-09-26-at-12.02.38-PM.png)

We could see that there's a hierarchy of nested constituents, noun and verb phrases.

**Again, how is this being used?** We could use the "chunks" in the tree to provide us nuggets of information. Such information could be useful for sentence segmentation or practical cases where you need to split a long sentence into multiple logical lines. In fact, we don't need to do a full parse; "chunking" or "shallow parsing" methods are a quick way to get such information.

## **Semantic**

Semantic concerns the literal meaning of sentences and phrases. The word "plant" could mean:

- an organism
- manufacturing facility
- the action of sowing 

and knowing which one is relevant requires comprehending the context(eg. looking at surrounding words is one way).

NLP practitioners and researchers often argued that many NLP models lack the semantic knowledge required to generalize to real-world problems although they perform well to NLU tasks. Semantic capabilities are like a holy grail of the NLP world which many models claim to possess as it's the key to the understanding the meaning of language, and true understanding is the prerequisite for many tasks such as NLG.

> I think the biggest open problems are all related to natural language understanding. \[...\] **we should develop systems that read and  
> understand text the way a person does**, by forming a representation of the world of the text, with the agents, objects, settings, and the  
> relationships, goals, desires, and beliefs of the agents, and everything else that humans create to understand a piece of text.   
> Until we can do that, all of our progress is in improving our systems’ ability to do **pattern matching**.
> 
> Kevin Gimpel

There are a few methods we generally agreed on that could test for semantic capabilities:

**1) Name entities recognition:**  
\[Jim\]**Person** bought 300 shares of \[Acme Corp.\]**Organization** in \[2006\]**Time**.

**2) Textual Entailment**  
An example of a positive TE (text entails hypothesis) is:

hypothesis: Giving money to a poor man has good consequences.  
text: If you help the needy, God will reward you.

Given the hypothesis, the model would determine whether the text entails the hypothesis (yes/no).

For more NLU tasks, check out the GLUE benchmark.

**How is this connected to us?** 

We could use this to reason whether the model has attained a certain level of linguistic competency by pure observation of model outputs. Or we could use probing tasks\[2\] to unearth the linguistic features possibly encoded in neural models. Perhaps by projecting these encoding into an interpretable space, we could get a sense of which or what kind of layers exhibit such useful information and how can we exploit them. Following the success of BERT, there has been an interesting slew of experiments\[2\]\[3\] which tries to unpack the rich hierarchy of linguistic signals that BERT embedded.

Also, if we could extract the semantic of a sentence, we normalized it into a "surface form" and handle/manipulate it objectively, linking it to a knowledge base to pull up more related information. 

![](/post_images/4fa8db181a62506bdb82cfc50c90f4fe.gif)

I believe that achieving semantics through various means will be an active area of research for many more years to go it interweaves semantic, pragmatic, cognitive and social aspects. Of course, there will always be [debates](https://medium.com/huggingface/learning-meaning-in-natural-language-processing-the-semantics-mega-thread-9c0332dfe28e) on "what is true meaning".

### **Pragmatics**

Pragmatics is possibly the far frontier of NLP and as we move to higher levels of linguistic capabilities, we are also increasing the level of abstraction and ambiguity. It means going beyond the literal compositional meaning of the words, relying on knowledge of general principles of human communication, but not on extra-linguistic and contextual knowledge. 

The difference between pragmatics and semantics can get a little murky but pragmatics are usually about social relevance, context of use, and intentions.

![](/post_images/11.jpg)

I mentioned about the context in both semantics and pragmatics but the "context" is different. Here's a more specific example:

**Semantics:**

"Today is thursday" means today is not weekends. We have just use the literal meaning (thursday is a weekday?)

**Pragmatics**

"Today is thursday" means a suggestion, implying an invitation or even asking someone out! The context here could be from a previous conversation or a situational context. 

### **Closing up for today**

To close things up we could look at linguistics at different levels from a bird eye view:

![](/post_images/main-qimg-e363f4900d548525699bbf7657555411-300x300.png)

There's more to the things I have talked about in this post and the Stanford CS224 course is a huge source of my knowledge so I would encourage anyone who wants to know more to check that out.

I think it's an exciting time to be alive in the NLP and AI(generally) community as it's a golden era where compute resources are abundant and communication thrives between research groups.

References: 

\[1\] Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. Assessing the ability of LSTMs to learn syntax-sensitive dependencies.  
\[2\] What does BERT learn about the structure of language?  
\[3\] Open Sesame: Getting Inside BERT’s Linguistic Knowledge  
Selected images from: [https://www.hackerearth.com/blog/developers/natural-language-processing-components-and-implementations/](https://www.hackerearth.com/blog/developers/natural-language-processing-components-and-implementations/)  
Useful links:  
[https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72)  
[https://slideplayer.com/slide/4671548/](https://slideplayer.com/slide/4671548/)  
[https://en.wikipedia.org/wiki/Natural\_language\_processing#Major\_evaluations\_and\_tasks](https://en.wikipedia.org/wiki/Natural_language_processing#Major_evaluations_and_tasks)
