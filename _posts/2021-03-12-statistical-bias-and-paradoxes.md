---
title: "Statistical Bias and Paradoxes that creep​ up in your data analysis"
date: "2021-03-12"
categories: 
  - "data-science"
  - "random-questions"
  - "statistics"
tags: 
  - "bias"
  - "paradox"
coverImage: "/post_images/paradox.png"
---

Statistical bias could creep up on our analysis and caused us to communicate the wrong insights and drive home the wrong conclusions.

There are actually many different kinds of bias, although most of us only associate bias with sampling bias, which is a kind of selection bias. Today, let’s put on an unbiased lens and learn about the different types of bias!

## 1\. Selection bias

Selection bias is the bias introduced by the selection of individuals, groups or data for analysis in such a way that proper **randomization** is not achieved, thereby ensuring that the sample obtained is not representative of the population intended to be analyzed.

Selection bias is an umbrella term that refers to the general idea as describe above and there are further divisions:

### 1.1 Sampling bias

Your sample is biased because of **non-random sampling**. To give an example, imagine that there are 10 students in a room and you ask if they prefer soccer or volleyball. If you only surveyed the three females and because all of them chosen volleyball, you concluded that the **majority** of people like volleyball, you’d have demonstrated sampling bias.

![](/post_images/sampling_bias.jpg)

As illustrated above, if you have selected and equal number of students from different genders, the conclusion would be very different.

**How to avoid:** Slicing and dicing your samples' variables by their proportion or statistics might give you a clue. When a A/B test is completed, we should also perform the same test to make sure that proper sampling is done or else the A/B test result will not be valid.

### 1.4 Confirmation bias

Confirmation bias is the tendency to favour information that confirms one’s beliefs. For example: If we have spent large amount of resource to conduct a survey to find out if Brand X is more popular then Brand Y then we would be very tempted to selectively look for evidence that supports Brand X, and perhaps ignore supporting data for Brand Y.

![](/post_images/1DpWsgJTRvs1WUNsUdtUJdg.png)

**How to avoid**: Look for ways to challenge what you think you see. Seek out information from a range of sources, and use an approach such as the [Six Thinking Hats](https://www.mindtools.com/pages/article/newTED_07.htm)  technique to consider situations from multiple perspectives.

## 2\. Survivorship bias

The phenomenon where only those that ‘survived’ a long process are included or excluded in an analysis, thus creating a biased sample.

A great example provided by Sreenivasan Chandrasekar is the following:

> _“We enroll for gym membership and attend for a few days. We see the same faces of many people who are fit, motivated and exercising everyday whenever we go to gym. After a few days we become depressed why we aren’t able to stick to our schedule and motivation more than a week when most of the people who we saw at gym could. What we didn’t see was that many of the people who had enrolled for gym membership had also stopped turning up for gym just after a week and we didn’t see them.”_

## 3\. Simpsons Paradox

Also known as the **Yule-Simpson effect**, it’s an effect in which a trend appears in several different groups of data but disappears or even reverses when these groups are combined.

This is a real-life example from a medical study comparing the success rates of two treatments for [kidney stones](https://en.wikipedia.org/wiki/Kidney_stone).

The table below shows the success rates and numbers of treatments for treatments involving both small and large kidney stones, where Treatment A includes open surgical procedures and Treatment B includes closed surgical procedures. The numbers in parentheses indicate the number of success cases over the total size of the group.

If we look at the aggregated success rate, we could claim that treatment B is performs better than treatment A because higher proportion of patients who have received treatment B has recovered. But the trend has reversed if we look at the groups themselves; treatment A looks more successfully in both stone sizes.

![](/post_images/1IfVjfdGD7RPwLDC6WzT9Uw.png)

### What actually happened?

While there are higher proportion of recovery in treatment A, the proportion is associated to lower sample size. On the other hand, although the proportion is smaller for treatment B, the larger sample size dominated the whole sample size.Therefore, when we aggregate the groups to a high level, the direction is reversed.

To analyse it holistically, we need to think about the process used to generate this data and a confounding variable(severity of case). If we know that doctors would favor B for smaller stones(because B is less invasive and there’s no need to be invasive if it’s deem less severe) and most importantly, patients who are less severe are likely to recover in the first place.

Therefore, we could say that the success rate is more influenced by the severity of the case rather the choice of treatment; if we look at the group of patients with large stones using treatment A (group 3) does worse than the group with small stones (groups 1 and 2), even if the latter used the inferior treatment B (group 2).

### So what?

Where it matters is the decision making situations where it poses the following dilemma: Which data should we consult in choosing an action, the aggregated or the partitioned?

My question to you: Which treatment do you think its the best?

Simpson’s Paradox is an interesting statistical phenomenon because it reminds us that **the data we are shown is not all the data there is.** We can’t be satisfied only with the numbers or a figure, we have to consider the data generation process — the causal model — responsible for _the data._ Once we understand the mechanism producing the data, we can look for other factors influencing a result that are not shown.

We have to consider other variables(confounding variables) not shown here, and plays a huge role in affecting treatment type and stone size, like severity of treatment.

## 4\. Base rate fallacy

This fallacy occurs when we disregard important information when making a judgement on how likely something is. 

If, for example, we hear that someone loves music, we might think it’s more likely they’re a professional musician than an accountant. However, there are many more accountants than there are professional musicians. Here we have neglected that the _base rate_ for the number of accountants is far higher than the number of musicians, so we were unduly swayed by the information that the person likes music.

Look out: The base rate fallacy occurs when the base rate for one option is substantially higher than for another.

**Another Example**

Consider testing for a rare medical condition, such as one that affects only 4% (1 in 25) of a population.

Let’s say there is a test for the condition, but it’s not perfect. If someone has the condition, the test will correctly identify them as being ill around 92% of the time. If someone _doesn’t_ have the condition, the test will correctly identify them as being healthy 75% of the time.

So if we test a group of people, and find that over a quarter of them are diagnosed as being ill, we might expect that most of these people really do have the condition. But we’d be wrong.

![](/post_images/base_rate_fallacy-1024x557.png)

According to our numbers above, of the 4% of patients who are ill, almost 92% will be correctly diagnosed as ill (that is, about 3.67% of the overall population). But of the 96% of patients who are not ill, 25% will be _incorrectly_ diagnosed as ill (that’s 24% of the overall population).

What this means is that of the approximately 27.67% of the population who are diagnosed as ill, only around 3.67% actually are. So of the people who were diagnosed as ill, only around 13% (that is, 3.67%/27.67%) actually are unwell.

### Solution

Consider the false positive, false negative rate of the test, and not just pure accuracy. In this case, the false positive rate is starkly high at 25% and the false negative rate is 8%, so just from thinking about this rate and the number of people who are likely to be healthy should ring some bells in our head.
