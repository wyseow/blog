---
title: "Do you know what are the most talked about topics in Singapore?"
date: "2017-08-21"
categories: 
  - "data-mining"
  - "data-preprocessing"
  - "data-visualization"
  - "news"
  - "toy-projects"
tags: 
  - "beautiful-soup"
  - "data-mining"
  - "nltk"
  - "python"
  - "word-cloud"
coverImage: "/post_images/sgwordcloudlogo.jpg"
---

There are tons of news publications in Singapore, from the traditional The Straits Times, Today, and also the digital ones like ChannelNewsAsia.com and many more to grab your latest news fix but let's face it. \*Cough Cough\*. It's beginning to look like what they want to write, not what want to read and know.

Unless you have been hiding under a rock, you would notice that there's an increasing number of "alternative news site" appearing and people are increasingly getting their news from social media and forums, which is much more updated and access to topics that relate to everyone's heart. In the spirit of that, I put on my geeky hat and created a word cloud that collects all the most talked about topics and discussions from Singapore's major social media sites through some data mining techniques.

**Introducing, [WhatsUpSG.com](http://whatsupsg.com)!**

The below image is taken from WhatsUpSG.com.

\[caption id="attachment\_225" align="alignnone" width="678"\][![](/post_images/map_output-2-1024x671.png)](http://whatsupsg.com) Our glorious Singapore map!\[/caption\]

When you visit the website, you will see the word cloud in the shape of our Singapore map. As always, the size of the words express the frequency of the topics. As of today, you can see that people are talking about "SEA Games" and even "ghost month" since the annual Ghost Festival is just around the corner.

Data is mined and a new word cloud is formed every few hours, ensuring that everyone will know what are the most talked about words or phrases in the most updated speed that anyone can get! If you are struggling to come out with new conversation topics with your lunch colleagues, it's a good place to gather some ideas as well.

**Technical details - How I did it**

**Data mining** Using Python's Beautiful Soup framework, the script crawled the content of some web pages, and extracted the information that is important to us, like topic titles and excerpt of discussions, while ignoring ads and other irrelevant content. There's a [post](http://gator4205.temp.domains/~datageeko/easy-data-mining-for-your-data-science-projects/) earlier that describes how you could do data mining using PHP. But I'll post a new one (part 2) which describes how you could use Python's Beautiful Soup and Selenium to data mine dynamic websites.

**Data Pre-processing** With all the huge chunk of data collected, the script uses the popular Natural Language Processing(NLP) library, [NLTK](http://www.nltk.org) to do some pre-processing work like tokenising, removal of stop words and common words, reduce it to n-grams, create a frequencies list, before we feed it into the Word Cloud.

**Word Cloud** Finally, after all the heavy lifting, we feed the data to this awesome [Word Cloud library](http://amueller.github.io/word_cloud/index.html) and after choosing the best looking Singapore map in Google images for about 3 hours, I set it to a mask so that the word cloud would follow the outline of the Singapore map.

**Auto update** I set up a cron job that triggers the whole process every few hours in my Linux server that runs 24/7 and it's pretty fun to see all these work being automated!

I'll be sharing the codes soon in the form of IPython Notebook and blog posts, so stay tuned if you want to find out more about it. Although I don't get any monetary rewards from it,  I find this toy project a fun and rewarding exercise to sharpen my data mining and NLP skills.

**What's next:** I'm working with a friend who is a huge football fan to come out with a machine learning model that predicts EPL match results. P.S. It's giving a accuracy score of up to 65%! I'll share more in September!
