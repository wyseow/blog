---
title: "Easy Data Mining For Your Data Science Projects"
date: "2017-06-06"
categories: 
  - "data-mining"
coverImage: "/post_images/MineyPortraitHD.png"
---

Data science projects lives on data. Without huge amount of unbiased data to explore and play with, your seaborn graphs could be skewed, prediction models could be unreliable, and your company might even make the wrong business decision. I have a couple of toy data science projects that derive its data sources from website and while API could fulfil your data requirements most of the time, sometimes the data you need just isn't given-wrapped up nicely for you in a "webService->getWhateverFunctionHere()".

Here's what you put on your hacker-cap, play on your terms(horray, no API requests limits) and scrap the data on the website using a asernal of weapons. Also, the web scraping scripts and techniques could be re-applied to a new data project you have instead of spending time on learning a new thick API documentation. I believe that it's a skill that every data analyst or geek should have in his pocket.

There's tons of pros on scraping the websites, let's get to scraping.

**Preparing for attack**

\[caption id="attachment\_39" align="alignleft" width="300"\]![](/post_images/ilario-spolverini-a-regiment-preparing-the-siege-of-a-distant-fortified-town-300x201.jpg) Yes, I'm damn serious about this siege...\[/caption\]

The first thing you should do is to visit the website (obviously) and navigate around the website to do a first hand-inspection of the behavior of the website, especially the main webpage that you need to fetch the data from. Why do you do that? This is to answer: 1) Is the website protected by cloudflare? If yes, you will need to bypass it or else it will block your request. 2) Is javascript required to navigate and load your targeted webpage. AJAX? 3) The structure of the URL. What parameters are required, random value in variables, pagination, etc..

**Take 1 step further**

Inspect the web element that stores your data using the Web Inspector. If you are using Firefox or Safari, right click on the element and click "Inspect Element". An example: ![](/post_images/Screen-Shot-2017-06-07-at-12.04.13-am.png) Is the data in clear text? How is it being structured? If you are lucky, your data is lay out nicely in tables, with proper hierarchy, and sequential order. If your data is buried in a stack of web elements that do not do not relate well together, or in a random order, time to make a cup of coffee because it's going to take some work. In worse case scenario, it might be much easier to find another website that provide the same data and yet has a simplified HTML markup to deal with.

The method of scraping depends on the complexity of the website, and you can choose from an arsenal of weapons to go about it.

**Dealing with static websites**

If website is static, for example reddit.com, where the whole content is loaded just simply by doing a GET request of the webpage, you are in luck! You could simply use PHP curl to fetch the page, and use [DOMDocument framework](http://php.net/manual/en/class.domdocument.php) to form the dom document tree, then use Xpath to find the HTML Elements which contain the data you are scraping for.

**Case Study: Scraping the reddit news at reddit.com** For the sake of our example, imagine that your data science project involves scraping the Reddit news and perhaps based on the content of the news you could predict the trends of New York Stock Exchange(NYSE) index.

And that means we have to scape and collect a list of Reddit news links from it's front page.

In depth, this is how it's going to happen:

Step 1: You could fetch the targeted page using plain-o PHP Curl. But I personally use this library **[here](https://github.com/KyranRana/cloudflare-bypass)** to help me bypass Cloudflare just in case the page is protected by Cloudflare. It's just a wrapper on CURL that requests and maintains the cookie that Cloudflare is looking for. Include the library files at the top of your PHP script as follows:

\[php\] require\_once 'cloudflare/libraries/httpProxyClass.php'; require\_once 'cloudflare/libraries/cloudflareClass.php'; \[/php\]

Step 2: Perform a request using the library

\[php\] $requestLink = "http://your-targeted-url-here.com"; $requestPage = json\_decode($httpProxy->performRequest($requestLink)); \[/php\]

Step 3 : If page is protected by Cloudflare(returns a 503), spoof a fake cookie that Cloudflare is looking for. And use the fake cookie to perform a request again. Remember to spoof the user agent as well(more to that later).

\[php\] if($requestPage->status->http\_code == 503) { cloudflare::useUserAgent("your custom user agent here"); if($clearanceCookie = cloudflare::bypass($requestLink)) { $requestPage = $httpProxy->performRequest($requestLink, 'GET', null, array( 'cookies' => $clearanceCookie )); $requestPage = json\_decode($requestPage); } } \[/php\]

Finally, we have the HTML content of the targeted page!

Step 4: Let's build a DOM structure using the HTML codes we got so that we can fetch the items we want easily.

\[php\] $pageContentHTML = $requestPage->content; $dom = new DOMDocument; $dom->loadHTML($pageContentHTML); \[/php\]

Right now, you can treat the $dom object as the targeted webpage, and you can find the web element that holds the data you want. There are a few ways to go about it. One powerful way is to use xPath syntax.

XPath is a powerful tool. If you know the essence of it, you could scrape all the web elements in all kinds of complicated, heavily-buried-in-1000-divs HTML structure. I'll release a tutorial on XPath shortly and I'll reveal an easy way that you can form a XPath syntax easily without learning all the basics-sounds like a dream eh?

Back to our case study. If you look at the inner HTML structure of the reddit front page: [![](/post_images/Screen-Shot-2017-06-08-at-11.00.24-pm.png)](http://gator4205.temp.domains/~datageeko/wp-content/uploads/2017/06/Screen-Shot-2017-06-08-at-11.00.24-pm.png)You could see that there's a pattern in the news links. If we select every hyperlinks that has an **attribute of "data-outbound-url"** with a **value containing "out.reddit.com"**, we could get hold of all reddit news links!

Step 5: Use XPath to scrape the links, and store the links into an array for further analysis.

\[php\] $redditLinks = array(); $xpath = new DOMXPath($dom); $urls = $xpath->query('//a\[contains(@data-outbound-url,"out.reddit.com")\]/@href'); foreach($urls as $url) array\_push($redditLinks,$url->nodeValue); \[/php\]

So there you have it! All the reddit news links in $redditLinks array. You could go on further by iterating through this array and parse the content and more. Rinse and repeat, the same techniques applies.

Before you jump into scraping real websites, there are things that you should take note of, to avoid detection from the targeted website and prolong the longevity of your script:

1) Websites generally don't like web scraping bots. Most are friendly and don't put any gateway to protect it, but some spent tons of money to protect their content because they deserve it. Be sensible about how aggressive your scraping is. 2) Take effort to spoof the user agent of the script. Notice that the line 2 of the first chunk of codes is somewhere that you could set a custom user agent. Find a [list of common user agents](http://www.browser-info.net/useragents) used by normal(non-bot) browsers and use their user agent string to pretend to be them. This would avoid being detected as bots. 3) Set a timer/interval of the frequency of scraping a website. Scripts could go very fast-about 10 pages or so in a blink of an eye-and you shouldn't add too much extra load to the website's server. After all, you don't want it to break, do you? So be nice, and throttle your speed of fetching pages and resources.

**Dealing with dynamic websites**

The above techniques only applies if your targeted webpage is static. If the website is dynamic, or to be specific, the data that you want to scrape is:

1) loaded by AJAX which triggers only when an actual web browser with javascript-enabled loads the webpage 2) or loaded only when you do some mouse events or manual actions(eg. scroll down)

Brace yourself for more coding and time.

A good example of such website is [http://www.gap.com/browse/category.do?cid=65179&departmentRedirect=true#pageId=0&department=136](http://www.gap.com/browse/category.do?cid=65179&departmentRedirect=true#pageId=0&department=136) Go ahead and click it, it's SFW ;)

As you can see, the "product boxes" only appear about a couple of seconds after you load the page. It's probably designed to load a modest X number of products only after the rest of the page has been completed and this is a classic case of AJAX lazy loading for performance issue.

On top of that, some of the product boxes are only loaded when you scroll down and to grab the chunk of text in the box, you would need to simulate the scrolling down mouse event for the data to appear before scraping it.

Before you thought that playtime is over, this is when fun-time becomes serious work, and we have to call in a bigger brother - Selenium. We'll talk about how we could use Selenium to stimulate a real browser load of the webpage and eventually scrape any data you want in the next post!

Thanks for reading and remember to share the post if you find it helpful!
