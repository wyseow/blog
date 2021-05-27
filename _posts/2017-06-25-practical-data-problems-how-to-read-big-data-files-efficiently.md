---
title: "Practical Data Problems - How to read big data files efficiently"
date: "2017-06-25"
categories: 
  - "data-preprocessing"
coverImage: "/post_images/500521-how-to-send-large-files-2.jpg"
---

Panda's read\_table or read\_csv is probably the number 1 method that comes to everyone's mind when you need to read the rows of data into dataframe. After all, you could do that in just 2 lines:

\[python\] import pandas as pd data = pd.read\_table('filename.txt') \[/python\]

Neat huh? These 2 simple lines would go work well with many cases.

But, guess what happens if you attempt to read a file that has at least millions of rows, and over GBs in file size, which is pretty common in production data. This is what you will see:

\[caption id="attachment\_88" align="alignnone" width="512"\]![](/post_images/ucl3.png) Uh oh..it's the nasty out of memory error again...\[/caption\]

Depending on the hardware spec of the machine you are running it on, you would see this message at some point of loading the data. It took about 30+ seconds for me on a Macbook pro running on intel core i5 with 4GB ram.

Hitting this message in the output is a rude awakening that it's time we have to do play a few tricks to load our big data file efficiently. So how do we go about doing it?

**Reading just part of the data - Attempt #1**

Perhaps you don't need the full 5451020 lines in your dataframe? If you just doing some analysis and want to get a feel of the data, you could limit the function to just read X rows using the "nrows" parameter. In this example, we are reading just 200 rows:

\[python\] import pandas as pd data = pd.read\_table('filename.txt',nrows=200) data.shape #Prints (200, 17) \[/python\]

There you go, 200 samples of the big data file that crashes your machine. However, this method will only fetches the first X rows and it introduces other problems. The first X rows might not be representative of the whole dataset in terms of distributions, mean and other important statistical matrix.

**Reading just part of the data - Attempt #2**

We could also take in big data file chunk by chunk, in a memory-friendly size, and combine it together into a dataframe at the end. This is where we use the "chunksize" parameter. I also use the time library and some time methods to track the amount of time it takes to execute my reading of big data files.

\[python\] import pandas as pd import time #num of lines to read at one time chunksize = 100 chunks=\[\]

start = time.time()

for chunk in pd.read\_table('filename.txt', chunksize=chunksize): chunks.append(chunk) data = pd.concat(chunks, ignore\_index=True,axis=0) print(data.shape) #Prints (5200000, 18)

end = time.time() print(end - start) #Prints 340.2000164985657 \[/python\]

It takes a longer time at 340 seconds, but at the very least, you have the entire dataset in your dataframe! All 520000 rows in my case.

When you specify the chunksize parameter, the function returns a iterator where each iteration is a chunk of X rows that you have specified. You would store X rows in an array, and then concatenate all the X rows together into a dataframe **only after** you have finish reading the file.

It's important to note that you do not perform a dataframe concatenation at every iteration of reading the file because each "pd.concat" operation actually creates an additional dataframe to copy the new one into the old one, causing a N^2 operation.

**Extra tips on handling big data files**

![](/post_images/Extra1-923x1024-270x300.gif)

If you are reading a flat file, chances are that the columns are separated by a delimiter. For csv files, it's usually a tab. For some others, it could be a "|" or something. Instead of separating the data into columns by the delimiter after loading the dataframe, do this to separate the data into the columns **as it is loaded** into the dataframe. It saves you 1 step of intensive computation:

\[python\] for chunk in pd.read\_table('filename.txt', chunksize=chunksize, delimiter='|') \[/python\]

By specifying the delimiter, each row of data is separated into the columns when loaded into the dataframe.

However, when you do that, you might run into another issue. What if the data in one of the column has this delimiter? That would unexpectedly create another column of data isn't it? In that case you would run into this error:

\[python\] ValueError: Expecting 18 columns, got 19 in row 3476575 \[/python\]

The short answer is to skip the problematic rows by using the parameter "error\_bad\_lines":

\[python\] for chunk in pd.read\_table('filename.txt', chunksize=chunksize,delimiter='|', error\_bad\_lines=False): \[/python\]

And now it will just inform you that it has skipped line X:

\[python\] b'Skipping line 3476575: expected 18 fields, saw 19\\n' \[/python\]

So that's all we got for now! If you have some helpful tricks that make everyone's lives easier with loading big data files for data science projects, feel free to share below!
