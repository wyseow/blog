---
title: "How to deploy a Hadoop/Spark Cluster with multiple machines"
date: "2017-08-03"
categories: 
  - "hadoop"
  - "spark"
tags: 
  - "hadoop"
  - "spark"
coverImage: "/post_images/hadoop_elephant.jpg"
---

When you take your machine learning models to the production level, especially in an enterprise setting, you will need your models to give you a fast and reliable response. And this is where Spark comes into the picture. Spark offers a reliable distributed/clustered computing framework that sits on top of the Hadoop framework and if you go the extra mile of configuring the HDFS and YARN, it can even achieve even more resiliency in your product. To start things small, let's start with Spark and we'll see how the other components fit in.

**Installing Spark on Mac/Unix based machine**

Note: I assume that you have an existing Python installation.

Step 1: Go to the official spark download page [here](http://spark.apache.org/downloads.html) and download the .tgz distribution: ![](/post_images/Screen-Shot-2017-07-30-at-10.48.21-PM-1024x330.png)

**Note:** We are selecting the pre-built Hadoop because we don't want to install the entire Hadoop framework just because of Spark. Of course, if you are satisfied with Spark's capabilities and want to take advantage of Hadoop's modules in enhancing your existing Spark installation, you can always install Hadoop later on and just "point" your Spark installation to the entire Hadoop installation. Anyway, selecting this option would allow Spark to install a few Hadoop files which are just enough to let Spark talk nicely with it and get things going.

Step 2: Extract the tgz file into a directory of your choice and rename the folder into a more readable name. For me, this is my spark directory:

\[php\] /Users/machineA/Documents/spark \[/php\]

Step 3: Fire up your terminal and edit your ~/.bash\_profile to add the Spark directory to your classpath:

\[php\] vi ~/.bash\_profile \[/php\]

Step 4: Add the following lines:

\[php\] export SPARK\_HOME=/Users/machineA/Documents/spark export PATH=$SPARK\_HOME/bin:$PATH \[/php\]

Replace the spark directory path with your own.

Step 5: Save the changes by executing:

\[php\] source ~/.bash\_profile \[/php\]

Okay! In 5 Steps, you have basically completed Hadoop/Spark installation! You have added the Spark directory into your classpath and now we are ready to give the sparkling new Spark a try!

Start pyspark by typing:

\[php\]pyspark\[/php\]

If your classpath has been set correctly, your machine will automatically load the pyspark interactive shell in your Spark installation. You should see this: ![](/post_images/Screen-Shot-2017-07-30-at-11.09.39-PM-1024x718.png) If you can see this artwork(I wonder how long does it take them draw this), it means Spark can be successfully run on your machine!

You can type some stuff in this interactive shell to interact with the Spark framework, including submitting a Spark job(we will come to that later). A simple one is to query the Spark version:

\[php\]sc.version\[/php\]

It returns a "2.2.0" for me.

What you have done is an installation of Spark on just 1 machine, and that doesn't sound very exciting as a cluster, isn't it? Let's see how we can hook a few machines together to build a true Spark cluster that could process all your exciting machine learning tasks in parallel!

In this example, I have also installed Spark on another machine, machine B. So we have machine A and B now. Let's treat the machine A as the master, and machine B as the slave. In production scenarios, there could be many more machines like machine B that serves like the slaves.

The general idea is that the master machine primarily orchestrates, splits and collect back the tasks from all the slave machines. There's a lot of technical details on who-do-what(we will come to that) but that is the general picture.

Let's see how we connect and start the Spark cluster with master and slave machines.

Step 1: Start the master in machine A:

\[php\] cd /Users/machineA/Documents/spark/sbin ./start-master.sh \[/php\]

You shouldn't see any error message. When the master has started, Spark will create a monitoring page at port 8080 by default.

Fire up your bowser, navigate to "http://localhost:8080" and you should see some details of your master machine like the Spark Master URL that your slaves can connect to. The URL should be in the form of: spark://masterhostname:7070

Step 2: Start the slave in machine B and connect to the master in machine A:

\[php\] cd /Users/machineA/Documents/spark/sbin ./start-slave.sh spark://masterhostname:7070 \[/php\]

The first parameter in the script is the Spark Master URL-which you have just acquired-of the master machine you are connecting, to form part of the cluster.

Give it a while and refresh the monitoring page at your master machine. You should see the connected slaves such as: ![](/post_images/Screen-Shot-2017-07-29-at-1.28.55-am-1024x388.png) If you can see the above, that means now you have a fully connected cluster (multiple machines) across the network! In your production setting, you can add more slaves to your master machine in similar manner.

So how do you give some work to the slaves? We'll use "spark-submit" program to send the job to the master, and let the master distribute the tasks to the workers in the cluster.

Step 1: Create a test file, **test.py** with the content below, in your master machine(machine A):

\[python\] import pyspark as ps import random

spark = ps.sql.SparkSession.builder \\ .appName("rdd test") \\ .getOrCreate()

random.seed(1)

def sample(p): x, y = random.random(), random.random() return 1 if x\*x + y\*y < 1 else 0

count = spark.sparkContext.parallelize(range(0, 10000000)).map(sample) \\ .reduce(lambda a, b: a + b)

print("Pi is (very) roughly {}".format(4.0 \* count / 10000000)) \[/python\]

Step 2: From the terminal of your master machine, submit the application/job(test.py) to your master machine by using spark-submit:

\[php\] spark-submit --master spark://masterhostname:7070 /path/to/test.py \[/php\]

We are configuring on the fly to submit the job to the master at the indicated Spark Master URL and the path of the python file to execute. The program will start a series of process which consists of things like re-checking all the connected slaves in the cluster, splitting up the tasks in stages, send the application files and dependencies to all the worker machine, disseminate the tasks down to every worker and finally collecting back the results from the workers.

If the above execution is successfully, you should see the following message:

\[php\]Pi is (very) roughly 3.141317\[/php\]

Actually, when the execution is on-going, you can visit your master machine monitoring page(http://localhost:8080) and you could see that the allocation of resources of your worker machines on the running application. Something like this: ![](/post_images/RunningaSparkapplicationinStandaloneMode2-1024x297.jpg)

Okay! We are all done! You have a fully connected Spark cluster now and from here, there are a few things you could explore: 1) Tweaking the memory settings of your worker machines to make sure you are not under-utilizing your resources. 2) Look into implementing Hadoop's HDFS to provide a clustered file system for all your cluster machine. 3) And use even more modules in the Hadoop ecosystem to make your machine learning cluster setup even stronger and reliable!

If you are installing a full-blown Hadoop installation, remember to add this in your .bash\_profile to point your Spark to Hadoop directory so that they can play nice together:

\[php\] export HADOOP\_HOME=/usr/local/hadoop export LD\_LIBRARY\_PATH=$HADOOP\_HOME/lib/native/:$LD\_LIBRARY\_PATH \[/php\]

 

**When things do not work out :(** If the above installation steps are not working well for you, take a look at the scenarios below.

**Problem: Unable to start master using "start-master.sh"**

When you execute this command:

\[php\] ./start-master.sh \[/php\]

it gives you something like:

\[php\] failed to launch: nice -n 0 /your-spark-path/spark-class org.apache.spark.deploy.master.Master --host your-master-host --port 7077 --web-ui-port 8080 \[/php\]

**Solution:** Chances are the ports that Spark is trying to open (7077 and 8080) when launching the master, are being used by existing applications. I know quite a few web servers used port 8080 as well so that might be the case. Assign new ports by explicitly override the default value in your start-master.sh command like this:

\[php\] ./start-master.sh --web-ui-port:8888 \[/php\]

**Problem: Worker machines cannot connect to master or vice versa** This is a common problem in corporate or enterprise setting as the servers have undergone security hardening, which is a process where they block all applications, implement strict user access rights and most importantly, block all the ports leaving a few essential ones open. Remember that when it comes to enterprise servers and machines, it's always whitelist instead of blacklist for all ports and applications.

**Solution: Check and open port if necessary** Note: For unix based machines only

To check whether a port(e.g. 8080) has been blocked or not, enter this:

\[php\] iptables -nL | grep 8080 \[/php\]

If this port is not allowed to open or accepted in the IP Tables, nothing will be returned.

If the port is allowed to open or accepted, you should see this:

\[php\] ACCEPT tcp -- 0.0.0.0/0 0.0.0.0/0 tcp dpt:8080 \[/php\]

To allow a port(e.g. 8888) into the IP Tables, do this:

\[php\] sudo /sbin/iptables -I INPUT -p tcp -m -tcp -m tcp --dport 8888 -j ACCEPT \[/php\]

Test again, shut down the slave/master and start them up again to test the connectivity. If it works, do this to save the changes you just did:

\[php\]sudo /sbin/service iptables save\[/php\]

Here's a list of ports used by Spark master and workers at one glance: ![](/post_images/Screen-Shot-2017-08-03-at-9.37.24-PM-1024x860.png)

 

I hope these installation steps have been helpful to you, and hopefully in the near future you could take advantage of this powerful distributed computing capability offered by Spark!

Tip: Look into [MLLib library](https://spark.apache.org/mllib/) that is part of the Spark framework, because there is a good chance that they have a "scalable version" of your machine learning algorithm that you are about to use. It will fit directly with Spark's API and there are tons of machine learning algorithms and utilities implemented in the library and ready to work in just a few lines.

Let me know if the steps didn't work well, or comment below if you have something that might help other readers as well. :)
