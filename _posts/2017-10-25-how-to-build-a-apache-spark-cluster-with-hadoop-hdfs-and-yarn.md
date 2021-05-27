---
title: "How to build a Apache Spark Cluster with Hadoop HDFS and YARN"
date: "2017-10-25"
categories: 
  - "hadoop"
  - "spark"
tags: 
  - "cluster"
  - "hadoop"
  - "spark"
  - "yarn"
coverImage: "/post_images/spark_cluster.jpg"
---

In our earlier post, we built a pretty light 2-nodes Apache Spark cluster without using any Hadoop HDFS and YARN underneath. We didn't point the spark installation to any Hadoop distribution or set up any "HADOOP\_HOME" as a PATH environment variable and we have deliberately set the "master" parameter to a spark master node.

Today, instead of using the **Standalone Mode**, which uses a simple cluster manager available as part of the Spark distribution, we are going to use **Apache YARN**, a powerful resource negotiator. So what's the difference?

YARN, being a dedicated resource negotiator has much richer resource scheduling capabilities such as prioritising workload and data locality. Data locality is a very cool idea where the YARN assign containers which execute tasks that are physically close to the nodes which are holding the required data for processing. Also, if you are looking to add other Hadoop applications in your project such as HDFS, Zookeeper and Hive, it's a good start to use YARN as a base resource manager to manage the various application in the Hadoop ecosystem.

Note that these steps are applied on MacOS computers and it would be similar to implement on other Unix/Linux based machines.

This is the architecture that we are attempting to build, for make explaining easier: ![](/post_images/arch-1024x708.jpg) We are just focusing on building a 2-nodes cluster, a master and slave machine. In the above diagram, it shows 3 slave machines.

We will be setting up 1 machine as a master that runs: 1) Resource Manager(for master) 2) Node Manager(for slave) 3) NameNode(for master) 4) DataNode(for slave)

Also, another machine will be set up a slave that runs: 1) Node Manager(for slave) 2) DataNode(for slave)

We are also running the Node Manager and Data Node in the master because we want it to serve both as a master and a slave to share the workload of the incoming jobs.

**Installing Hadoop(YARN and HDFS) on the nodes**

Apply these steps in all machines, regardless master or slave.

**Important**: Maintain the same installation directory path, structure, names and account across all nodes if you don't want to deal with a massive headache later. It happened to me big time.

**#1 - Install Java**

Ensure that you have at least Java 8 JDK installed. If you do a check like this in the terminal:

\[code\] javac -version \[/code\]

You should see this:

\[code\] java version "1.8.x\_xx" Java(TM) SE Runtime Environment (build 1.x.x\_xx-xxx) Java HotSpot(TM) Client VM (build 22.1-b02, mixed mode, sharing) \[/code\]

Download the latest one [here](http://www.oracle.com/technetwork/java/javase/downloads/index.html) if yours is not at least JDK 1.8

**#2 - Creating a Hadoop user** I would advise creating a dedicated Hadoop user to run all the Hadoop related processes as a good practice, security and to avoid user access headaches later.

The following codes will create group "hadoopgrp" and add the user "hadoopuser" to your local machine.

\[code\] sudo addgroup hadoopgrp sudo adduser --ingroup hadoopgrp hadoopuser \[/code\]

**#3 - Install Hadoop**

At the time of this writing, Hadoop 3.0 Beta is available, but I'm using 2.8.1 because it's the latest **stable** release. Head over to the download page [here](http://hadoop.apache.org/releases.html) and extract into "/var/local/hadoop". Remember that Hadoop files have to reside in the same path in all machines so choose a folder name/path that is available in all cluster machines.

Use the following command to change the owner of all the files in folder "hadoop" to the newly created "hadoopuser" user and "hadoopgrp" group:

\[code\] sudo chown -R hadoopuser:hadoopgrp hadoop \[/code\]

**#4 - Setting the PATH right**

We have to make the Hadoop installation known to Spark and other related applications in the machine so let's edit the PATH using the following command:

\[code\] vi ~/.bash\_profile \[/code\]

And make sure it looks like this:

\[code\] #define a HADOOP\_HOME variable export HADOOP\_HOME=/usr/local/hadoop

#add the HADOOP bin folder into PATH export PATH=$PATH:$HADOOP\_HOME/bin

#make sure you have a valid JAVA\_HOME set export JAVA\_HOME=\`/usr/libexec/java\_home\` \[/code\]

If you don't have an existing JAVA\_HOME set, remember to add one here.

**#5 - Creating HDFS Data folders**

We need to create 3 folders for HDFS to store actual data files, in both machines. Again, choose to create in a **common location** like "/usr/local/hadoop/data/"

1) Create folder for NameNode I have created "/usr/local/hadoop/data/namenode"

2) Create folder for DataNode I have created "/usr/local/hadoop/data/datanode"

3) Create temporary folder This is for HDFS to write temporary files. I have created "/usr/local/hadoop/data/tmp" for this purpose.

**#6 - Configuration files**

There are several files to be edited here.

#6.1 - hadoop-env.sh (/usr/local/hadoop/etc/hadoop/) Open up "/usr/local/hadoop/etc/hadoop/hadoop-env.sh" and change the "JAVA\_HOME" to your own java path. It should look something like this:

\[code\] export JAVA\_HOME=\`/usr/libexec/java\_home\` \[/code\]

#6.2 - core-site.xml (/usr/local/hadoop/etc/hadoop/)

In this file, we define the location of the HDFS temp folder which you have just created and the port which HDFS is operating at. I'm leaving most setting as default except for "hadoop.tmp.dir". It should look like this:

\[code\] <property> <name>hadoop.tmp.dir</name> <value>/usr/local/hadoop/data/tmp</value> <description>A base for other temporary directories.</description> </property>

<property> <name>fs.default.name</name> <value>hdfs://localhost:54310</value> <description>The name of the default file system. A URI whose scheme and authority determine the FileSystem implementation. The uri's scheme determines the config property (fs.SCHEME.impl) naming the FileSystem implementation class. The uri's authority is used to determine the host, port, etc. for a filesystem.</description> </property> \[/code\]

#6.3 - hdfs-site.xml(/usr/local/hadoop/etc/hadoop/)

In this file, we'll define the number of machines to replicate to, for every single file in HDFS. It should be the same number as the number of available slave machines(Data nodes). If you happen to set this to a value higher than the number of available slave nodes, you will start seeing "Zero targets found, forbidden1.size=1" type errors in the log files.

Since we have only 2 machines operating as slaves, we'll set dfs.replication to 2.

Remember that we have created 2 separate folders to store NameNode and DataNode data files? It's time to specify them here.

\[code\] <property> <name>dfs.replication</name> <value>2</value> <description>Default block replication. The actual number of replications can be specified when the file is created. The default is used if replication is not specified in create time. </description> </property> <property> <name>dfs.namenode.name.dir</name> <value>/usr/local/hadoop/data/namenode</value> <description>Determines where on the local filesystem the DFS name node should store the name table(fsimage). If this is a comma-delimited list of directories then the name table is replicated in all of the directories, for redundancy. </description> </property> <property> <name>dfs.datanode.name.dir</name> <value>/usr/local/hadoop/data/datanode</value> <description>Determines where on the local filesystem the DFS name node should store the name table(fsimage). If this is a comma-delimited list of directories then the name table is replicated in all of the directories, for redundancy. </description> </property> \[/code\]

#6.4 - yarn-site.xml(/usr/local/hadoop/etc/hadoop/)

Nothing exciting in this file where we just make sure the following configs are inside:

\[code\] <configuration> <property> <name>yarn.nodemanager.aux-services</name> <value>mapreduce\_shuffle</value> </property> </configuration> \[/code\]

As of Spark 2.0, there's no /etc/hadoop/masters file. Therefore you should just ignore if you see some tutorials on the internet mentioning it. Spark assumed the machine which starts the HDFS is the master.

However, you will still need to define the slaves configuration in the /etc/hadoop/slaves file.

**#7 - Naming master and slave** For ease of configuring the master and slave machines, let's assign unqiue hostnames to the machines. For master machine, let's name it "master". And for slave machine, let's name it "slave". I know it's obvious but as far as infrastructure configurations are concerned, it's best to keep things simple.

In your master machine, edit the /etc/hosts file to be like this:

\[code\] 192.168.0.1 master 192.168.0.2 slave \[/code\]

Remember to replace the IP addresses above with your machines IP addresses.

**#8 - Setting up SSH for connectivity between nodes** Hadoop uses password-less SSH connections between nodes for communications. For that to happen, the machines have to generate a pair of SSH keys and share the keys with the machine so that they could talk to each other using the keys.

Note that after we have the keys, we are going use it to connect to **both itself(localhost) and the other machine**.

Login as the hadoopuser:

\[code\] su - hadoopuser \[/code\]

Create a pair of SSH keys with empty password so that we don't need to enter password everytime they interact:

\[code\] ssh-keygen -t rsa -P "" \[/code\]

You should see the following. Just press enter when there's any prompt to save the keys in the default places.

\[code\] Generating public/private rsa key pair. Enter file in which to save the key (/home/hadoopuser/.ssh/id\_rsa): Created directory '/home/hadoopuser/.ssh'. Your identification has been saved in /home/hadoopuser/.ssh/id\_rsa. Your public key has been saved in /home/hadoopuser/.ssh/id\_rsa.pub. The key fingerprint is: 9b:82:ea:58:b4:e0:35:d7:ff:19:66:a6:ef:ae:0e:d2 hadoopuser@... The key's randomart image is: \[...snipp...\] \[/code\]

Add the newly generated key into your(localhost) authorized key list to enable access.

\[code\] cat $HOME/.ssh/id\_rsa.pub >> $HOME/.ssh/authorized\_keys \[/code\]

Try to **connect to itself(localhost)** through access:

\[code\] ssh localhost \[/code\]

You should see this. Type "yes" when prompted to save the fingerprint to the known\_host file. We are triggering the prompt now so that the it will not be prompted when connection is initiated by the Hadoop daemon later on:

\[code\] The authenticity of host 'localhost (::1)' can't be established. RSA key fingerprint is d7:87:25:47:ae:02:00:eb:1d:75:4f:bb:44:f9:36:26. Are you sure you want to continue connecting (yes/no)? yes Warning: Permanently added 'localhost' (RSA) to the list of known hosts. ... ... \[/code\]

If you can see something above, it means you are able to connect to localhost through itself.

Type this to exit the SSH connection:

\[code\] exit \[/code\]

Besides sharing the SSH keys with localhost itself, **go to the master machine** and share the key pair with the slave machine.

The following command will prompt you for the login password for user hadoopuser on slave, then copy the public SSH key to the slave machine.

\[code\] ssh-copy-id -i $HOME/.ssh/id\_rsa.pub hadoopuser@slave \[/code\]

Moving forward, the Hadoop process will establish a SSH connection using the key you have shared to start/stop HDFS and YARN nodes.

Try to connect **from master to slave** using SSH:

\[code\] ssh master \[/code\]

and you should see this if you have successfully shared the SSH key:

\[code\] The authenticity of host 'master (192.168.0.1)' can't be established. RSA key fingerprint is 3b:21:b3:c0:21:5c:7c:54:2f:1e:2d:96:79:eb:7f:95. Are you sure you want to continue connecting (yes/no)? yes Warning: Permanently added 'master' (RSA) to the list of known hosts. ... ... \[/code\]

**#9 Pointing the existing Spark installation to Hadoop**

If you have been following me from the previous Spark installation blog post [here](http://gator4205.temp.domains/~datageeko/how-to-deploy-a-hadoopspark-cluster-with-multiple-machines/), you have to point the existing Spark installation to your Hadoop installation so that Spark will know where(host address) to trigger the start of the YARN Resource Manager when you submit the Spark program through the network.

In your spark-env.sh file (/usr/local/spark/conf/), ensure that your "HADOOP\_CONF\_DIR" is set to your Hadoop configuration folder:

\[code\] export HADOOP\_CONF\_DIR=$HADOOP\_HOME/etc/hadoop \[/code\]

**#10 - Formatting HDFS filesystem**

We are almost there! It's time to format the name node of your HDFS so that it's ready to store and retrieve files. Note that by formatting, you will lose the existing HDFS files if you have any.

The following codes executes the hadoop program and instructs it to format NameNode's file space.

\[code\] ./usr/local/hadoop/bin/hadoop namenode -format \[/code\]

You should see something like this: ![](/post_images/Screen-Shot-2017-10-05-at-1.12.13-PM-1024x718.png) It says something like "...NameNode has been successfully formatted..."

We are finally done with the configuration! Let's test out the cluster by starting the various daemons to make sure our configurations are correctly set.

**#11 - Testing the cluster**

There is a 2-step process to start your cluster. 1) Start the HDFS daemons from the master machine. 2) Start the YARN daemons from the master machine.

**#11.1 - Starting HDFS daemons**

In the **master machine** terminal, run following command:

\[code\] ./usr/local/hadoop/sbin/start-dfs.sh \[/code\]

The command will first start the NameNode on your master machine, and start the DataNodes running in both master and slave machines. Note that we have previously configured to run both NameNode and Datanode in our master machine.

You should see these results: ![](/post_images/Screen-Shot-2017-10-05-at-1.55.05-PM-1024x718.png)

How do we know whether it's working?

If you execute this in your **master machine**, you should see NameNode and DataNode daemon running:

\[code\] jps \[/code\]

\[code\] 14799 NameNode 15314 Jps 14880 DataNode 14977 SecondaryNameNode \[/code\]

Most importantly, if you check it out on the **slave machine**, the DataNode daemon is also started!

\[code\] jps \[/code\]

\[code\] 15183 DataNode 15616 Jps \[/code\]

Let's try to put something inside the HDFS and see if the replication is working. You wouldn't want your file to reside in just 1 machine when we have a cluster for that.

Find any testing file in your computer and execute this:

\[code\] ./usr/local/hadoop/bin/hadoop dfs -copyFromLocal /home/documents/yourfile.png yourfile.png \[/code\]

What this does is that it copies your file from "/home/documents/yourfile.png" to the main directory in DataNode, as specified in "dfs.datanode.name.dir" setting. While it doesn't translate directly to the hierarchy as you expected when you navigate to your DataNode folder, you can still see the files using the utility tools described below.

There's a Web UI of the NameNode which you access from the master machine: http://localhost:50070/ ![](/post_images/Screen-Shot-2017-10-05-at-1.56.43-PM-1-1024x825.png)

This web UI page is useful for checking the health of all your nodes. If you click on the "Datanode" menu at the top, you will see all your running Datanodes: ![](/post_images/Screen-Shot-2017-10-06-at-9.58.31-AM-1024x828.png)

You should see both your master and slave machines inside the list, along with some information like total space, etc.

You can also click on the "Browse directory" in the "Utilities" tab to see all your files in the HDFS system. Since we have set the replication factor to 2, which means files will be replicated to both master and slave machine, you will see "2" under the replication column. ![](/post_images/Screen-Shot-2017-10-06-at-10.01.41-AM-1024x829.png)

If you could see the results described above, great! Looks like your master machine can talk to slave machine very well!

If the DataNode couldn't be started in your slave machine, just head over to the "logs/hadoop-hduser-datanode-slave.log" file in your slave machine. It would give you a hint on what's going on.

Here are some other things to check for: 1) Check the SSH configurations to make sure they can establish connection password-less 2) Check the permission of the HDFS folder in your slave machine. Make sure it's writable by the connecting account. 3) Check the host mapping files again to make sure the IP address is correct. If you have dynamic IP addresses, it could be changed everytime your machine connects to the internet.

**#11.2 - Starting YARN daemons**

Let's move on to start the YARN daemons.

If you are using Linux/Unix systems like Ubuntu, you could start the YARN Resource Manager from the master machine:

\[code\] ./usr/local/hadoop/sbin/start-yarn.sh \[/code\]

But if you are using MacOS like me, there's a bug that keeps resetting the JAVA\_HOME path when the "start-yarn.sh" script is instructing all the YARN Node Manager in the cluster to start. So, you will need to manually start the YARN resource and node managers manually.

So, execute this instead if you are using MacOS. In master machine:

\[code\] ./usr/local/hadoop/sbin/yarn-daemon.sh start resourcemanager ./usr/local/hadoop/sbin/yarn-daemon.sh start nodemanager \[/code\]

And in your slave machine:

\[code\] ./usr/local/hadoop/sbin/yarn-daemon.sh start nodemanager \[/code\]

Again, run the "jps" command to check whether the processes are running:

\[code\] jps \[/code\]

\[code\] 14880 DataNode 14660 ResourceManager 14977 SecondaryNameNode 14799 NameNode 14765 NodeManager 15314 Jps \[/code\]

In your master machine, you should see 2 additional processes, ResourceManager and NodeManager

\[code\] jps \[/code\]

\[code\] 15183 DataNode 15616 Jps 14760 NodeManager \[/code\]

In your slave machine, you should see just 1 additional process, NodeManager

YARN also comes with a web UI interface for the ResourceManager. It can be accessed here: http://localhost:8088/

You would probably see a screen like this, that shows the 2 active nodes and some of the health statistics. ![](/post_images/Screen-Shot-2017-10-06-at-11.19.51-AM-1024x299.png)

To recap, we have built and started almost all the services in our planned architecture: ![](/post_images/arch-1024x708.jpg)

**#12 - Running Spark program on our Hadoop YARN and HDFS cluster**

Finally, let's test our cluster by running an example Spark program and we shall see how every component in the Hadoop ecosystem work together to execute a Spark program.

Modify the over-used Spark Pi example from the official site and add a few more iteration to simulate a longer running time:

\[python\] import pyspark as ps import random

spark = ps.sql.SparkSession.builder \\ .appName("rdd test") \\ .getOrCreate()

random.seed(1)

def sample(p): x, y = random.random(), random.random() return 1 if x\*x + y\*y < 1 else 0

for x in range(0,10): count = spark.sparkContext.parallelize(range(0, 10000000)).map(sample) \\ .reduce(lambda a, b: a + b)

print("Pi is (very) roughly {}".format(4.0 \* count / 10000000))

\[/python\]

Next, we will run the Spark program using spark-submit just like last tutorial, except that we are going to explicitly specify to run using our yarn cluster by passing the value "yarn" in the "master" parameter

\[code\] ./usr/local/spark/bin/spark-submit --master yarn test.py \[/code\]

So what is happening behind the hood when you run that command?

Let's take it step by step...

Firstly, because we didn't specify the "deploy-mode", it's defaulted to "client", which means that our terminal serves as the driver when we submit the application to our cluster.The driver is a processing unit that coordinates, builds DAG computation and do some processing needed to complete the Spark program.

But wait, what's the difference in the various deploy mode? Many who are new to Spark get confused with this setting and "--master" setting easily. They are actually entirely different. Basically, there are 2 deploy modes: Standalone and Cluster.

In Standalone mode, the machine that triggers the process is assumed to be the driver, which is responsible for talking to the YARN Resource manager to get available resources and coordinate the tasks between the executors. Since the machine itself is involved in the whole running process, when the machine loses network connection, or shuts down, the running process will be disrupted. On that other hand, the machine would be able to monitor the running process through an interactive shell. Many people refer to this setup as "running out of the cluster".

In cluster mode, the machine that triggers the process communicates to YARN Resource Manager, which in turn issues a YARN container to host as the driver. From here, the machine which triggers the process is no longer connected to the cluster because the process is coordinated by the driver in the YARN container.

Note that we assumed we have used YARN in our cluster instead of other solutions such as Mesos. Technically, deploy mode is configured through "--deploy-mode".

Back to the application submission process...

The driver contacts the YARN resource manager for workers to run the program. To improve performance through data locality, the YARN manager is smart enough to find worker nodes that resides together with the HDFS data blocks that contains the files used by the program. If it cannot find such worker nodes, it will find any worker nodes, and move the required DFS data blocks to that worker node.

After workers nodes have been identified, YARN Resource Manager contacts the Node Manager in the worker nodes to create a YARN container, which in turn run a Spark executor. These identified executors will now receive instructions directly from the Spark driver. ![](/post_images/spark-yarn-f1.png)

Finally, as the driver process the program that you have written, it breaks down certain tasks and it instructs the executors to run them. Note that the driver also run some codes, besides doing the coordinating work. Therefore, it's best that we write Spark program in a way that minimise operations in Driver, and maximise operations in executors.

Generally speaking, transformations like map, filter, flatMap are processed by executors, where combine-like operations like reduce and aggregate are processed in driver. This is a overly-simplification example as complex operations such as Machine Learning and SparkSQL could interleave between driver and executor processing

Okay, back to our Spark program!

If you look at the Resource Manager via the web UI, you could see that it's running 1 application: ![](/post_images/Screen-Shot-2017-10-06-at-11.19.51-AM-1024x299.png)

You could even click on the Application name to delve deeper: ![](/post_images/Screen-Shot-2017-10-06-at-11.20.01-AM-1024x528.png)

But to find out which executors are processing the tasks, you have to fire up the Spark Web UI, which is alive as long as the program is being executed. The address is: http://localhost:4040 ![](/post_images/Screen-Shot-2017-10-07-at-12.29.03-AM-1024x572.png)

In this screen, you could see the execution timeline of this Spark program. You could see that there are a few reduce operations, and as we have expected, the operations have been tasked to both executors.

If you click on the "Executors" tab, you could see the health stats and technical details of the executors and the driver. As you can see below, we have a total of 3 executors/nodes running (1 driver and 2 executors). ![](/post_images/Screen-Shot-2017-10-07-at-12.33.30-AM-1024x593.png)

If you are experiencing problems with the executor executing the task, you could click on the "stdout" link to read the error log without physically entering the executor machine and digging out the logs. I personally find this feature very convenient for troubleshooting cluster issues.

When the Spark program has finished running successfully, go back to the YARN Resource Manager Web UI page, and you should see a "SUCCEEDED" status for that application, instead of the initial "ACCEPTED".

**#13 - How to stop the cluster**

I almost forgotten to write this as I was carried away with firing the cluster up! To stop the cluster, execute the startup workflow in the opposite sequence:

1) Stop the YARN daemons on the master and slave machines. 2) Stop the HDFS daemons on the master machine.

**#13.1 - Stopping YARN daemons**

As mentioned previously, if you are using MacOS, you have to stop the YARN daemon on master and slave machines separately: In master machine:

\[code\] ./usr/local/hadoop/sbin/yarn-daemon.sh stop resourcemanager ./usr/local/hadoop/sbin/yarn-daemon.sh stop nodemanager \[/code\]

In slave machine:

\[code\] ./usr/local/hadoop/sbin/yarn-daemon.sh stop nodemanager \[/code\]

or if you are using Linux/Unix systems like Ubuntu, you could simply stop them using the stop-yarn.sh script: Only in master machine:

\[code\] ./usr/local/hadoop/sbin/stop-yarn.sh \[/code\]

**#13.2 - Stopping HDFS daemons**

To stop all the HDFS daemons, simply issue the following command in master machine:

\[code\] ./usr/local/hadoop/sbin/stop-dfs.sh \[/code\]

And that entails a complete loop of a Spark cluster integrated with Hadoop YARN and HDFS! There are still tons of bells and whistles to configure and this set up is just a basic infrastructure to illustrate a simple Spark program execution through a cluster environment.

I hope this guide is helpful for your use case and let me know below if you hit any issues!
