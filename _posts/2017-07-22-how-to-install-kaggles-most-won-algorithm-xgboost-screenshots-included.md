---
title: "How to install Kaggle's Most Won Algorithm - XGBoost (Screenshots included)"
date: "2017-07-22"
categories: 
  - "machine-learning"
coverImage: "/post_images/omega_weapon_by_htanjo-d6u79cf-2-4.jpg"
---

If you are on this page, chances are you have heard of the incredible capability of XGBoost. Not only it "boasts" higher accuracy compared to similar boasted tree algorithms like GBM (Gradient Descent Machine), thanks to a more regularized model formalization to control over-fitting, it enables many Kaggle Masters to win Kaggle competitions as well. In fact, it's probably the most popular machine learning algorithm at the data science space right now!

Today we shall see how you can install the XGBoost library in your workspace to start using it for your data science project or even Kaggle competition!

You might be thinking that we can't we just download from the Anaconda repo list by doing a one-liner like this as suggested by many websites:

\[php\] conda install -c anaconda py-xgboost=0.60 \[/php\]

Moreover, it seems to work in all platforms! Be it windows(32 or 64 bits), linux, OSX and you don't need to deal with the frustrating libraries dependencies during installation.

But wait! It works perfectly until you are in the thick of the action. Depending on your dataset or structure of sparse matrix, you might hit all sorts of error messages like "feature name mismatches", blah blah, and this is a good example that I have hit on for several days:

\[php\] ValueError: feature\_names mismatch: \['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', ... f38732', 'f38733', 'f38734', 'f38735', 'f38736', 'f38737', 'f38738', 'f38739'\] \[\] expected f4057, f36350, f1683, f1914, f33121, f16637, f21443, f10995, f36221, f24340, f15968, f7863, f38732, ... f19897, f33500, f37792, f30259, f20094, f27943, f5788, f14369, f9074 in input data \[/php\]

If you look at the issue tracker at GitHub, I'm not the only one: https://github.com/dmlc/xgboost/issues/1238

That's the best reason why we are building the latest XGBoost library on scratch. The latest version fixes all the nasty bugs till date and although you have to put in a few good hours to build the library, it's better than stumbling on a bug and backtracking all the way back.

So make yourself a good cup of coffee, put on your geeky glasses and do these step by step...

**Installing on OSX**

1) Get Homebrew

This is a very useful open source installer that contains all the nifty tool you need to install libraries, which you will need it later to build the XGBoost files. Fortunately, Installing is straightforward.

Simply open a terminal, then paste and execute the instruction available on Homebrew home page, such as:

\[php\] /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" \[/php\]

You should see the following: ![](/post_images/Screen-Shot-2017-07-15-at-11.02.11-AM-1-1024x718.png)

2) Get GCC Compiler with Multi threading enabled. **This is important** as you will want XGBoost to take advantage of your quad/whatever core in your CPU and run your fitting/prediction as soon as possible. The following commond will download, and build GCC. Be warned that it's going to take a long time so grab a coffee while you are waiting.

\[php\] brew install gcc --without-multilib \[/php\]

If you see the following screens while waiting, chances are you are on track! ![](/post_images/Screen-Shot-2017-07-15-at-11.02.48-AM-1024x718.png)

![](/post_images/Screen-Shot-2017-07-15-at-12.28.38-PM-1024x718.png) 3) Finally, Get XGBOOST! Go to the directory you want it to be downloaded (cd command), then type the git clone command and execute it. This would download the latest build from the official XGBoost GitHub repo:

\[php\] cd <directory> git clone --recursive https://github.com/dmlc/xgboost \[/php\]

Note: It creates a directory named xgboost inside your directory. ![](/post_images/Screen-Shot-2017-07-16-at-12.04.07-AM-1024x718.png)

4) Find out your GCC compiler version as we will be putting the version number into the XGBoost config file:

\[php\]which gcc-7\[/php\]

OR try which version is it being installed by testing with different version number ![](/post_images/Screen-Shot-2017-07-16-at-12.28.17-AM-1024x494.png) In my case, I have version 7 (gcc-7). Now that you know your GCC compiler version. Let's put the version number in your config file.

5) Open make/config.mk

\[php\] vi make/config.mk \[/php\]

6) And uncomment these two lines.

\[php\] export CC = gcc export CXX = g++ \[/php\]

7) Swap in your GCC Compiler version number:

\[php\] export CC = gcc-7 export CXX = g++-7 \[/php\]

It should look like this: ![](/post_images/Screen-Shot-2017-07-16-at-2.07.35-PM-1024x718.png) 8)Copy the config file to your main xgboost directory for building:

\[php\] cd <directory>/xgboost cp make/config.mk . \[/php\]

9) Finally, it's time to build our directory:

\[php\]make -j4\[/php\]

You should see something like this: ![](/post_images/Screen-Shot-2017-07-16-at-12.26.42-AM-1024x718.png) ![](/post_images/Screen-Shot-2017-07-16-at-12.28.25-AM-1024x718.png) 10) Now that the build is done, we can use install the XGBoost onto your Python package:

\[php\] cd python-package; sudo python setup.py install \[/php\]

![](/post_images/Screen-Shot-2017-07-16-at-12.29.12-AM-1024x718.png) **If you see the above message, you are done! Congratulations! You are on your way to an award winning machine learning algorithm!**

Now, go ahead and head to your Jupyter Notebook and do a:

\[php\] import xgboost as xib \[/php\]

And you shouldn't see any error. That means the installation is successful!

 

**Installing on Windows**

For windows, we need more work to get build the XGBoost. I think I probably spent like 1-1.5 hours to get it done, so grab a coffee or two if you need to.

1) Download and install Visual Studio Community version which is free! For my case, you can find the installation [here](https://www.visualstudio.com/downloads/). Ensure that you have chosen all the C++ build tools during your installation. In this example I have installed Visual Studio 2017.

2) Download and install CMake [here](https://cmake.org/download/). Unzip/extract to a directory to your choice and ensure there's a **cmake.exe** being extracted.

3) Download Git for Windows [here](https://git-for-windows.github.io/) and run the installation. Simply run through the installation with default settings(except for the option shown below-select the bash option) and you will do fine. ![](/post_images/Untitled0.png) Once the installation is done, look for a program called **Git Bash** in your start menu or program files.

4) Create a directory where you want to download the XGboost source code to.

Okay! Finally we are done with all the installation and can start with the actual building of the codes!

4) Launch the **Git Bash** and "go" to the directory(you just created above) where you want to download XGBoost source code by typing the cd command in the bash terminal like the following:

\[php\] cd c:/Users/vargeeks/code/ \[/php\]

5) Download the XGBoost source code by typing the following commands one by one:

\[php\] git clone --recursive https://github.com/dmlc/xgboost cd xgboost git submodule init git submodule update \[/php\]

Now that the codes are in, our next step is to build it to generate a dll and exe file.

6) Launch your **Developer Command Prompt** in your Visual Studio folder: ![](/post_images/Untitled2.png)

7) Using the **dev command prompt**, create build folder and "go" into it by typing:

\[php\] mkdir build cd build \[/php\]

8) Use the **cmake.exe** that you have extracted to build the codes:

\[php\] C:\\dev\\cmake-3.6.2-win64-x64\\bin\\cmake.exe .. -G”Visual Studio 15 2017 Win64″ \[/php\]

Note: Replace the cmake.exe path to your one and the replace the Visual Studio version with the one you have installed ![](/post_images/Untitled3.png) 9) And then:

\[php\] msbuild.exe /t:Clean,Rebuild /p:Configuration=Release xgboost.sln \[/php\]

![](/post_images/Untitled4.png) 10) Okay, at this point of time a xgboost.dll and xgboost.exe will be built in different folders(from my experience). Find them and put them into the "python-package" folder like this: ![](/post_images/Untitled5.png) 11) In your **Developer Command Prompt**, "go" to your "python-package" folder:

\[php\] cd ..\\python-package \[/php\]

12) Finally, install the xgboost package that you have built:

\[php\] python setup.py install \[/php\]

Go ahead and give it a try at your Jupyter Notebook. You should be able to import like this:

\[php\] import xgboost as xgb xr = xgb.XGBRegressor() \[/php\]

If you didn't see any errors, congrats! You did it!

The above guide is based on the convergence of several guides and my own successful experience. Let me know below if there's any error when you install using the guide.

Have fun with XGBoost and have a go at the Kaggle competition using your new-found weapon! We'll see how you can push the limit of XGBoost by evaluating the number of iterations during model training and tuning the best hyper-parameters in the next blog post. Stay tuned.
