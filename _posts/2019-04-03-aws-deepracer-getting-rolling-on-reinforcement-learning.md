---
title: "AWS DeepRacer - Getting Rolling on Reinforcement Learning"
date: "2019-04-03"
categories: 
  - "reinforcement-learning"
coverImage: "/post_images/deepracer.jpg"
---


### What is the AWS DeepRacer?

According to AWS: “AWS DeepRacer is the fastest way to get rolling with machine learning, literally. Get hands-on with a fully autonomous 1/18th scale race car driven by reinforcement learning, 3D racing simulator, and global racing league.”

Here what I think it is: It presents a wonderful opportunity for us to get our wet feet into reinforcement learning in a fun way.

 

### Why learn reinforcement learning?

Reinforcement learning (RL) has many uses beyond cool robotics projects; there is an increased application of RL in NLP, financial trading, and various other domains that are traditionally trained using supervised learning. The biggest attraction of RL is that it enables models to learn complex behaviour without labeled data.

### Main building blocks of RL

Let's take a look at how reinforcement learning works by using the AWS DeepRacer as an example.

![](/post_images/Screen-Shot-2019-04-03-at-9.35.56-PM-1024x330.png)

RL is about creating a model that can be used by an agent to choose which actions to take in an environment in order to achieve a specific goal. To put things into context, the **agent** is the DeepRacer car, the **environment** is the race track and what the car can see through its camera, **actions** are steering and throttle used to control the car, **goal** is obviously to complete a lap as quickly as possible without going out of the track.

### How does a RL model learn?

With these key components of RL in mind, let's see how they work together to train a RL model.

![](/post_images/Screen-Shot-2019-04-03-at-9.47.33-PM-1024x508.png)

The camera at the front of the car captures the images of the front track when it's moving, and these images represent the **state** of the **environment.** The **agent** (DeepRacer) uses the **state**(images) and the **model** to decide which **action** to choose(steer left/right or speed up). When this action interacts with the environment, the agent receives the **reward** and updated state(images). The model in the agent learns from these response and the whole cycle repeats.

At the beginning, the untrained reinforcement model will enter an **exploration** stage where it choose actions at random to explore the environment but over time it will be trained and notice which actions lead to better outcomes. Then it will gradually enter an **exploitation** stage which starts to exploit some of the knowledge that it's gained to repeatedly take actions that lead to better outcomes. In practice, the speed of transition between exploration and exploitation is a hyperparamter to tune and we should be well familiar with the trade-offs, just like any other ML algorithms.

### Which action to take?

At each timestep, the model is going to choose the action that it thinks will lead to the eventual outcome with the highest reward. As a result of choosing that action, the model gets in return a reward which quantify how good or bad an outcome is. Every action doesn't necessarily returns a reward; in some cases, an agent needs to perform a series of steps to finally receive a +1 reward and that is the true advantage of reinforcement learning.

All of these experiences obtained will be used to update the model, by learning which actions in each state will lead to the maximized cumulative reward. We could imagine that as a lookup table:

### ![](/post_images/1f8PhlUiGYVv_FJcgyFNGfw.png)

This forms the basis of q-learning and q-table, which we will talk more in-depth in future post. But you get the idea.

Although maintaining this lookup table to retrieve the maximum cumulative reward for a given state seems easy, it's not possible to explore every state action combination in some scenarios such as continuous actions, or steering a wheel by 1 degree, 2.5 degree, and so on.

To overcome this problem, we could approximate the value function by using a method called Policy Gradient, which we will be talking in a future post.

### Training the DeepRacer

Let's get back to the AWS DeepRacer. We don't need to put the RL model physically into the DeepRacer; that would slow down the whole training/evaluation cycle.

AWS recommends everyone to use the DeepRacer console as an end-to-end platform to train, evaluate and simulate RL models.

While we do not have access to the DeepRacer console as it required a whitelist access and many people have reported waiting for very long for it, we can look at some screenshots of the platform to see how it works.

\[caption id="attachment\_451" align="alignleft" width="412"\]![](/post_images/Screen-Shot-2019-04-04-at-10.23.53-AM-1024x743.png) It offers creation of a RL model from a pretrained one.\[/caption\]

 

 

 

\[caption id="attachment\_452" align="alignnone" width="431"\]![](/post_images/Screen-Shot-2019-04-04-at-10.27.13-AM-1024x716.png) We could select the environment to simulate in.\[/caption\]

<iframe src="https://www.youtube.com/embed/LxIp9Vl1ZmQ" width="560" height="315" frameborder="0" allowfullscreen="allowfullscreen"></iframe>

 Interactive training process where we could see the model's ability to take action that leads to higher cumulative rewards and a simulation of the environment.

### Using reward to incentivise correct driving behaviour

One thing we need to configure here is the reward function. Remember that reward drives the decision of the action and by returning the appropriate amount of reward from a particular action, the model learns to adjust its appropriately.

![](/post_images/Screen-Shot-2019-04-04-at-2.20.54-PM-1024x568.png)

The console offers a panel where we could program the logic of the reward function in Python.

\[caption id="attachment\_455" align="aligncenter" width="678"\]![](/post_images/Screen-Shot-2019-04-04-at-2.04.14-PM-1024x824.png) As a simple example, it returns a reward of 1 if DeepRacer stays at the center of the track, 0.5 if it strays off a little, 0.1 if it's close to the edge and 0.001 if it went off the track.\[/caption\]

Back in the above diagram, this reward function could be seen as a manipulatable component of the environment module. In some non-editable environments such as OpenAI Gym, Mario games, we could also further adjust the amount of reward sent to the model in additional to the default rewards given by environment, and this could be seen as a reward function as well.

### Under the hood of DeepRacer Console

As a geek, we are tempted to find out whats behind the console. Actually, the console is merely a platform that puts a series of AWS services to facilitate the training process.

![](/post_images/Screen-Shot-2019-04-04-at-2.29.14-PM-1024x630.png)

We could see that the training part(model, reward, parameters) which we saw above is powered by the Amazon SageMaker. Then it kicks off the AWS RoboMaker to create a simulation of the environment. The trained models are stored in our S3 buckets, the first person view video will be stored in the Kinesis, and the metrics(cumulative rewards) are stored in CloudWatch which provided the data behind training graph.

We could also use the Amazon Sagemaker and RoboMaker without going through the console. There are example notebooks for building models and tutorials to link these services together [here](https://docs.aws.amazon.com/deepracer/latest/developerguide/train-evaluate-models-using-sagemaker-notebook.html).

### Costs

Refer [here](https://aws.amazon.com/deepracer/pricing/) for costs.

### Hardware Details

![](/post_images/181210-under-deepracers-hood.jpg)

### DeepRacer League

Refer [here](https://aws.amazon.com/deepracer/league/) for the latest information and league standings.

### Beyond AWS DeepRacer

The realm of RL is far beyond Deep Racing. There are tons of interesting environments available at [OpenAI Gym](https://gym.openai.com), from the classic control problems like balancing a pole to Atari games.

\[video width="750" height="525" mp4="http://gator4205.temp.domains/~datageeko/wp-content/uploads/2019/04/mountaincar-dqn.mp4"\]\[/video\]

This is a result of a trained RL model (Deep Q-Network), tasked to get an under-power car to the top of the hill with flag. We could see that in the beginning, the agent(car) is exploring and actually receives a -1 reward for every step and once it has received a +10 reward for reaching the flag, it will learn how to model it's future behaviour to reach the maximal reward.

Refer [here](https://github.com/openai/gym/wiki/MountainCar-v0) to the expected rewards and states in this environment.

### Other fun examples

<iframe src="https://www.youtube.com/embed/qv6UVOQ0F44?start=125" width="560" height="315" frameborder="0" allowfullscreen="allowfullscreen"></iframe>

In the future posts, we are going to take look at the simplest RL method, Q-learning and gradually move to  Deep learning methods (Deep Q-learning Network (DQN)) and some sophisticated approaches from DeepMind papers (Actor Critic, A2C, A3C, etc)

learn to make short term decisions while optimising for long-term goal
