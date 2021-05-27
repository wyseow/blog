---
title: "Model building and performance tips for PyTorch"
date: "2019-06-03"
categories: 
  - "deep-learning"
tags: 
  - "memory"
  - "pytorch"
coverImage: "/post_images/pytorch.jpg"
---

Here are some key observations and lessons learned from building a brand new Seq-to-Seq model for sentence summarization and training it against a 1 million samples dataset.

## General

1) Always maintain codes in Git repo; it's an efficient way to ensure that different training machines have the exact same codes. Avoid the temptation to just edit(vi) the local copy in one machine and re-apply the same changes by manually typing or sending edited file(scp) to another machine. Our human brains can't never remember better than "git diff --word-diff" - comparison of differing files on a word level. Once you have edited the codes, commit and push to Git. Every time before you train the model, you just need to do a "git pull" to refresh the machine with the latest codes. Also, this is for reproducibility, which is one of the most important aspects of research.

2) Talking about reproducibility, always remember to set all the random seeds after importing all your libraries(at the top). In PyTorch this is how we could do it. Randomness comes from GPU as well, so it's important set the seed at torch.cuda interface.

\[code\] seed = 0 torch.manual\_seed(seed) torch.backends.cudnn.deterministic = True torch.backends.cudnn.benchmark = False np.random.seed(seed)

torch.cuda.manual\_seed(seed) torch.cuda.manual\_seed\_all(seed) random.seed(seed) \[/code\]

3) Have a settings file/mechanism to parameterise every hyperparameters and training parameters. Structures like this sounds like an overhead in the beginning of prototyping but we should always start small (maybe not a full-blown options class but a simple dictionary object or even organising all the parameters at the top of the file would be good enough). This also pave the way for systematic experimentation because eventually your model is going to be ready for X variants of experiments and all these efforts are going it easier to track all your attempts.

## Model building

1. Do some small test at every milestone where you have completed a key feature. A simple one would be to go through the tensors and make sure the dimensions tally when pass through the transformations. It's easy to be seem right when you are wrong.
2. Build a very very small train/valid/test dataset and use it to pull through a few epochs for testing before testing on GPU which makes debugging much more difficult.
3. When you use the small dataset for testing, pay attention to the training and validation loss when you try to run the model for a few epochs. Training loss should decrease for every single epoch and validation loss should decrease for at least the first 2 epochs before any overfitting comes into the picture. Some people in the machine learning community advocate the practice of feeding just a few samples into the model to see if it overfits, because it should. A few things could happen when you test:
    1. If training loss does not decrease, chances are the model is unable to learn anything. Check if the layers have been connected correctly. Are the gradients passing through? Is the back propagation working?
    2. If validation loss does not decrease, check whether dataset is consistent between training and validation set and make sure the loss calculation is similar. Perhaps the training procedure is different and as a result some details have been missed out?
4. Remove any bells and whistles(regularisation, etc) and start with a baseline model that couldn't go wrong and slowly build the model up.
5. Check the DL framework's documentation for default values of functions, never assume.

## Memory/Performance

1) Although it's fun and fast to prototype DL models in Jupyter notebooks, try to code it in modular manner because memory of global-scoped variables do not get release if the variable is still alive. By having training procedures in their own functions ensure that the variable are locally-scoped, and when the function has finished executing and jumps back to the calling line, the local vars of the functions are dropped and the memory allocated to them are released as well. This has allowed me to fit another 100-150 samples in a batch during training.

2) To be even more aggressive on saving memory for our pint-sized GPU, we could even delete(del) the variables by the end of the batch. When a new batch begins, CUDA will allocate new memory for the tensors of new batch and not direct overwrite the previous batch memory space. So at any one time, where batch^i > 1 there will be 2 batches worth of memory allocated. Many people have encountered similar problems and complain crashing after 2nd batch. We would usually do this in a PyTorch training procedure:

\[code\] #loop over the data loader to fetch the next batch of input #Iteration #1 for batch\_idx, batch\_inputs in enumerate(self.train\_dataloader, 0): #In first iteration, and at this point, PyTorch will allocate memory for "batch\_inputs". Let's say 1GB. #feed the batch into the model... batch\_loss = self.train\_one\_batch(batch\_inputs,'train')

#Iteration #2 for batch\_idx, batch\_inputs in enumerate(self.train\_dataloader, 0): #In second iteration, and at this point, PyTorch will allocate another memory for "batch\_inputs". Again, new 1GB space allocated. #In another words, a total of 2GB has been allocated as the previous allocation is not overwritten. #Reason: the variable is still alive at this point of time. #feed the batch into the model... batch\_loss = self.train\_one\_batch(batch\_inputs,'train')

#Iteration #N: What we should do for batch\_idx, batch\_inputs in enumerate(self.train\_dataloader, 0): batch\_loss = self.train\_one\_batch(batch\_inputs,'train')

#we should delete the variable to force the memory to be released as well #so that at any one time, there will only be 1 instance of memory allocated to "batch\_inputs" del batch\_inputs \[/code\]

3) Don't accumulate gradient history across your training loop. Tensors that have "required\_grad = True" will keep history so when you collect these tensors in a list over many training loops, they quickly add up to a sizeable memory. We should detach the variable or just retrieve the underlying data, be it numpy array or a scalar value. Many people do this to collect training loss:

\[code\] total\_loss = 0 #train model for 10000 iteration for i in range(10000): optimizer.zero\_grad() output = model(input) #calculate training loss loss = criterion(output) loss.backward() optimizer.step()

#loss is accumulated over the training loops; it will include the whole chain of gradient history, #not just the scalar value that you actually want(eg. 0.76) total\_loss += loss

#Instead, do this! .item() will return the actual value and not the whole tensor. #total\_loss += loss.item() \[/code\]

3) "nvidia-smi" doesn't necessarily show the behaviour of released memory, because Pytorch has a "caching memory allocator" to speed up memory allocations. This allows fast memory deallocation without device synchronizations. The unused memory managed by the allocator will still show as if used in nvidia-smi.

4) Don't push every tensor to GPU unless you need it for calculation in GPU. Sounds obvious but easy to miss.
