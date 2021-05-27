---
title: "Backprop, Autograd and Squeezing in larger batch using PyTorch"
date: "2019-08-15"
categories: 
  - "deep-learning"
  - "machine-learning"
tags: 
  - "gradients"
  - "pytorch"
coverImage: "/post_images/math_thumb.png"
---

Backprogation is a beautiful play of derivatives which we have taken for granted. We often do a simple one-liner:

loss.backward()

to leverage the power of automatic differentiation in many deep learning framework without much thought. Today let's look at a developed view of backpropagation as backward flow in real-valued circuits.

### Motivation

Given some function $f(x)$ ,we are primarily interested in deriving the gradients of $f$ with respect to $x$ (in other words ∇$f(x)$). Intutitivly, this means that we are trying to find out how much $f$ will change when a tiny bit of $x$ is changed and finding out this change is important because we will know how much difference to update the parameters with, when we want to steer the function($f$) into a particular direction(minimize or even maximize).

This could be expressed in:

$\\frac{df(x)}{dx} = \\lim\_{h\\ \\to 0} \\frac{f(x + h) - f(x)}{h}$

where the rate of change of a function ($f(x)$) with respect to that variable($x$) surrounding an infinitesimally small region near a particular point.

In practice, the input $x$ are fixed and we are mostly concerned with adjusting the neural network weights and bias parameters ($W, b$)

### A mini-network on paper

Let's try to build a mini neural network on paper and differentiate it by hand. This network takes the expression $f(x,y,z) = (x+y)z$ and we could break the expression down into 2 composed functions: $q=x+y$ and $f=qz$

![](/post_images/mininet1.png)

We are interested in finding out the gradient of $f$ with respect to the inputs $x,y,z$. We could use chain rule to get the derivatives of them.

$\\frac{\\partial f}{\\partial x} = \\frac{\\partial f}{\\partial q}\\frac{\\partial q}{\\partial x}$

which is actually just a mutiplication of 2 numbers that hold the gradient once we computed them.

First, we perform the **forward pass**, which is simply the computation from inputs to outputs (depicted in green). The green values at $x,y,z$ could be seen as input values into the network.

Secondly, we perform the **backward pass** that performs the backpropogation which starts at the end and recursively applies the chain rule to compute the gradients (shown in red) all the way to the inputs of the circuit. The gradients can be thought of as flowing backwards through the circuit.

Let's hand calculate them:

$\\frac{\\partial f}{\\partial f} = 1$ (derivative of itself)

$\\frac{\\partial f}{\\partial q} = \\frac{\\partial}{\\partial q}(q\*z) = z =-4$

$\\frac{\\partial f}{\\partial z} = \\frac{\\partial}{\\partial z}(q\*z) = q =3$

$\\frac{\\partial f}{\\partial x} = \\frac{\\partial f}{\\partial q} \\frac{\\partial q}{\\partial x} = -4 \* \\frac{\\partial}{\\partial x}(x+y) = -4\*(1+0) =-4 $

$\\frac{\\partial f}{\\partial y} = \\frac{\\partial f}{\\partial q} \\frac{\\partial q}{\\partial y} = -4 \* \\frac{\\partial}{\\partial y}(x+y) = -4\*(1+0) =-4 $

All the above values are in red.

### Effects of interactions between gates

We could see that the **addition gates** always takes the gradient and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged).

On the other hand, the **multiply gate** takes the input values, swaps them and multiplies by its local gradient.

Imagine that one of the inputs to the multiply gate is very small(0.1-1) and the other is very big (100-512) then it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. In other words, the scale of the data has an effect on the magnitude of the gradient for the weights. For example, if you multiplied all input data examples $x\_{i}$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you’d have to lower the learning rate by that factor to compensate. This is why **preprocessing matters a lot**, sometimes in subtle ways. And having a good understanding for how the gradients flow through the network can help us debug some of these cases.

### Local gradients

The beauty of back-propgation is not going through the mechanical process of retriving the derivatives of $x,y,z$. Instead, we could look at it as series of gates where you could compute the local gradient of it's inputs with respect of output value at the gate completely indepdently. During the backpropogation, we could just multiply the gradient from upstream into the local gradient.

![](/post_images/mininet2.png)

In this example, the local gradients are in blue. From the addition gate($q$), we could already know the local gradient(+1) without knowing all other inputs and values upstream. Therefore, during the backward pass, we simply have to multiply it the upstream gradient (-4 \* 1) =-4. Hence the whole backprogation could be simplify as a process of multiplying the upstream gradient with the local gradient calculation of each gate. A nice way to think about it is: Force X Local Gradient

### Autograd in PyTorch

Let's explore the concept in the PyTorch framework where it uses the same mechanism for back-prop. These Pytorch tensors(x,y) could be seen as the (x,y) in the previous example except that we do a multiplication instead of addition. We create 2 tensors with the following attributes and put them through a multiplication gate(operation in Pytorch terms) to produce another tensor. By the way, PyTorch builds the computation graph as you define the interaction between the tensors and in the forward pass.

![](/post_images/autograd1.png)

Most of the attributes in the tensor are self explantory and we want to focus on the "**grad\_fn**" attribute, which points to the backward function, which is the calculation needed to compute the **local gradients**. Each operation in Pytorch has a "backward version". In this case, it's "MulBackward". When we do a backward pass(tensor.backward()), the upstream gradient is pass from the end of the graph and follow the path to invoke the associated "grad\_fn" to compute the local gradient. This local gradient is multiplied with the upstream gradient and in turn stored in the tensor's "**.grad**" attribute, and the cycle continues until the start of the graph. At the end, Pytorch clears the computation graph.

![](/post_images/autograd2.png)

Note that not every tensor is "eligible" for the gradient because only tensors which are leaves and explicitly initialized will receive the gradients.

### Gradient accumulation

PyTorch allows us to take finer control of the whole backpropagation process and we do fancy things with it. Since each "tensor.backward()" calculates the gradients for every parameters and store(add) in each of their ".grad" attribute, we could make several backward pass and let the gradients in ".grad" accumulate before calling optimizer.step() to perform a step of gradient descent. This is the usual practice when training a neural network in PyTorch:

{% highlight python %}
#We feed in 1 batch of 100 samples, here we compute gradients for 100 samples and update the parameters with them
for i, (inputs, labels) in enumerate(batches):
    #inputs: 100x512x512
    predictions = model(inputs)                     # Forward pass (builds a graph)
    loss = loss\_function(predictions, labels)       # Compute loss function
    loss.backward()                                 # Backward pass(compute grad and parameter's .grad updated)
    optimizer.step()                                # Now we can do an optimizer step (update the parameters with gradients)
    model.zero\_grad()                               # Reset gradients tensors (all .grad becomes 0, doesn't reset unless we call it)
{% endhighlight %}

And you might ask what's the use of accumulating gradients?

Consider a case where a batch of samples(batch size of 100) are simply too big to fit into the memory and after some trial-and-error, the most that you could fit in your GPU is probably batch size of 20. This is how we could train with the same 100 samples by accumulating gradients for 5 steps of 20 samples.

{% highlight python %}
#5 batches of 20 samples, here we compute gradients for 20 samples but don't update the parameters with them
#we accumulate the gradients in .grad until we have accumlate 5 
accumulation\_steps = 5                              #we accumlate for 5 steps(20x5=100)
for i, (inputs, labels) in enumerate(batches):
    #inputs: 20x512x512
    predictions = model(inputs)                     # Forward pass (builds a graph)
    loss = loss\_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation\_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass(compute grad and parameter's .grad updated)
    if (i+1) % accumulation\_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step (update the parameters with gradients)
        model.zero\_grad()                           # Reset gradients tensors
{% endhighlight %}

Note that this is different from just training a smaller batch size because this would be computing loss and gradients based on 20 samples, not 100 samples.

{% highlight python %}
#We feed in 5 batch of 20 samples, here we compute gradients for 20 samples and update the parameters with them
for i, (inputs, labels) in enumerate(batches):
    #inputs: 20x512x512
    predictions = model(inputs)                     # Forward pass (builds a graph)
    loss = loss\_function(predictions, labels)       # Compute loss function
    loss.backward()                                 # Backward pass(parameter's .grad updated)
    optimizer.step()                                # Now we can do an optimizer step (update the parameters with gradients)
    model.zero\_grad()                               # Reset gradients tensors (all .grad becomes 0, doesn't reset unless we call it)
{% endhighlight %}

### Gradient checkpointing

What happens if you can't even pass 1 sample through the network?

Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass. It can be applied on any part of a model.

Specifically, in the forward pass, function will run in torch.no\_grad() manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the function parameter. In the backwards pass, the saved inputs and function is retreived, and the forward pass is computed on function again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

![](/post_images/grad_ckpt1.gif)

As the forward pass progresses, the nodes in the computational graph store the intermediary values required for backpropagation. You could imagine that the more layers the network has, the higher amount of memory it will use.

![](/post_images/grad_ckpt2.gif)

Instead of saving all of them, we could save memory by forgetting nodes as they are consumed and recomputing them later. But this could leads to high number of computational steps.

![](/post_images/grad_ckpt3-300x87.png)

A middle ground give rise to **gradient checkpointing**, which is to select some key nodes as "checkpoints" to save the intermediate results so nodes don't need to recompute all the way back for the values.

![](/post_images/grad_ckpt4.gif)

For a chain of length n, generalization of this strategy is to place checkpoints every sqrt(n) steps.

I have not personally experimented with gradient checkpointing but it looks promising and pluasible because Pytorch's native library support this function directly.

Some sample implementation codes can be seen here:  
[https://github.com/prigoyal/pytorch\_memonger/blob/master/tutorial/Checkpointing\_for\_PyTorch\_models.ipynb](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)

In conclusion, I have roughly went through backpropogation in neural networks and relating the mechanisms of the process to the corresponding functions in PyTorch. I hope you have learn something and thanks for reading.

References:  
1) https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255  
2) https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9  
3) https://pytorch.org/docs/stable/checkpoint.html  
4)https://github.com/prigoyal/pytorch\_memonger/blob/master/tutorial/Checkpointing\_for\_PyTorch\_models.ipynb  
5) https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c  
6) http://karpathy.github.io/neuralnets/
