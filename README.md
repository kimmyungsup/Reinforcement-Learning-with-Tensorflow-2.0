# RL-with-TF-2.0
Tensorflow 2.0 Reinforcement Learning

## Reinforcement Learning with Tensorflow 2.x

This repository is for guide code about RL with **tensorflow 2.0**(tf 2.0).
There are some changes with tf 2.0, but those are not that difficult and makes your code more intuitive.
In this repository, I will show you very basic codes of RL algorithms so that you can easily understand.
<br>
<br>


## Difference with old TF version

There are remarkable changes with tf 2.0 which are still applied in old version(tf 1.14 or keras), but those are may not familiar with you. Here are some important changes that I think.
<br>
1. Eager execution<br>
After **Eager execution** is applied in tensorflow, you don't have to define static graph for calculation. Instead, you can get tensor return instantly when you execute tensor operation. And also, you can convert tensor to numpy anytime(That can be output of your model).

2. No session, Auto graph<br>
In Tf 2.0, you have to write no session code, beacuse eager execution is default. But it is much easier than before. **All of operation can interact with python codes.** And you can also write your model with python's contorl op(for, while, if...) 

3. with Keras<br>
It is recomended that write your model with keras. It will be much easier, but powerful. We can write model as **Sub-classing API**(like pytorch). If you have experience with pytorch, you will know how nice it is. And you can easily compute and manipulate gradients by 'gradient tape', and 'trainable_variablse'. This will be more powerful than before, and also it is applied in this repository's code.
<br>

## Features
Here is some notes for this code 
1. Sub-classing API model
2. Gradient tape for compute gradient(not using model.compile)
3. Models are seperated by classes
<br>

### used version
tensorflow-gpu == 2.0.0rc1<br>
gym == 0.14.0<br>
numpy == 1.17.2<br>
(python = 3.6.9)<br>

