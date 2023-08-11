This repository contains all resources of my talk in SWB's Data Science Bootcamp for University of Lagos. 

Topics covered:
1. Basic image processing, understanding 3d scenes
2. Introduction to supervised learning in the CV domain
3. How to interpret vision models
4. Advances in supervised learning in the CV domain

You will find the main presentation in [cv_bootcamp.ipynb](./cv_bootcamp.ipynb). To run this presentation on colab, please use [this link](https://colab.research.google.com/github/cybershiptrooper/CV_bootcamp_SWB/blob/main/cv_bootcamp.ipynb). If you are visiting after the talk, you can just read through the notebook, and you won't miss a thing!

To run this on your local system, use the following steps:

```
git clone https://github.com/cybershiptrooper/CV_bootcamp_SWB.git
cd CV_bootcamp_SWB/resources 
source get_cifar10.sh
```

You will need pytorch, numpy and matplotlib to run the notebook.

Misc: 

Bonus exercises are listed under the [bonus_exercises folder](./bonus_exercises/). 
It consists of two exercises:
1. Implementing Convolutions for multiple RGB images
2. Implementing the attention mechanism you learnt in the last session

Both using numpy

Solutions and other resources can be found under [resources](./resources/). Apart from the datasets used, it has all the [auxiliary notes](./resources/notes/), [solutions](./resources/solns/) to exercises in the notebook and some [useful utilities](./resources/utils/) to train on cifar 10.

