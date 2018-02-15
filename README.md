## Siamese Neural Network

This is a work-in-progress implementation of Siamese Neural Network based on Inception-Resnet-V2 architecture
as described ![here](https://arxiv.org/abs/1602.07261) with contrastive loss as described
![here](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf).
There are couple changes in terms of the filter size as the numbers shown in the article seems to be a bit off - you
can't do resnet shortcut summation without these changes.

Most of the implementation uses Keras, but there some places, that rely on Tensorflow (and Tensorboard for logging).
Full list of requirements can be found in requirements.txt. OpenCV bindings to Python are also required.