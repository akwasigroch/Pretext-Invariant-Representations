# Self-Supervised Learning of Pretext-Invariant Representations - implementation

My implementation of the paper "Self-Supervised Learning of Pretext-Invariant Representations" by Ishan Misra and Laurens van der Maaten (https://arxiv.org/abs/1912.01991).

The main objective of the algorithm is to learn representations that are invariant to the transformations. This is obtained by the loss function that minimizes the distance between representations of the original image and modified image. At the same time, the distance between the original image and other images from the dataset is maximized.

The implementation contains two types of pretext task:
 - Jigsaw transformation pretext task (jigsaw.py)
 - Rotation pretext task (rotation.py)

I decided to test the code on the small dataset of skin lesions (2000 images). My initial experiments have shown promising results of using self-supervised pretraining on a small dataset (both self-supervised pretraining and downstream task training were performed the small dataset):
- training of ResNet50 from scratch (AUC around 0.55)
- training of ResNet50 initialized by weights obtained by self-supervised pretraining (AUC around 0.7)
- training of ResNet50 initialized by weights of network pre-trained on Imagenet (AUC around 0.8)


## Dependencies
- Pytorch 1.3.1
- PIL
- Numpy 


