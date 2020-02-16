# Self-Supervised Learning of Pretext-Invariant Representations - implementation

My implementation of the paper "Self-Supervised Learning of Pretext-Invariant Representations" by Ishan Misra and Laurens van der Maaten (https://arxiv.org/abs/1912.01991).

I decided to test the code on the small dataset of skin lesions (2000 images). My initial experiments show promising results of using self-supervised pretraining:
- training of ResNet50 from scratch (AUC around 0.55)
- training of ResNet50 initialized by weights obtained on self-supervised pretraining (AUC around 0.7)
- training of ResNet50 initialized by weights of network pretrained on Imagenet (AUC around 0.8)


## Dependencies
- Pytorch 1.3.1
- PIL
- Numpy 


