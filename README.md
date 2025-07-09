# Graph-based-Consistency-Learning-for-Contrastive-Multi-View-Clustering
Code for "Graph based Consistency Learning for Contrastive Multi-View Clustering" 


We pre-train our model with 100 epochs on Caltech dataset in a single-gpu, run: 
python train.py --data_root [Root directory of dataset] --learning_rate 0.0003 --epochs 100  --batch_size 256 --gpu 0

train.py: The purpose of this file is to train model.
dataloader.py: The purpose of this file is to load various datasets.
loss.py: The purpose of this file is to calculate the loss.
metric.py: The purpose of this file is to calculate three kinds of clustering metric.
network.py: The purpose of this file is to obtain the network of our entire model.
utils.py: The purpose of this file is to build common tool approaches.
