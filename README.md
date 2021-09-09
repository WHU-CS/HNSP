# HNSP

This repository provides a reference implementation of HNSP proposed in "High Order Node Similarity Preserving Method for Robust Network Representation",
Zhenyu Qiu, Jia Wu, Wenbin Hu, Bo Du, Zhongzheng Tang, Xiaohua Jia , and Erik Combria

The noise in the network will affect the performance of network embedding. 
The NSP aims to exploit node similarity to address the problem of social network
embedding with noised and learn the a representations for nodes in a social network with noise.

## Basic Usage

### Noise network dataset generation

First of all, to simulate a network with noise, you need to generate the 
noised network dataset by the following command. 

`python add_noise.py --dataset dataset/xx-network.txt  --ratio x`

#### --dataset:*input_filename*

The input file should be an edge list and the nodes are numbered starting 
from 0, e.g:

```
0,1
0,2
1,3
1,5
```

#### --ratio:*noise_ration*

The ratio of noised added, and its value range if [0,1].

### Optimal parameters of the comprehensive similarity index

After obtain a network with noise, you need to search the optimal parameters of the comprehensive similarity index by the following command.
`python hsc.py --dataset dataset/xx-network.txt  --node_size n`

### Network embedding

After the above two steps, you can learn the representations of nodes by the following command.

`python HNSP.py --dataset dataset/xx.txt --node_size n --parameters n1 n2 n3 n4 --noise_network dataset/xx.txt  --lables dataset/xx-labels.txt`

#### --dataset:*input_filename*

The input file should be an edge list and the nodes are numbered starting 
from 0.

#### --node_size:

The node size of the input network 

#### --parameters

The optimal parameters of the comprehensive similarity index 

#### --noise_network

The network that needs to be embedded  

#### --labels

The node label file used in node classification task

