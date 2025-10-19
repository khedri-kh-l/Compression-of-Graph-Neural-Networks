\title{Pruning and Quantization Impact on Graph Neural Networks}

## Overview

Graph neural networks (GNNs) are known to operate with high accuracy on learning from graph-structured data, but they suffer from high computational and resource costs. Neural network compression methods are used to reduce the model size while maintaining reasonable accuracy. Two of the common neural network compression techniques include pruning and quantization. In this research, we empirically examine the effects of three pruning methods and three quantization methods on different GNN models, including graph classification tasks, node classification tasks, and link prediction. We conducted all experiments on three graph datasets, including Cora, Proteins, and BBBP. Our findings demonstrate that unstructured fine-grained and global pruning can significantly reduce the model's size(50\%) while maintaining or even improving precision after fine-tuning the pruned model. The evaluation of different quantization methods on GNN shows diverse impacts on accuracy, inference time, and model size across different datasets. 


## Contact

If you have any technical questions, please submit a new issue.

If you have other questions, please contact us:
Khatoon.l Khedri [khedri.kh.l@gmail.com] \\
 Qifu Wen [qfwen@bu.edu] \\
Reza Rawassizadeh [Rezar@bu.edu].
