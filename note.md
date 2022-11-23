核心代码 graph.py, tgcn.py, st-gcn.py
work_dir: model.pt -> result
config.yaml -> args
图卷积: 图像特征->邻接矩阵
邻接矩阵*权重矩阵

python main.py recognition -c config/st_gcn/ntu-xsub/train.yaml

model->net.st_gcn.model
- Model Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

- Model Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence, 帧数
            :math:`V_{in}` is the number of graph nodes, 关节数
            :math:`M_{in}` is the number of instance in a frame. 人数

- graph.py

  - 构建graph的参数：araph args -> ntu-rgb+b & spatial (数据集和策略)
  - uniform: A=normalized adjacency matrix
    the st_gcn model -> self.graph gets D^(-1)A for 'uniform' (normalized adjacency matrix)
  - 构建的self.A为AD(归一化的邻接矩阵)
  - self.A的形状为[\[\]\[\]]，因此A[0]访问到矩阵

- st_gcn.py

  - A=self.graph.A, 但不是参数、不需要梯度

  - spatial_kernel_size=A.size(0) # 矩阵

    temporal_kernel_size=9

    kernel_size = (temporal_kernel_size, spatial_kernel_size)

  - class st-gcn


