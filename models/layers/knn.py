import torch
import torch.nn as nn
import warnings
# from knn_cuda import KNN as KNNCUDA


@torch.no_grad()
def knn_point(k, query, support=None):
    """Get the distances and indices to a fixed number of neighbors
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
    """
    if support is None:
        support = query
    
    # Add shape validation
    if len(query.shape) != 3:
        raise ValueError(f"Expected query to be 3D tensor [B,N,C], got shape {query.shape}")
    if len(support.shape) != 3:
        raise ValueError(f"Expected support to be 3D tensor [B,N,C], got shape {support.shape}")
        
    dist = torch.cdist(query, support)
    k_dist = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return k_dist.values, k_dist.indices
    

class KNN(nn.Module):
    """Get the distances and indices to a fixed number of neighbors

    Reference: https://gist.github.com/ModarTensai/60fe0d0e3536adc28778448419908f47

    Args:
        neighbors: number of neighbors to consider
        p_norm: distances are computed based on L_p norm
        farthest: whether to get the farthest or the nearest neighbors
        ordered: distance sorted (descending if `farthest`)

    Returns:
        (distances, indices) both of shape [B, N, `num_neighbors`]
    """
    
    def __init__(self, neighbors, 
                 farthest=False, 
                 sorted=True, 
                 **kwargs):
        super(KNN, self).__init__()
        self.neighbors = neighbors
        self.farthest = farthest
        self.sorted = sorted
        
        if kwargs:
            warnings.warn(f"Unused parameters in KNN initialization: {kwargs}")

    @torch.no_grad()
    def forward(self, query, support=None):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        if support is None:
            support = query
            
        # Add validation
        if not torch.is_tensor(query):
            raise TypeError(f"Expected query to be torch.Tensor, got {type(query)}")
            
        if len(query.shape) != 3:
            raise ValueError(f"Expected query shape [B,N,C], got {query.shape}")
            
        if self.neighbors > query.shape[1]:
            warnings.warn(f"Number of neighbors ({self.neighbors}) is larger than number of points ({query.shape[1]})")
            
        dist = torch.cdist(query, support)
        k_dist = dist.topk(k=self.neighbors, dim=-1, largest=self.farthest, sorted=self.sorted)
        return k_dist.values, k_dist.indices.int()


# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        # self.knn = KNNCUDA(k * self.dilation, transpose_mode=True)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


