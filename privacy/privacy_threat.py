#%%
import torch
from rich.progress import track

#TODO :
# - Check to(device)
# - implement GAE
# - dataloader


#%%
def loss_GNN(adj:  torch.Tensor, victim_model, X :  torch.Tensor, y:  torch.Tensor):
    """
    
    
    :param adj: the adjacency matrix of the graph
    :type adj: torch.Tensor
    :param victim_model: the model that we want to attack
    :param X: The input data
    :type X: torch.Tensor
    :param y: the true labels
    :type y: torch.Tensor
    """
    return torch.norm(victim_model(adj,X)-y)

def loss_smooth(adj: torch.Tensor,X : torch.Tensor) -> torch.Tensor:
    """
    
    :param adj: The adjacency matrix of the graph
    :type adj: torch.Tensor
    :param X: the input data, a tensor of shape (n,d) where n is the number of nodes and d is the
    dimension of the node features
    :type X: torch.Tensor
    """
    degree_vector = adj.sum(1).flatten()
    degree_vector[degree_vector==0]=1e-3
    D = torch.diag(degree_vector)
    L = D - adj
    D_sqrt_inv = D.pow(-1/2)
    L_norm = torch.chain_matmul(D_sqrt_inv, L, D_sqrt_inv)
    return torch.trace(torch.chain_matmul(X.t(), L_norm, X))

def loss_penalization(adj : torch.Tensor) ->  torch.Tensor :
    """
    > This function takes in a tensor of shape (n,n) and its norm
    
    :param adj: the adjacency matrix of the graph
    :type adj: torch.Tensor
    """
    return torch.norm(adj, p='fro')

def adj_mat2vec(adj: torch.Tensor, device) -> torch.Tensor:
    """
    > We take the lower triangular part of the adjacency matrix, flatten it, and return it
    
    :param adj: torch.Tensor
    :type adj: torch.Tensor
    :return: The flattened adjacency matrix
    """


    assert torch.all(adj == adj.t()), 'Adjency Matrix not symmetric'

    n_dim = adj.shape[0]
    flattened = torch.tril(adj, diagonal=-1).t().flatten().to(device)
    return flattened[:n_dim*(n_dim-1)/2]

def adj_vec2mat(adj_vec: torch.Tensor, n_dim: int, device)-> torch.Tensor:
    """
    > It takes a vector of length $(n-1)/2$ and returns a symmetric matrix of size $n \times n$
    
    :param adj_vec: the vector of adjacency values
    :type adj_vec: torch.Tensor
    :param n_dim: the number of nodes in the graph
    :type n_dim: int
    :param device: the device to run the model on
    :return: The adjacency matrix
    """
    
    adj_mat = torch.zeros((n_dim, n_dim)).to(device)
    triang_indices = torch.tril_indices(row=n_dim, col=n_dim, offset=-1)
    adj_mat[triang_indices[0], triang_indices[1]] = adj_vec
    return adj_mat + adj_mat.t()

def projection(a : torch.Tensor):
    """
    It takes a tensor and returns a tensor with all values between 0 and 1
    
    :param a: the tensor to be projected
    :type a: torch.Tensor
    :return: The projection of a onto the simplex
    """
    return torch.clamp(a, min=0, max=1)



def learning_rate(t: int):
    return 1e-5

class Adjency_optimize() :

    def __init__(self, alpha : float, beta : float, device, n_epochs : int = 100, ) -> None:
        self.n_epochs = n_epochs  
        self.loss = lambda adj, victim_model, X, y : loss_GNN(adj, victim_model, X, y)+ alpha * loss_smooth(adj,X) + beta * loss_penalization(adj) 
        self.device = device
        
        

    def retrieve(self, adj: torch.Tensor, victim_model,X: torch.Tensor,y : torch.Tensor):
        """
        > We take the gradient of the loss function with respect to the adjacency matrix, and then we update
        the adjacency matrix by subtracting the gradient from it
        
        :param adj: the adjacency matrix of the graph
        :type adj: torch.Tensor
        :param victim_model: The model that we want to attack
        :param X: the data
        :type X: torch.Tensor
        :param y: the true labels of the data
        :type y: torch.Tensor
        :return: The final adjacency matrix and the loss values at each epoch.
        """

        n_dim = adj.shape[0]

        adj_mat = adj

        losses = []

        for t in track(range(self.n_epochs)):

            loss = self.loss(adj_mat, victim_model, X, y).to(self.device)

            losses.append(loss)

            grad = torch.autograd.grad(loss,adj_vec := adj_mat2vec(adj_mat, self.device),)[0]

            adj_vec = projection(adj_vec-torch.mul(grad, learning_rate(t)))

            adj_mat = adj_vec2mat(adj_vec, n_dim=n_dim, device = self.device)

            assert torch.all(adj_mat == adj_mat.t()), 'Adjency Matrix not symmetric'


        return adj_mat, losses


class GAE:
    pass
            



          




