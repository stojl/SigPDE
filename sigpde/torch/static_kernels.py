import torch

class LinearKernel():
    def __init__(self, scale=1.0):
        self.scale = scale
        
    def pairwise(self, X, Y):
        return self.scale * torch.bmm(X, Y.permute(0,2,1))

    def gram(self, X, Y):
        return self.scale * torch.einsum('ipk,jqk->ijpq', X, Y)

class RBFKernel():
    def __init__(self, sigma):
        self.sigma = sigma

    def pairwise(self, X, Y):
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        return torch.exp(-dist/self.sigma)

    def gram(self, X, Y):
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        return torch.exp(-dist/self.sigma)