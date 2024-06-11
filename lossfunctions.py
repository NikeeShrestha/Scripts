import torch

##Loss to make uncorrelated latent variables
class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambd=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        N, D = z1.size()

        # Normalize the representations along the batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / N

        # Loss function
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

##MSE loss for decoder
class MSElossFunc(torch.nn.Module):
    def __init__(self):
        super(MSElossFunc, self).__init__()

    def forward(self, predicted, target):
        return torch.mean((predicted-target) ** 2)

##data augment functions
class dataaugment:
    def __init__(self, data):
        self.data=data
        
    def add_noise(self, noise_level=0.01):
        noise = torch.rand_like(self.data) * noise_level
        return self.data+noise
    
    def random_feature_dropping(self, drop_prob=0.1):
        mask = torch.rand_like(self.data) > drop_prob
        return self.data * mask
    
    def shuffle_data(self):
        indices = torch.randperm(self.data.size(0))
        return self.data[indices]