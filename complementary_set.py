import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torchvision.datasets as dsets
from torchvision import transforms
import numpy as np
from utils import View
from sklearn.metrics import roc_auc_score

class VAENet(nn.Module):
    def __init__(self, hidden_size, device='cuda'):
        super(VAENet, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),    #  in_channels, out_channels, kernel_size, stride, padding, diction , groups, bias
                                                        # B,  1, 28, 28
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # B,  32, 14, 14
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # B, 64,  7,  7
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            View((-1, 128 * 3 * 3)),                # B, 128, 3, 3
        )
        self.linear1 = nn.Linear(128 * 3 * 3, hidden_size)
        self.linear2 = nn.Linear(128 * 3 * 3, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128 * 7 * 7),              # B, hidden_size
            View((-1, 128, 7, 7)),                            # B, 128,  7,  7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # B, 128, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # B, 64, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 1),                     # B, 32, 28, 28
        )

    def forward(self, x):
        z, z_mu, z_logvar = self.encoding(x)
        recon_x = self.decoding(z)

        return recon_x, z_mu, z_logvar

    def reparametrizing(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
        z = eps.mul(std).add_(mu)
        return z

    def encoding(self, x):
        z = self.encoder(x)
        z_mu  = self.linear1(z)
        z_logvar = self.linear2(z)
        z = self.reparametrizing(z_mu, z_logvar)

        return z, z_mu, z_logvar

    def decoding(self, z):
        recon_x = self.decoder(z)
        recon_x = torch.sigmoid(recon_x)
        return recon_x

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('fc') != -1:
        torch.nn.init.kaiming_normal(m.weight.data)
        torch.nn.init.kaiming_normal(m.bias.data)

def sampling(mu, sigma, device='cuda'):
    # size = (L,) + mu.size()
    # l_epsilon = torch.randn(size)
    z = mu + sigma * torch.randn_like(sigma).to(device)
    z = Variable(z)

    return z


def get_kl(mu, sigma, train=True, y=None, s=20):

    if train:
        assert y is not None
        idx_n, idx_a = (y == 0), (y != 0)  # normal index, anomaly index
        n_mu, n_sig, a_mu, a_sig = mu[idx_n], sigma[idx_n], mu[idx_a], sigma[idx_a]
        total_len = len(mu)
        # KL for normal
        KL_n = (-0.5 - (n_sig.log()) + 0.5 * (n_sig ** 2) + 0.5 * (n_mu ** 2)).sum()
        var_a = a_sig ** 2
        s_pow = s ** 2
        repeat_n = a_mu.size(0)*a_mu.size(1)    # batch * hidden_size
        # KL for anomaly
        KL_a = (torch.sqrt(2 * np.pi / (var_a + 1)) * (-(a_mu ** 2) / (2 * (var_a + 1))).exp() + \
                (a_mu ** 2 + var_a) / (2 * s_pow) - a_sig.log()).sum() + \
               repeat_n * (np.log(s * (np.sqrt(s_pow + 1) - 1)) -0.5 *np.log(s_pow + 1) + 0.5*(np.log(2 * np.pi) - 1))
        return (KL_n + KL_a).div(total_len)
    else:
        # -(1/2) -log(sigma) + (1/2)sigma^2 + (1/2)mu^2
        return (-0.5 - (sigma.log()) + 0.5 * (sigma ** 2) + 0.5 * (mu ** 2)).sum(-1)


hidden_size = 500

lr = 0.0001
epoch = 200
batch_size = 100
L = 1
C = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = VAENet(hidden_size=hidden_size, device=device).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

for e in range(epoch):
    valid = 0
    for b, data in enumerate(trainloader):
        x, y = data
        x, y = Variable(x).to(device), Variable(y).to(device)
        p_x_z, z_mu, z_logvar = net(x)
        z_sigma = z_logvar.mul(0.5).exp()
        # assertion
        p_x_z = torch.clamp(p_x_z, min=1e-8, max=1-(1e-8))
        recon_loss = (x*torch.log(p_x_z) + (1-x)*torch.log(1-p_x_z)).sum((-1,-2)).squeeze().mean()

        # y == 1,2,3 -> normal(0), else -> anomaly(1)
        anomaly_label = torch.where(((y == 1)| (y == 2) | (y == 3)), torch.zeros_like(y), torch.ones_like(y))
        # s^2 = 400
        KL = get_kl(z_mu, z_sigma, train=True, y=anomaly_label, s=20)

        # ELBO
        loss = -(recon_loss - C*KL)
        valid += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('loss ',valid/len(trainloader))
    print('recon loss {0:.4f} KL {1:.4f}'.format(-recon_loss, KL))

results = None
y_true = None
for data in testloader:
    x, y = data
    x = Variable(x).to(device)

    p_x_z, z_mu, z_logvar = net(x)
    z_sigma = z_logvar.mul(0.5).exp()

    # assertion
    p_x_z = torch.clamp(p_x_z, min=1e-8, max=1 - (1e-8))
    recon_loss = (x * torch.log(p_x_z) + (1 - x) * torch.log(1 - p_x_z)).sum((1, -2, -1))
    # s^2 = 400
    KL = get_kl(z_mu, z_sigma, train=False)

    # ELBO
    loss = -(recon_loss - C * KL)

    if results is None:
        results = torch.cat((-recon_loss, loss, KL)).view(-1,3)
        y_true = y
    else:
        results = torch.cat((results, torch.cat((-recon_loss, loss, KL)).view(-1,3)))
        y_true = torch.cat((y_true, y))

# y == 1,2,3 -> normal(0), else -> anomaly(1)
y_true = torch.where((y_true==1)|(y_true == 2)|(y_true == 3), torch.zeros_like(y_true), torch.ones_like(y_true))
if torch.cuda.is_available():
    results = results.detach().cpu().numpy()
else:
    results = results.detach().numpy()

print('test results using AUROCs')
score_RL = roc_auc_score(y_true, results[:,0])
score_ELBO = roc_auc_score(y_true, results[:,1])
score_KL = roc_auc_score(y_true, results[:,2])

print('RL : {0:.4f} / ELBO : {1:.4f} / KL : {2:.4f} '.format(score_RL, score_ELBO, score_KL))
