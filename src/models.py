import numpy as np
import torch
import torch.nn as nn
###########CHANGED#############
import torch.nn.functional as F 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Logistic:
    def __init__(self, param):
        self.w = param
    
    def forward(self, x):
        return sigmoid(x @ self.w / np.sqrt(len(x)))

class MLP:
    def __init__(self, param):
        self.w1, self.b1, self.w2, self.b2 = param
    
    def __call__(self, x):
        x = sigmoid(x @ self.w1 + self.b1)
        return sigmoid(x @ self.w2 + self.b2)
    
class Bilinear:
    def __init__(self, param):
        self.M = param
    
    def __call__(self, context, features):
        return sigmoid(features @ self.M.T @ context / np.sqrt(len(context)*features.shape[1])).reshape(-1)

class Winrate: # dependent on context and bid
    def __init__(self, mode, context_dim, param=None, agents=None):
        self.mode = mode
        self.context_dim = context_dim
        if mode=='simulation':
            self.agents = agents
        elif mode=='logistic':
            self.model = Logistic(param)
        elif mode=='MLP':
            self.model = MLP(param)
    
    def __call__(self, context, bid):
        if self.mode=='simulation':
            pass
        else:
            if len(bid)==1:
                return self.model(np.concatenate([context, bid])) 
            else:
                x =np.concatenate([
                    ###############CHANGED################
                    np.tile(context.reshape(1,-1), (len(bid),1)),
                    bid.reshape(-1,1)
                ], axis=1)
                return self.model(x)
        
class CTR:
    def __init__(self, mode, context_dim, item_features, param): # item_features = ad
        self.mode = mode
        self.d = context_dim
        self.item_features = item_features
        self.K = item_features.shape[0] # number of items
        self.h = item_features.shape[1] # item feature dimension
        if mode=='bilinear':
            self.model = Bilinear(param)
        elif mode=='MLP':
            self.model = MLP(param)
    
    def __call__(self, context):
        if self.mode=='bilinear':
            return self.model(context, self.item_features)
        elif self.mode=='MLP':
            ##############CHANGED##################
            return self.model(np.concatenate([
                np.tile(context.reshape(1,-1), (len(self.item_features),1)),
                self.item_features.reshape(self.K, self.h)
            ], axis=1))

                              
class LogisticRegression(nn.Module):
    def __init__(self, context_dim, items, mode, rng, lr, c=1.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.items_np = items
        self.items = torch.Tensor(items).to(self.device)
        self.K = items.shape[0] # number of items
        self.d = context_dim
        self.h = items.shape[1] # item feature dimension
        self.c = c
        self.nu = nu

        self.M = nn.Parameter(torch.Tensor(self.d, self.h)) # CTR = sigmoid(context @ M @ item_feature)
        nn.init.kaiming_uniform_(self.M)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.uncertainty = []
        self.S0_inv = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.S_inv = np.eye(self.h*self.d)
        self.S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.sqrt_S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)

    def forward(self, X, A):
        return torch.sigmoid(torch.sum(F.linear(X, self.M.T)*self.items[A], dim=1))
    
    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.loss(X, A, y)
            loss.backward()
            optimizer.step()
    ##################CHANGED###############
        # y = self(X,A).numpy(force = True)
        y = self(X, A).detach().cpu().resolve_conj().resolve_neg().numpy()
        y = y * (1 - y)
        contexts = contexts.reshape(-1,self.d)
###############CHANGED###################33
        #self.S_inv = self.S0_inv.numpy(force=True)
        self.S_inv = self.S0_inv.detach().cpu().resolve_conj().resolve_neg().numpy()
        for i in range(contexts.shape[0]):
            context = contexts[i]
            item_feature = self.items_np[A[i]]
            phi = np.outer(context, item_feature).reshape(-1)
            self.S_inv += y[i] * np.outer(phi, phi)
        self.S = torch.Tensor(np.diag(np.diag(self.S_inv)**(-1))).to(self.device)
        self.sqrt_S = torch.Tensor(np.diag(np.sqrt(np.diag(self.S_inv)+1e-6)**(-1))).to(self.device)

    def loss(self, X, A, y):
        y_pred = self(X, A)
        m = self.flatten(self.M)
        return self.BCE(y_pred, y.reshape(y_pred.shape)) + torch.sum(m.T @ self.S0_inv @ m / 2)
    
    def estimate_CTR(self, context, UCB=False, TS=False):
        # context @ M @ item_feature = M * outer(context, item_feature)
        X = []
        context = context.reshape(-1)
        for i in range(self.K):
            X.append(np.outer(context, self.items_np[i]).reshape(-1))
        X = torch.Tensor(np.stack(X)).to(self.device)
        with torch.no_grad():
            if UCB:
                m = self.flatten(self.M)
                bound = self.c * torch.sum((X @ self.S) * X, dim=1, keepdim=True)
                U = torch.sigmoid(X @ m + bound)
                ###################CHANGED#######################
                # return U.numpy(force=True).reshape(-1) 
                return U.detach().cpu().resolve_conj().resolve_neg().numpy().reshape(-1)

            elif TS:
                m = self.flatten(self.M)
                m = m + self.nu * self.sqrt_S @ torch.Tensor(self.rng.normal(0,1,self.d*self.h).reshape(-1,1)).to(self.device)
                out = torch.sigmoid(X @ m)
                return out.numpy(force=True).reshape(-1)
            else:
                m = self.flatten(self.M)
                return torch.sigmoid(X @ m).numpy(force=True).reshape(-1)

    def get_uncertainty(self):
        ###########numpy(force = True)
        S_ = self.S.detach().cpu().resolve_conj().resolve_neg().numpy()
        eigvals = np.linalg.eigvals(S_).reshape(-1)
        return eigvals.real

    def flatten(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0]*tensor.shape[1], -1))
    
    def unflatten(self, tensor, x, y):
        return torch.reshape(tensor, (x, y))

# ==========winrate estimators==========

class NeuralWinRateEstimator(nn.Module):
    def __init__(self, context_dim, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.H = 16
        if self.skip_connection:
            self.linear1 = nn.Linear(context_dim, self.H-1)
        else:
            self.linear1 = nn.Linear(context_dim+1, self.H)
        self.linear2 = nn.Linear(self.H, 1)
        self.BCE = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.relu(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            return torch.sigmoid(self.linear2(hidden_))
        else:
            hidden = torch.relu(self.linear1(x))
            return torch.sigmoid(self.linear2(hidden))
    
    def loss(self, x, y):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.relu(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            logit = self.linear2(hidden_)
        else:
            hidden = torch.relu(self.linear1(x))
            logit = self.linear2(hidden)
        return self.BCE(logit, y)