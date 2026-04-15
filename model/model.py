import torch
from torch import nn
from torch.optim import Optimizer

class CIN(nn.Module):
    def __init__(self, m, d_model, layer_sizes):
        super().__init__()
        self.m = m
        self.d_model = d_model
        self.layer_sizes = layer_sizes
        self.filters = nn.ParameterList()
        
        prev_H = m
        for H_k in layer_sizes:
            W = nn.Parameter(torch.Tensor(H_k, prev_H, m))
            nn.init.kaiming_normal_(W)
            self.filters.append(W)
            prev_H = H_k

    def forward(self, X_0):
        X_k = X_0
        hidden_layers = []
        for W in self.filters:
            X_k = torch.einsum('fhm,bhd,bmd->bfd', W, X_k, X_0)
            hidden_layers.append(X_k)
            
        pooled_layers = [torch.sum(layer, dim=-1) for layer in hidden_layers]
        p_plus = torch.cat(pooled_layers, dim=-1) 
        return p_plus


class xDeepFM(nn.Module):
    def __init__(self, device, d_model, num_users, num_tweets, clip_dim):
        super().__init__()
        self.device = device
        self.d_model = d_model
        layer_sizes = [100, 100, 100]

        self.userIds = nn.Embedding(num_embeddings=num_users, embedding_dim=d_model)
        self.tweetIds = nn.Embedding(num_embeddings=num_tweets, embedding_dim=d_model)
        self.clip_projection = nn.Linear(clip_dim, d_model)

        self.linear = nn.Linear(self.d_model * 3, 1)

        self.dnn = nn.Sequential(
            nn.Linear(self.d_model * 3, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model),
        )

        self.cin = CIN(m=3, d_model=d_model, layer_sizes=layer_sizes)
        self.output_unit = nn.Linear(d_model + sum(layer_sizes) + 1, 1)

        num_parameters = sum([param.numel() for param in self.parameters()])
        print("xDeepFM parameters: ", num_parameters)
    
    def forward(self, user_id, tweet_id, clip_tensor) -> torch.Tensor:
        user_embeddings = self.userIds(user_id)
        tweet_embeddings = self.tweetIds(tweet_id)
        clip_embeddings = self.clip_projection(clip_tensor)

        cin_input_tensor = torch.stack([user_embeddings, tweet_embeddings, clip_embeddings], dim=1)
        _input_tensor = torch.cat([user_embeddings, tweet_embeddings, clip_embeddings], dim=-1)

        linear_result = self.linear(_input_tensor)
        deep_result = self.dnn(_input_tensor)          
        cin_result = self.cin(cin_input_tensor)        

        combined_features = torch.cat([linear_result, cin_result, deep_result], dim=-1)
        logits = self.output_unit(combined_features)
        return logits


class FTRL(Optimizer):
    def __init__(self, params, alpha=0.01, beta=1.0, l1=0.0, l2=0.0):
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['z'] = torch.zeros_like(p.data)
                    state['n'] = torch.zeros_like(p.data)

                z, n = state['z'], state['n']
                alpha, beta, l1, l2 = group['alpha'], group['beta'], group['l1'], group['l2']

                sigma = (torch.sqrt(n + grad ** 2) - torch.sqrt(n)) / alpha
                z.add_(grad - sigma * p.data)
                n.add_(grad ** 2)

                z_val = z.clone()
                mask = z_val.abs() > l1
                
                w = torch.zeros_like(p.data)
                w[mask] = - (z_val[mask] - z_val[mask].sign() * l1) / ((beta + torch.sqrt(n[mask])) / alpha + l2)
                p.data.copy_(w)

        return loss