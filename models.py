import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        
        # Wider network with no biases
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64, bias=False),
            nn.SELU(),

        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.SELU(),
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.SELU(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(16, 4, bias=False),
            nn.SELU(),
            nn.Linear(4, 1, bias=False)
        )
        
        # Initialize weights using xavier uniform
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')  # Kaiming Normal for weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Bias initialized to zero (standard approach)
    
    
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Forward pass without residual connections
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return self.output(x3)

class ClientModel(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(ClientModel, self).__init__()
        
        # Encoder without residual connections and no biases
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, 64, bias=False),
            nn.SELU(),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.SELU(),
        )
        
        self.encoder3 = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.SELU(),
        )
        
        self.latent_projection = nn.Sequential(
            nn.Linear(16, latent_dim, bias=False),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')  # Kaiming Normal for weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Bias initialized to zero (standard approach)
    
    
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Forward pass without residual connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        latent = self.latent_projection(e3)
        
        return latent








# class ServerModel(nn.Module):
#     def __init__(self, input_size, device=None):
#         super(ServerModel, self).__init__()
        
#         # Set device
#         self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Wider network with no biases
#         self.layer1 = nn.Sequential(
#             nn.Linear(input_size, 32, bias=False),
#             nn.SELU(),
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Linear(32, 16, bias=False),
#             nn.SELU(),
#         )
        
#         self.layer3 = nn.Sequential(
#             nn.Linear(16, 8, bias=False),
#             nn.SELU(),
#         )
        
#         self.output = nn.Sequential(
#             nn.Linear(8, 4, bias=False),
#             nn.SELU(),
#             nn.Linear(4, 1, bias=False)
#         )
        
#         # Initialize weights using kaiming normal
#         self.apply(self._init_weights)
        
#         # Move model to specified device
#         self.to(self.device)
    
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')  # Kaiming Normal for weights
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)  # Bias initialized to zero (standard approach)
    
#     def forward(self, x):
#         if not torch.is_tensor(x):
#             x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         elif x.device != self.device:
#             x = x.to(self.device)
            
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
            
#         # Forward pass without residual connections
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         return self.output(x3)


# class ClientModel(nn.Module):
#     def __init__(self, input_size, latent_dim, device=None):
#         super(ClientModel, self).__init__()
        
#         # Set device
#         self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Encoder without residual connections and no biases
#         self.encoder1 = nn.Sequential(
#             nn.Linear(input_size, 64, bias=False),
#             nn.SELU(),
#         )
        
#         self.encoder2 = nn.Sequential(
#             nn.Linear(64, 32, bias=False),
#             nn.SELU(),
#         )
        
#         self.encoder3 = nn.Sequential(
#             nn.Linear(32, 16, bias=False),
#             nn.SELU(),
#         )
        
#         self.latent_projection = nn.Sequential(
#             nn.Linear(16, latent_dim, bias=False),
#         )
        
#         # Initialize weights
#         self.apply(self._init_weights)
        
#         # Move model to specified device
#         self.to(self.device)
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')  # Kaiming Normal for weights
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)  # Bias initialized to zero (standard approach)
    
#     def forward(self, x):
#         if not torch.is_tensor(x):
#             x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         elif x.device != self.device:
#             x = x.to(self.device)
            
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
            
#         # Forward pass without residual connections
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         latent = self.latent_projection(e3)
        
#         return latent