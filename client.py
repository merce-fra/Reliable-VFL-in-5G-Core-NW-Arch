import flwr as fl
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from models import ClientModel
from typing import Optional, Union, Tuple, List, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import os
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import json

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data , data_test,n_run, optimized, latent_dim,reliability,power,device):
        self.latent_dim = latent_dim
        self.cid = cid
        self.n_run = n_run
        self.optimized = optimized
        self.device = device
        self.train = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.train.shape[1],latent_dim = self.latent_dim)
        self.test = torch.tensor(StandardScaler().fit_transform(data_test)).float()

        self.reliability = reliability
        self.power = power
        self.best_model_client = None


        try:
            saved_params = torch.load(f"model_weights/weights_optimized={self.optimized}_client_{self.cid}_n_run_{self.n_run}.pth", weights_only=True)
            with torch.no_grad():
                for param, saved_param in zip(self.model.parameters(), saved_params):
                    param.copy_(saved_param)
            print(f"Loaded saved model for client {self.cid}")
        except FileNotFoundError:
            print(f"No saved model found for client {self.cid}, using initialized model.")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.embedding = self.model(self.train)
    def get_parameters(self, config):
        pass

    


    def fit(self, parameters, config):
    # Retrieve values from the config dictionary
        
        best_value = config.get("best", 1.0)  

        os.makedirs("best_model_weights", exist_ok=True)

        if best_value == 0:
            print(f"Client {self.cid} saving model")
            torch.save([param.detach().clone() for param in self.model.parameters()],f"best_model_weights/model_weights_optimized={self.optimized}_client_{self.cid}_n_run_{self.n_run}.pth")

        prob = torch.rand(1).item()
        available = self.power if prob <= self.reliability else 0
        
        self.optimizer.zero_grad()
        self.embedding = self.model(self.train)
        if not self.embedding.requires_grad:
            self.embedding.requires_grad_(True)
        if available == 0:
            with torch.no_grad():
                self.embedding = torch.zeros_like(self.embedding)
        
        embedding_np = self.embedding.detach().cpu().numpy()
        result = [embedding_np, available, int(self.cid)]

        return result, len(self.train), {}
    

    def evaluate(self, parameters, config) -> Optional[Tuple[float, Dict[str, List]]]:
        # Determine client availability based on reliability
        prob = torch.rand(1).item()
        available = self.power if prob <= self.reliability else 0
        
        self.model.zero_grad()
        
        # Only backward pass if client is reliable/available
        client_gradients = torch.from_numpy(parameters[int(self.cid)])
        
        if available > 0:
                self.embedding.backward(client_gradients)

                self.optimizer.step()
        os.makedirs("model_weights", exist_ok=True)

        torch.save([param.detach().clone() for param in self.model.parameters()],
           f"model_weights/weights_optimized={self.optimized}_client_{self.cid}_n_run_{self.n_run}.pth")

        with torch.no_grad():
            self.embedding_test = self.model(self.test)

            
            # Zero out embeddings if client is unreliable
        if prob > self.reliability:
            self.embedding_test = torch.zeros_like(self.embedding_test)
        

        embedding_np = self.embedding_test.detach().cpu().numpy()
        json_embedding_str = json.dumps(embedding_np.tolist())
        

        del self.embedding_test
        

        return 0.0, len(self.test), {
            "params": json_embedding_str, 
            "availability": available,
            "id": int(self.cid)
        }



        
class FlowerClient_test(fl.client.NumPyClient):
    def __init__(self, cid, data, data_test, n_run, optimized, latent_dim, reliability, power, device):
        self.latent_dim = latent_dim
        self.cid = cid
        self.optimized = optimized
        self.n_run = n_run
        self.device = device
        self.train = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.train.shape[1], latent_dim=self.latent_dim)

        self.test = torch.tensor(StandardScaler().fit_transform(data_test)).float()
        self.reliability = reliability
        self.power = power
        self.best_model_client = None

        try:
            saved_params = torch.load(f"best_model_weights/model_weights_optimized={self.optimized}_client_{self.cid}_n_run_{self.n_run}.pth",weights_only=True)
        

            with torch.no_grad():
                for param, saved_param in zip(self.model.parameters(), saved_params):
                    param.copy_(saved_param)
        except FileNotFoundError:
        # File doesn't exist yet, using the initialized model
            print(f"No saved model found for client {self.cid}, using initialized model.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.embedding = self.model(self.train)
    def get_parameters(self, config):
        pass
    

   
    

    def evaluate(self, parameters, config) -> Optional[Tuple[float, Dict[str, List]]]:
        # Determine client availability based on reliability

        prob = torch.rand(1).item()
        available = self.power if prob <= self.reliability else 0
 
        

        with torch.no_grad():
            self.embedding_test = self.model(self.test)
            
        # Zero out embeddings if client is unreliable
        if prob > self.reliability:
            self.embedding_test = torch.zeros_like(self.embedding_test)
        

        embedding_np = self.embedding_test.detach().cpu().numpy()
        json_embedding_str = json.dumps(embedding_np.tolist())
        

        del self.embedding_test
        
        # Return results (loss, num_examples, metrics)
        return 0.0, len(self.test), {
            "params": json_embedding_str, 
            "availability": available,
            "id": int(self.cid)
        }

