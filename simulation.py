

import flwr as fl
import numpy as np
from strategy import Strategy,Strategy_test
from client import FlowerClient,FlowerClient_test
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from task import get_partitions_and_label
from read_data import read_params
import seaborn as sns
import os
from torch import rand
from flwr.common import Context
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from helper import get_power_of_two_indices, delete_model_weights
import torch




curr = os.getcwd()
curr = os.path.join(curr,'params.yml')
params = read_params(curr)
optimized_ = params.get('simulation').get('Optimized')
n_run = params.get('simulation').get('n_run')

latent_dim = params.get('simulation').get('latent_dim')
n_rounds = params.get('simulation').get('n_rounds')
n_rounds_test = params.get('simulation').get('n_rounds_test')
alpha = params.get('simulation').get('alpha')
beta = params.get('simulation').get('beta')
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for optimized in optimized_:
    torch.manual_seed(3)
    np.random.seed(3)
    for i in range(0,n_run):
        print(f"latent_dim = {latent_dim}, n_run = {n_run}, optimized = {optimized}") 

        partitions, partitions_test, label, label_test, n , reliability = get_partitions_and_label(params,i,optimized,device)
        for client_id in range(n):
            delete_model_weights(client_id,optimized,i)
        power= get_power_of_two_indices(reliability)
        if optimized == True:
            total_latent_dim = latent_dim

            relative_reliability = reliability / np.sum(reliability) 

            scaled_latent_dims = np.round(relative_reliability * total_latent_dim).astype(int)  

            difference = total_latent_dim - np.sum(scaled_latent_dims)

            for _ in range(abs(difference)):
                idx = np.argmax(scaled_latent_dims) if difference < 0 else np.argmin(scaled_latent_dims)
                scaled_latent_dims[idx] += 1 if difference > 0 else -1
        else:
            scaled_latent_dims = np.round(latent_dim / n ).astype(int)  
            scaled_latent_dims = [scaled_latent_dims]*n
            difference = latent_dim - sum(scaled_latent_dims)

            for _ in range(abs(difference)):
                idx = np.argmax(scaled_latent_dims) if difference < 0 else np.argmin(scaled_latent_dims)
                scaled_latent_dims[idx] += 1 if difference > 0 else -1
        
        print(f"Clients Embeddings Dimensions : {scaled_latent_dims}")

        def client_fn(cid):
            client_latent_dim = scaled_latent_dims[int(cid)]
            return FlowerClient(cid, partitions[int(cid)],partitions_test[int(cid)], i, optimized, client_latent_dim,reliability[int(cid)],power[int(cid)],device).to_client()
        
        client_app = fl.client.ClientApp(client_fn=client_fn)


        def server_fn(context: Context):
            server_config = ServerConfig(num_rounds=n_rounds)
            strategy = Strategy(label, label_test, i,n,scaled_latent_dims,optimized,n_rounds,device)
            return ServerAppComponents(
                strategy=strategy,
                config=server_config,
            )
            
        server_app = fl.server.ServerApp(server_fn=server_fn)
        
        # Start Flower server
        results = fl.simulation.run_simulation(
            client_app=client_app,
            server_app = server_app,
            num_supernodes=n,
        );

        def client_fn(cid):
            client_latent_dim = scaled_latent_dims[int(cid)]
            return FlowerClient_test(cid, partitions[int(cid)],partitions_test[int(cid)], i, optimized, client_latent_dim,reliability[int(cid)],power[int(cid)],device).to_client()

        client_app = fl.client.ClientApp(client_fn=client_fn)


        def server_fn(context: Context):
            server_config = ServerConfig(num_rounds=n_rounds_test)
            strategy = Strategy_test(label, label_test, i,n,scaled_latent_dims,optimized,n_rounds_test,params,device)
            return ServerAppComponents(
                strategy=strategy,
                config=server_config,
            )
            
        server_app = fl.server.ServerApp(server_fn=server_fn)
        
        # Start Flower server
        results = fl.simulation.run_simulation(
            client_app=client_app,
            server_app = server_app,
            num_supernodes=n,
        );
        
        if not optimized:
            results_dir = Path("_static/results")
            results_dir.mkdir(exist_ok=True)
            np.save(str(results_dir / f"results_not_optimized_{i}.npy"), results)
        
        else:
            results_dir = Path("_static/results")
            results_dir.mkdir(exist_ok=True)
            np.save(str(results_dir / f"results_optimized_{i}.npy"), results)





