import flwr as fl
import torch
import io
import numpy as np
import os
from pathlib import Path
import torch.nn.functional as F
from read_data import read_params
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from typing import Optional, Union, Tuple, List, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
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
import pickle
from models import ServerModel
import json
import numpy as np
curr = os.getcwd()
curr = os.path.join(curr,'params.yml')
params = read_params(curr)
optimized = params.get('simulation').get('Optimized')
from helper import concatenate_embeddings_by_client_order
     

class Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        labels,
        label_test,
        n_run,
        num_clients,
        latent_dim,
        optimized,
        n_rounds,
        device,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        proximal_mu=10,  
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.test_loss = []
        self.training_loss = []
        self.test_label = label_test
        self.latent_dim = latent_dim
        self.num_clients = num_clients
        self.n_run = n_run
        self.optimized = optimized
        self.n_rounds = n_rounds
        self.min_loss = 0
        self.i = 0
        self.device = device
        self.power = []
        self.proximal_mu = proximal_mu
        self.best = 1
        self.previous_model_state = None
        

        self.model = ServerModel(sum(self.latent_dim))



        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.05)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)  # 0.5% decay
        self.criterion = nn.HuberLoss(reduction='mean',delta=1.5)
        self.label = torch.tensor(labels).float().unsqueeze(1)
        self.best_model = [param.detach().clone() for param in self.model.parameters()]
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get all available clients
        client_manager.wait_for(num_clients = self.num_clients,timeout=15)
        clients = client_manager.all().values()

        # Create configuration with best value indicating if best model is found and round number
        config = {
            "best": float(self.best),
            "round": server_round  # The current round index
        }
        
        # Create fit instructions with parameters and config
        fit_ins = FitIns(parameters, config)
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
    self,
    rnd,
    results,
    failures,):
        # Clean up any leftover gradients from previous rounds
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]

        availability = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[1])
            for _, fit_res in results
        ]
        
        order = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[2])
            for _, fit_res in results
        ]

        embeddings_aggregated = concatenate_embeddings_by_client_order(embedding_results, order)
        
        # Compute total availability before any expensive operations
        total_availability = sum(availability)

        
        # Prepare the embedding for model input
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        output = self.model(embedding_server)
        
        # Main task loss
        task_loss = self.criterion(output, self.label)
        
        
        if total_availability > 0:
            print(f"availability = {availability}")
            
            # Clear gradients before backward pass
            self.optimizer.zero_grad()
            
            # Compute gradients
            task_loss.backward()
            
            # Update model parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Get gradients, detach and convert to numpy
            grads = embedding_server.grad.split(list(self.latent_dim), dim=1)
            np_grads = [grad.detach().cpu().numpy() for grad in grads]
            parameters_aggregated = ndarrays_to_parameters(np_grads)
            del grads

        else:
            print("All clients Failed, failure to update model")
            
            # No need to compute gradients when all clients failed
            with torch.no_grad():
                zero_grads = [torch.zeros_like(embed).cpu().numpy() 
                            for embed in embedding_server.split(list(self.latent_dim), dim=1)]
                parameters_aggregated = ndarrays_to_parameters(zero_grads)
                del zero_grads
            
        
        # Store loss for tracking
        self.training_loss.append(task_loss.item())  
        
        metrics_aggregated = {
            "loss_train": task_loss.item(),
            "total_loss": task_loss.item(),
            "available_clients": total_availability
        }
        
        # Clean up tensors to free memory
        del embedding_server
        del embeddings_aggregated
        del embedding_results
        
        # Save results at the end of training
        if rnd == self.n_rounds:
            results_dir = Path("_static/results")
            results_dir.mkdir(exist_ok=True)
            if self.optimized:
                np.save(str(results_dir / f"train_results_optimized_{self.n_run}.npy"), self.training_loss)
            else:
                np.save(str(results_dir / f"train_results_not_optimized_{self.n_run}.npy"), self.training_loss)
        
        return parameters_aggregated, metrics_aggregated
    



    def evaluate_config(self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get all available clients
        client_manager.wait_for(num_clients = self.num_clients,timeout=15)
        clients = client_manager.all().values()

        # Create configuration with best value and round number
        config = {
            "round": server_round  # The current round index
        }
        
        # Create fit instructions with parameters and config
        fit_ins = FitIns(parameters, config)
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    






    def aggregate_evaluate(
    self,
    server_round,
    results,
    failures):
        self.best = 1
    
        availability_ = []
        order_ = []
        embeddings = []
        for _,j in results:
            embeddings.append(torch.tensor(np.array(json.loads(j.metrics["params"])),dtype=torch.float32))
            availability_.append(j.metrics["availability"])
            order_.append(j.metrics["id"])
        print("test clients availability")
        
        print(f"order = {order_}, availability_ = {availability_}, sum = {sum(availability_)}")
        self.power.append(sum(availability_))
        embedding=concatenate_embeddings_by_client_order(embeddings,order_)
        # embedding=torch.cat(embeddings, dim=1)
        with torch.no_grad(): 
            outputs = self.model(embedding)
            loss_test = self.criterion(torch.squeeze(outputs), torch.Tensor(self.test_label))
        if server_round == 1 or self.min_loss > loss_test.item():
            print(f"self.min_loss = {self.min_loss}")
            torch.save([param.detach().clone() for param in self.model.parameters()],f"model_weights_optimized={self.optimized}_server_{self.n_run}.pth")
            self.best = 0
            self.min_loss = loss_test.item()
        test_loss = {"loss_test": loss_test}
        print(f"test loss = {loss_test}")
        self.test_loss.append(loss_test)
        if server_round == self.n_rounds:
            results_dir = Path("_static/results")
            results_dir.mkdir(exist_ok=True)
            if self.optimized == True:

                np.save(str(results_dir / f"test_results_optimized_{self.n_run}.npy"), self.test_loss)
                np.save(str(results_dir / f"power_results_optimized_{self.n_run}.npy"), self.power)
            else:
                np.save(str(results_dir / f"test_results_not_optimized_{self.n_run}.npy"), self.test_loss)
                np.save(str(results_dir / f"power_results_not_optimized_{self.n_run}.npy"), self.power)
        return   test_loss, {} 



class Strategy_test(fl.server.strategy.FedAvg):
    def __init__(
        self,
        labels,
        label_test,
        n_run,
        num_clients,
        latent_dim,
        optimized,
        n_rounds,
        device,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        proximal_mu=10,  # Proximal term coefficient
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.test_loss = []
        self.training_loss = []
        self.test_label = label_test
        self.latent_dim = latent_dim
        self.num_clients = num_clients
        self.n_run = n_run
        self.optimized = optimized
        self.n_rounds = n_rounds
        self.min_loss = 0
        self.i = 0
        self.device = device
        
        self.power = []
        self.proximal_mu = proximal_mu
        self.model = ServerModel(sum(self.latent_dim))

        # Load the saved parameters
        saved_params = torch.load(f"model_weights_optimized={self.optimized}_server_{self.n_run}.pth", weights_only=True)

        #  Assign the saved parameters to the model
        with torch.no_grad():
            for param, saved_param in zip(self.model.parameters(), saved_params):
                param.copy_(saved_param)

        #  use state_dict() for getting initial_parameters
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
      

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)  # 0.5% decay
        self.criterion = nn.HuberLoss(reduction='mean', delta=1.6)
        self.label = torch.tensor(labels).float().unsqueeze(1)



    def aggregate_evaluate(
    self,
    server_round,
    results,
    failures):
        
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Test phase - run = {self.n_run}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        availability_ = []
        order_ = []
        embeddings = []
        
        for _,j in results:
            embeddings.append(torch.tensor(np.array(json.loads(j.metrics["params"])),dtype=torch.float32))
            availability_.append(j.metrics["availability"])
            order_.append(j.metrics["id"])
        print("test clients availability")
        
        print(f"order = {order_}, availability_ = {availability_}, sum = {sum(availability_)}")
        self.power.append(sum(availability_))
        embedding=concatenate_embeddings_by_client_order(embeddings,order_)
        # embedding=torch.cat(embeddings, dim=1)
        with torch.no_grad(): 
            outputs = self.model(embedding)
            loss_test = self.criterion(torch.squeeze(outputs), torch.Tensor(self.test_label))

        test_loss = {"loss_test": loss_test}
        print(f"test loss = {loss_test}")
        self.test_loss.append(loss_test)
        if server_round == self.n_rounds:
            results_dir = Path("_static/results_final")
            results_dir.mkdir(exist_ok=True)
            if self.optimized == True:
                np.save(str(results_dir / f"final_test_results_optimized_{self.n_run}.npy"), self.test_loss)
                np.save(str(results_dir / f"final_power_results_optimized_{self.n_run}.npy"), self.power)
            else:
                np.save(str(results_dir / f"final_test_results_not_optimized_{self.n_run}.npy"), self.test_loss)
                np.save(str(results_dir / f"final_power_results_not_optimized_{self.n_run}.npy"), self.power)
        return   test_loss, {} 