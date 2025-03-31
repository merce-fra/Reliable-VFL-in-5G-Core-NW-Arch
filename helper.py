import torch
import os
def get_power_of_two_indices(values):
    """
    Takes a list of values and returns a list where each element is 2^position
    based on the position of that value in the sorted list.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of powers of 2 corresponding to each value's position if sorted
    """
    # Create a sorted version of the input list
    sorted_values = sorted(values)
    
    # Create a mapping from value to position in sorted list
    # For duplicates, this will keep the last occurrence
    value_to_position = {val: i for i, val in enumerate(sorted_values)}
    
    # Calculate 2^position for each value
    result = [2 ** value_to_position[val] for val in values]
    
    return result

# def concatenate_embeddings_by_client_order(embedding_results, order_tensors):
#     """
#     Concatenate client embeddings based on their client IDs.
    
#     Args:
#         embedding_results: List of embedding tensors from clients
#         order_tensors: List of tensors containing client IDs
        
#     Returns:
#         Tensor with concatenated embeddings in consistent client ID order
#     """
#     # Extract client IDs from tensors
#     client_ids = [order_tensor.item() if order_tensor.numel() == 1 else order_tensor[0].item() 
#                  for order_tensor in order_tensors]
    
#     # Create (client_id, embedding) pairs
#     client_embeddings = list(zip(client_ids, embedding_results))
    
#     # Sort by client ID to ensure consistent order
#     sorted_client_embeddings = sorted(client_embeddings, key=lambda x: x[0])
    
#     # Extract sorted embeddings
#     sorted_embeddings = [embedding for _, embedding in sorted_client_embeddings]
    
#     # Concatenate along dimension 1
#     embeddings_aggregated = torch.cat(sorted_embeddings, dim=1)
    
#     return embeddings_aggregated




def concatenate_embeddings_by_client_order(embedding_results, order_tensors):
    """
    Concatenate client embeddings based on their client IDs.
    
    Args:
        embedding_results: List of embedding tensors from clients.
        order_tensors: List of tensors or integers containing client IDs.
        
    Returns:
        Tensor with concatenated embeddings in consistent client ID order.
    """
    # Ensure order_tensors is a list of integers
    if isinstance(order_tensors[0], torch.Tensor):
        client_ids = [order_tensor.item() if order_tensor.numel() == 1 else order_tensor[0].item() 
                      for order_tensor in order_tensors]
    else:
        client_ids = order_tensors  # Assume it's already a list of integers
    
    # Create (client_id, embedding) pairs
    client_embeddings = list(zip(client_ids, embedding_results))

    # Sort by client ID to ensure consistent order
    sorted_client_embeddings = sorted(client_embeddings, key=lambda x: x[0])

    # Extract sorted embeddings
    sorted_embeddings = [embedding for _, embedding in sorted_client_embeddings]
    
    # Concatenate along dimension 1
    embeddings_aggregated = torch.cat(sorted_embeddings, dim=1)
    return embeddings_aggregated


def delete_model_weights(client_id,optimized,n_run):

    file_path = f"weights_optimized={optimized}_client_{client_id}_n_run_{n_run}.pth"
    
    # Check if the file exists and delete it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File not found: {file_path}")

# Delete files for clients 0 to 3 at the start of the simulation
