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
    sorted_values = sorted(values)
    
    # Create a mapping from value to position in sorted list
    # For duplicates, this will keep the last occurrence
    value_to_position = {val: i for i, val in enumerate(sorted_values)}
    
    result = [2 ** value_to_position[val] for val in values]
    
    return result




def concatenate_embeddings_by_client_order(embedding_results, order_tensors):
    """
    Concatenate client embeddings based on their client IDs.
    
    Args:
        embedding_results: List of embedding tensors from clients.
        order_tensors: List of tensors or integers containing client IDs.
        
    Returns:
        Tensor with concatenated embeddings in consistent client ID order.
    """
    if isinstance(order_tensors[0], torch.Tensor):
        client_ids = [order_tensor.item() if order_tensor.numel() == 1 else order_tensor[0].item() 
                      for order_tensor in order_tensors]
    else:
        client_ids = order_tensors  # Assume it's already a list of integers
    
    client_embeddings = list(zip(client_ids, embedding_results))

    sorted_client_embeddings = sorted(client_embeddings, key=lambda x: x[0])

    sorted_embeddings = [embedding for _, embedding in sorted_client_embeddings]
    
    embeddings_aggregated = torch.cat(sorted_embeddings, dim=1)
    return embeddings_aggregated


def delete_model_weights(client_id,optimized,n_run):


    file_s = f"best_model_weights/model_weights_optimized={optimized}_server_{n_run}.pth"
    
    if os.path.exists(file_s):
        os.remove(file_s)
        print(f"Deleted server file: {file_s}")


    file_path = f"model_weights/weights_optimized={optimized}_client_{client_id}_n_run_{n_run}.pth"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted client file: {file_path}")
    else:
        print(f"File not found: {file_path}")
    
    

    file_p = f"best_model_weights/model_weights_optimized={optimized}_client_{client_id}_n_run_{n_run}.pth"

    if os.path.exists(file_p):
        os.remove(file_p)
        print(f"Deleted client best model file: {file_p}")
    else:
        print(f"File not found: {file_p}")


