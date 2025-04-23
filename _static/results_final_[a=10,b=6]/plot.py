import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import matplotlib.patches as mpatches

ns = 10
def analyze_multiple_simulations(simulation_numbers=None):
    """
    Analyze multiple simulation results and average them.
    
    Parameters:
    simulation_numbers (list): List of simulation numbers to process. If None, use all available.
    """
    # If no specific simulation numbers are provided, find all available files
    if simulation_numbers is None:
        # Get all simulation files and extract unique simulation numbers
        not_opt_files = glob.glob('final_power_results_not_optimized_*.npy')
        opt_files = glob.glob('final_power_results_optimized_*.npy')
        
        # Extract simulation numbers from filenames
        all_numbers = []
        for file in not_opt_files + opt_files:
            num = int(file.split('_')[-1].split('.')[0])
            all_numbers.append(num)
        
        simulation_numbers = sorted(list(set(all_numbers)))
    
    print(f"Processing simulations: {simulation_numbers}")
    
    # Lists to store results from each simulation
    all_results = {
        'not_optimized': [],
        'optimized': []
    }
    
    # Process each simulation
    for sim_num in simulation_numbers:
        # File paths for this simulation
        not_opt_ids_file = f'final_power_results_not_optimized_{sim_num}.npy'
        not_opt_entries_file = f'final_test_results_not_optimized_{sim_num}.npy'
        opt_ids_file = f'final_power_results_optimized_{sim_num}.npy'
        opt_entries_file = f'final_test_results_optimized_{sim_num}.npy'
        
        # Check if all files exist
        if not all(os.path.exists(f) for f in [not_opt_ids_file, not_opt_entries_file, opt_ids_file, opt_entries_file]):
            print(f"Skipping simulation {sim_num} - not all files exist")
            continue
        
        # Load data
        ids1 = np.load(not_opt_ids_file)
        entries1 = np.load(not_opt_entries_file)
        ids2 = np.load(opt_ids_file)
        entries2 = np.load(opt_entries_file)
        
        # First filter out ID 0 for both datasets
        mask1 = ids1 != 0
        mask2 = ids2 != 0
        ids1_filtered = ids1[mask1]
        entries1_filtered = entries1[mask1]
        ids2_filtered = ids2[mask2]
        entries2_filtered = entries2[mask2]
        
        # Get a combined set of all unique IDs (excluding ID 0)
        all_filtered_ids = np.concatenate((ids1_filtered, ids2_filtered))
        all_unique_ids = np.unique(all_filtered_ids)
        
        # Calculate frequencies based on combined occurrences across both scenarios
        combined_freq_dict = {}
        total_combined_count = len(all_filtered_ids)
        
        if total_combined_count > 0:
            for unique_id in all_unique_ids:
                # Count occurrences in both datasets
                count = np.sum(all_filtered_ids == unique_id)
                combined_freq_dict[unique_id] = count / total_combined_count
        
        # Calculate average entries for each dataset
        avg_entries1 = {}
        avg_entries2 = {}
        weighted_entries1 = {}
        weighted_entries2 = {}
        
        # For not optimized dataset
        for unique_id in all_unique_ids:
            indices = np.where(ids1_filtered == unique_id)[0]
            if len(indices) > 0:
                avg_val = np.mean(entries1_filtered[indices])
                avg_entries1[unique_id] = avg_val
                # Use the combined frequency for weighting
                weighted_entries1[unique_id] = avg_val * combined_freq_dict[unique_id]
            else:
                avg_entries1[unique_id] = 0
                weighted_entries1[unique_id] = 0
        
        # For optimized dataset
        for unique_id in all_unique_ids:
            indices = np.where(ids2_filtered == unique_id)[0]
            if len(indices) > 0:
                avg_val = np.mean(entries2_filtered[indices])
                avg_entries2[unique_id] = avg_val
                # Use the combined frequency for weighting
                weighted_entries2[unique_id] = avg_val * combined_freq_dict[unique_id]
            else:
                avg_entries2[unique_id] = 0
                weighted_entries2[unique_id] = 0
        
        # Store simulation results - separately for each scenario but with combined frequencies
        not_opt_result = {
            'unique_ids': all_unique_ids,
            'avg_entries': avg_entries1,
            'weighted_entries': weighted_entries1,
            'combined_freq_dict': combined_freq_dict
        }
        
        opt_result = {
            'unique_ids': all_unique_ids,
            'avg_entries': avg_entries2,
            'weighted_entries': weighted_entries2,
            'combined_freq_dict': combined_freq_dict
        }
        
        # Store results separately for each scenario
        all_results['not_optimized'].append(not_opt_result)
        all_results['optimized'].append(opt_result)
        
        print(f"Processed simulation {sim_num}")
    
    # Get a master list of all unique IDs across all simulations (excluding 0)
    master_unique_ids = set()
    for result in all_results['not_optimized']:
        master_unique_ids.update(result['unique_ids'])
    for result in all_results['optimized']:
        master_unique_ids.update(result['unique_ids'])
    
    # Ensure we don't include ID 0 in analysis (should already be excluded)
    if 0 in master_unique_ids:
        master_unique_ids.remove(0)
    # if 1 in master_unique_ids:
    #     master_unique_ids.remove(1)
    # if 2 in master_unique_ids:
    #     master_unique_ids.remove(2)

        
    master_unique_ids = sorted(list(master_unique_ids))
    
    # Average results across all simulations, keeping scenarios separate
    avg_results = {
        'unique_ids': np.array(master_unique_ids),
        'avg_not_optimized': {},
        'avg_optimized': {},
        'weighted_not_optimized': {},
        'weighted_optimized': {},
        'avg_combined_freq': {}
    }
    
    # Initialize counters for each ID
    id_counters = {id_val: 0 for id_val in master_unique_ids}
    
    # Initialize sums for combined frequency
    combined_freq_sums = {id_val: 0 for id_val in master_unique_ids}
    
    # Sum up values across simulations for both scenarios
    for id_val in master_unique_ids:
        avg_not_opt_sum = 0
        avg_opt_sum = 0
        weighted_not_opt_sum = 0
        weighted_opt_sum = 0
        
        for idx, (not_opt_result, opt_result) in enumerate(zip(all_results['not_optimized'], all_results['optimized'])):
            # Update combined frequency sums
            if id_val in not_opt_result['combined_freq_dict']:
                combined_freq_sums[id_val] += not_opt_result['combined_freq_dict'][id_val]
                id_counters[id_val] += 1
                
            # Update not optimized sums
            if id_val in not_opt_result['avg_entries']:
                avg_not_opt_sum += not_opt_result['avg_entries'][id_val]
                weighted_not_opt_sum += not_opt_result['weighted_entries'][id_val]
            
            # Update optimized sums
            if id_val in opt_result['avg_entries']:
                avg_opt_sum += opt_result['avg_entries'][id_val]
                weighted_opt_sum += opt_result['weighted_entries'][id_val]
        
        # if avg_not_opt_sum == 0 and avg_opt_sum != 0:
        #     avg_not_opt_sum = avg_opt_sum
        # elif avg_opt_sum == 0 and avg_not_opt_sum != 0:
        #     avg_opt_sum = avg_not_opt_sum

        # if weighted_not_opt_sum == 0 and weighted_opt_sum != 0:
        #     weighted_not_opt_sum = weighted_opt_sum
        # elif weighted_opt_sum == 0 and weighted_not_opt_sum != 0:
        #     weighted_opt_sum = weighted_not_opt_sum
            
        # Calculate averages (handling the case where an ID might not be in all simulations)
        if id_counters[id_val] > 0:
            avg_results['avg_not_optimized'][id_val] = avg_not_opt_sum / id_counters[id_val]
            avg_results['avg_optimized'][id_val] = avg_opt_sum / id_counters[id_val]
            avg_results['weighted_not_optimized'][id_val] = weighted_not_opt_sum / id_counters[id_val]
            avg_results['weighted_optimized'][id_val] = weighted_opt_sum / id_counters[id_val]
            avg_results['avg_combined_freq'][id_val] = combined_freq_sums[id_val] / id_counters[id_val]
        else:
            # Handle IDs that were never found in the dataset
            avg_results['avg_not_optimized'][id_val] = 0
            avg_results['avg_optimized'][id_val] = 0
            avg_results['weighted_not_optimized'][id_val] = 0
            avg_results['weighted_optimized'][id_val] = 0
            avg_results['avg_combined_freq'][id_val] = 0
    
    # Return the averaged results
    return avg_results


def plot_averaged_histogram_results(avg_results):
    """Plot the averaged results as histograms with optimized settings for publication."""
    
    # Sort unique IDs for consistent plotting
    unique_ids = np.sort(avg_results['unique_ids'])
    
    # Prepare data for plotting
    avg_not_opt = [avg_results['avg_not_optimized'].get(id_val, 0) for id_val in unique_ids]
    avg_opt = [avg_results['avg_optimized'].get(id_val, 0) for id_val in unique_ids]
    weighted_not_opt = [avg_results['weighted_not_optimized'].get(id_val, 0) for id_val in unique_ids]
    weighted_opt = [avg_results['weighted_optimized'].get(id_val, 0) for id_val in unique_ids]
    combined_freq = [avg_results['avg_combined_freq'].get(id_val, 0) for id_val in unique_ids]
    
    # Convert IDs to strings for categorical plotting
    ids_str = [str(int(id_val)) for id_val in unique_ids]
    
    # Set global font size for clarity
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    bar_width = 0.35
    x = np.arange(len(ids_str))

    # --- Plot 1: Average Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - bar_width/2, avg_not_opt, bar_width, label='SoTA', color='tab:blue', alpha=0.7)
    ax.bar(x + bar_width/2, avg_opt, bar_width, label='Proposed', color='tab:orange', alpha=0.7)
    ax.set_title('Average Loss Across Simulations')
    ax.set_ylabel('Average Loss')
    ax.set_xlabel('Participating NWDAFs ID/tag Sum')
    ax.set_xticks(x)
    ax.set_xticklabels(ids_str, rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig('average_loss.pdf', format='pdf', dpi=300)
    plt.savefig('average_loss.svg', format='svg', dpi=300)

    # --- Plot 2: Frequency Distribution ---
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(x)
    ax.bar(x, combined_freq, color='tab:green', alpha=0.7)
    ax.set_title('Participating NWDAFs ID Distribution')
    ax.set_ylabel('Relative Frequency (Density)')
    ax.set_xlabel('Participating NWDAFs ID/tag Sum')
    ax.set_xticklabels(ids_str, rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig('frequency_distribution.pdf', format='pdf', dpi=300)
    plt.savefig('frequency_distribution.svg', format='svg', dpi=300)

    # --- Plot 3: Loss Distribution (Weighted) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    total_contribution_not_opt = [w * f for w, f in zip(avg_not_opt, combined_freq)]
    total_contribution_opt = [w * f for w, f in zip(avg_opt, combined_freq)]
    
    # ax.bar(x - bar_width/2, total_contribution_not_opt, bar_width, label='SoTA', color='tab:blue', alpha=0.7)
    # ax.bar(x + bar_width/2, total_contribution_opt, bar_width, label='Proposed', color='tab:orange', alpha=0.7)


    bars = ax.bar(x - bar_width/2, total_contribution_not_opt, bar_width, label='SoTA', color='tab:blue', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            va = 'bottom'
            y_pos = height + 0.002
        else:
            va = 'top'
            y_pos = height - 0.003  # Adjusted to prevent overlap
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}', ha='center', va=va, fontsize=7)
        
    bars = ax.bar(x + bar_width/2, total_contribution_opt, bar_width, label='Proposed', color='tab:orange', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            va = 'bottom'
            y_pos = height + 0.001
        else:
            va = 'top'
            y_pos = height - 0.003  # Adjusted to prevent overlap
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}', ha='center', va=va, fontsize=7)
        

    ax.set_title('Performance Comparison: Test Loss - Proposed Method vs. SoTA VFL\n'
                'Weighted by ID Sum Frequencies - Beta(10,6) $\\boldsymbol{[\mu = 0.625, \sigma^2 = 0.014]}$  ', fontsize=15, fontweight='bold')
    ax.set_xlabel('Participating NWDAFs ID/tag Sum', fontsize=18, fontweight='bold')
    ax.set_ylabel('Weighted Loss', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ids_str, rotation=45)
    
    ax.legend()
    plt.tight_layout()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    
    plt.savefig('weighted_loss.pdf', format='pdf', dpi=300)
    plt.savefig('weighted_loss.svg', format='svg', dpi=300)

    # --- Plot 4: Improvement Analysis ---
    differences = [o - n for o, n in zip(avg_opt, avg_not_opt)]
    weighted_differences = [o - n for o, n in zip(total_contribution_opt, total_contribution_not_opt)]

    print(f"weighted differences {weighted_differences}")

    # Use the filtered unique_ids instead of hardcoded range
    ids = np.array(unique_ids)
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14})
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color list based on value
    sota_blue = 'tab:blue'  # Blue color similar to the reference image
    proposed_orange = 'tab:orange'  # Orange color similar to the reference image

    # Apply colors based on values
    colors = [sota_blue if x > 0 else proposed_orange for x in weighted_differences]

    # Also update the legend to match these colors
    blue_patch = mpatches.Patch(color=sota_blue, label='Degradation compared to SoTA')
    orange_patch = mpatches.Patch(color=proposed_orange, label='Improvement over SoTA')
    ax.legend(handles=[blue_patch, orange_patch], 
            loc='upper right', 
            bbox_to_anchor=(0.99, 0.99),
            prop={'size': 14})
    # Plot bars
    bars = ax.bar(ids, weighted_differences, color=colors, width=0.7)

# Add data labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            va = 'bottom'
            y_pos = height + 0.002
        else:
            va = 'top'
            y_pos = height - 0.003  # Adjusted to prevent overlap
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.3f}', ha='center', va=va, fontsize=12)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label font size
    # ax.set_xlim(1.1,15.8)
    # Add labels and title
    ax.set_xlabel('Participating NWDAFs ID/tag Sum', fontsize=18, fontweight='bold')
    ax.set_ylabel('Difference in Weighted Loss', fontsize=18, fontweight='bold')
    ax.set_title('Performance Comparison: Proposed Method vs. SoTA VFL\n'
                'Weighted by ID Sum Frequencies - Beta(10,6) $\\boldsymbol{[\mu = 0.625, \sigma^2 = 0.014]}$  ', fontsize=15, fontweight='bold')
    # Set y-axis limits with appropriate padding
    min_val = min(weighted_differences)
    max_val = max(weighted_differences)
    y_range = max_val - min_val
    ax.set_ylim(min_val - 0.025, max_val + 0.025)  # Adjusted to fit all bars and labels

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Ensure everything fits
    plt.tight_layout()
    plt.savefig('improvement_distribution.pdf', format='pdf', dpi=1200)
    plt.savefig('improvement_distribution.svg', format='svg', dpi=1200)
    # Show plot
    plt.show()

    # Print summary statistics
    avg_not_opt_total = sum(weighted_not_opt)
    avg_opt_total = sum(weighted_opt)
    improvement = np.abs(sum(total_contribution_not_opt))- np.abs(sum(total_contribution_opt))

    print("Summary across all simulations:")
    print(f"Number of unique IDs: {len(unique_ids)}")
    print("\nAverage Performance:")
    print(f"- SoTA Avg. Loss during a test session: {avg_not_opt_total:.4f}")
    print(f"- Proposed Avg. Loss during a test session: {avg_opt_total:.4f}")

    if avg_not_opt_total != 0:
        rel_improvement = improvement / np.abs(sum(total_contribution_not_opt)) * 100
        print(f"Improvement: {improvement:.4f} ({rel_improvement:.2f}%)")
    else:
        print(f"Improvement: {improvement:.4f}")

    print("\nNote on frequencies:")
    print("Frequencies calculated based on combined occurrences across both scenarios")
    print("All calculations performed only on non-zero IDs")
    
    plt.show()


            
    # bars = ax.bar(x, weighted_differences, color='purple', alpha=0.7)
    
    # # Highlight negative values in red
    # for i, diff in enumerate(weighted_differences):
    #     if diff < 0:
    #         bars[i].set_color('red')
    
    # ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # ax.set_title('Proposed - SoTA Loss | Weighted by ID sum Frequencies (Scenario 1, Beta(8,2)) ')
    # ax.set_xlabel('Participating NWDAFs ID/tag Sum')
    # ax.set_ylabel('Difference in Weighted Loss')
    # ax.set_xticks(x)
    # ax.set_xticklabels(ids_str, rotation=45)
    # ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    # plt.tight_layout()

n_runs = np.arange(0,ns)
if __name__ == "__main__":
    # Analyze all available simulations and get averaged results
    print("Analyzing multiple simulations...")
    averaged_results = analyze_multiple_simulations(n_runs)
    
    # Plot the averaged results as histograms
    print("Creating histogram plots of averaged results...")
    plot_averaged_histogram_results(averaged_results)
