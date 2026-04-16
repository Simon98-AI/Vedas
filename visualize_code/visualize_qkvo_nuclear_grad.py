import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False  
plt.rcParams['mathtext.fontset'] = 'stix'  

split = "easy" # hard
BASE_PATH = "./qwen_vl/{}_qkv_grad".format(split) 
SAVE_DIR = os.path.join("./qwen_vl", "grad_nuclear_viz_sum")
os.makedirs(SAVE_DIR, exist_ok=True)

PARAM_MAP = {
    'q_proj.lora_A': ('Q', 'A'), 'q_proj.lora_B': ('Q', 'B'),
    'k_proj.lora_A': ('K', 'A'), 'k_proj.lora_B': ('K', 'B'),
    'v_proj.lora_A': ('V', 'A'), 'v_proj.lora_B': ('V', 'B'),
    'o_proj.lora_A': ('O', 'A'), 'o_proj.lora_B': ('O', 'B'),
}

EPOCHS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7]
PROJ_TYPES = ['Q', 'K', 'V', 'O']
PROJ_COLORS = {
    'Q': 'orange',  
    'K': '#C44E52',  
    'V': 'green',  
    'O': '#8A7090',  
}

PROJ_MARKERS = {'Q': 's', 'K': 's', 'V': 's', 'O': 's'}


METRIC_NUCLEAR = 'avg_nuclear_norm'      
METRIC_COMPLEXITY = 'avg_complexity_ratio'  

FIGSIZE = (7, 6)
DPI = 300
LINE_WIDTH = 4.0


def load_all_epochs(base_path: str, epochs: list) -> dict:

    all_data = {}
    for epoch in epochs:
        filepath = os.path.join(base_path, f"grad_stats_epoch_{epoch}.pkl")
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_data[epoch] = data
            n_layers = len(data)
            n_params = sum(len(p) for p in data.values())
            print(f"loading epoch {epoch}: {n_layers} layers, {n_params} param-stats")
    return all_data


def aggregate_summed_by_proj(all_epoch_data, epochs: list) -> dict:
    """
    return: {epoch: {proj_type: {'nuclear_norm_sum': value, 'complexity_ratio_avg': value, 'count': n}}}
    """
    aggregated = defaultdict(lambda: defaultdict(dict))
    
    for epoch in epochs:
        if epoch not in all_epoch_data:
            continue
        
        epoch_data = all_epoch_data[epoch]  # {layer_idx: {param_key: stats}}
        
        temp = defaultdict(lambda: {'nuclear_A': [], 'nuclear_B': [], 'complexity': []})
        
        for layer_idx, layer_params in epoch_data.items():
            for param_key, stats in layer_params.items():
                if param_key not in PARAM_MAP:
                    continue
                
                proj_type, comp_type = PARAM_MAP[param_key]
                
                if METRIC_NUCLEAR in stats:
                    if comp_type == 'A':
                        temp[proj_type]['nuclear_A'].append(stats[METRIC_NUCLEAR])
                    elif comp_type == 'B':
                        temp[proj_type]['nuclear_B'].append(stats[METRIC_NUCLEAR])
                
                if METRIC_COMPLEXITY in stats:
                    temp[proj_type]['complexity'].append(stats[METRIC_COMPLEXITY])
        
        for proj_type in PROJ_TYPES:
            nuclear_a = temp[proj_type]['nuclear_A']
            nuclear_b = temp[proj_type]['nuclear_B']
            complexity_vals = temp[proj_type]['complexity']
            
            if nuclear_a or nuclear_b:
                sum_nuclear = np.mean(nuclear_a) + np.mean(nuclear_b) if nuclear_a and nuclear_b else \
                             np.mean(nuclear_a) if nuclear_a else np.mean(nuclear_b)
                
                avg_complexity = np.mean(complexity_vals) if complexity_vals else 0.0
                
                aggregated[epoch][proj_type] = {
                    'nuclear_norm_sum': sum_nuclear,
                    'complexity_ratio_avg': avg_complexity,
                    'count_a': len(nuclear_a),
                    'count_b': len(nuclear_b)
                }
    
    return aggregated


def plot_summed_nuclear_by_epoch(aggregated: dict, epochs: list, save_path: str):
    
    plt.figure(figsize=FIGSIZE)
    
    has_data = False
    
    for proj in PROJ_TYPES:
        epoch_vals = []
        epoch_list = []
        
        for epoch in epochs:
            if epoch not in aggregated:
                continue
            if proj not in aggregated[epoch]:
                continue
            
            val = aggregated[epoch][proj].get('nuclear_norm_sum')
            if val is not None:
                epoch_vals.append(val)
                epoch_list.append(epoch)
                has_data = True
        
        
        plt.plot(
            epoch_list, epoch_vals,
            label="Proj_" + proj,
            color=PROJ_COLORS[proj],
            marker=PROJ_MARKERS[proj],
            linewidth=LINE_WIDTH,
            markersize=10,
            markevery=1,  
            alpha=0.9
        )
    
    if not has_data:
        plt.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('No Gradient Data Available')
    else:
        ax = plt.gca()
        plt.xlabel('Epoch', fontsize=16, fontweight='normal')
        plt.ylabel('Nuclear Norm $\| \Delta \mathbf{W}_{A} \|_{*} + \| \Delta \mathbf{W}_{B} \|_{*}$', fontsize=16, fontweight='normal')
        plt.xticks(fontsize=16, fontweight='normal')
        plt.yticks(fontsize=16, fontweight='normal')

        for spine in ax.spines.values():
            spine.set_linewidth(1.8)      
            spine.set_edgecolor('#000000') 


        plt.legend(
            loc='best', 
            fontsize=12, 
            frameon=True, 
            title_fontsize=10,
            edgecolor='gray'
        )
        
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()



def plot_summed_complexity_by_epoch(aggregated: dict, epochs: list, save_path: str):

    plt.figure(figsize=FIGSIZE)
    
    has_data = False
    
    for proj in PROJ_TYPES:
        epoch_vals = []
        epoch_list = []
        
        for epoch in epochs:
            if epoch not in aggregated:
                continue
            if proj not in aggregated[epoch]:
                continue
            
            val = aggregated[epoch][proj].get('complexity_ratio_avg')
            if val is not None and val > 0:
                epoch_vals.append(val)
                epoch_list.append(epoch)
                has_data = True
        
        
        plt.plot(
            epoch_list, epoch_vals,
            label=proj,
            color=PROJ_COLORS[proj],
            marker=PROJ_MARKERS[proj],
            linewidth=LINE_WIDTH,
            markersize=10,
            alpha=0.9
        )
    
    if has_data:
        plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Rank-1 limit')
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Complexity Ratio Avg (A+B)', fontsize=11)
        plt.title('Gradient Complexity Ratio Across Epochs', fontsize=13, fontweight='bold', pad=15)
        plt.legend(loc='best', fontsize=10, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_proj_comparison_single_epoch(aggregated: dict, epoch: int, save_path: str):

    if epoch not in aggregated:
        return
    
    plt.figure(figsize=(10, 6))
    
    x_labels = PROJ_TYPES
    x_pos = np.arange(len(x_labels))
    
    nuclear_vals = []
    for proj in PROJ_TYPES:
        val = aggregated[epoch][proj].get('nuclear_norm_sum')
        nuclear_vals.append(val if val is not None else 0.0)
    
    bars = plt.bar(
        x_pos, nuclear_vals,
        color=[PROJ_COLORS[proj] for proj in PROJ_TYPES],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    for i, (x, v) in enumerate(zip(x_pos, nuclear_vals)):
        if v > 0:
            plt.text(x, v + max(nuclear_vals)*0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(x_pos, x_labels, fontsize=11)
    plt.xlabel('Projection Type', fontsize=11, fontweight='bold')
    plt.ylabel('Nuclear Norm Sum (LoRA_A + LoRA_B)', fontsize=11, fontweight='bold')
    plt.title(f'Epoch {epoch} - Q/K/V/O Nuclear Norm Comparison (A+B)', 
              fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def print_statistics(aggregated: dict, epochs: list):
    print("\n" + "="*80)
    print("Summed Gradient Nuclear Norm Statistics (LoRA_A + LoRA_B)")
    print("="*80)
    
    for epoch in sorted(epochs):
        if epoch not in aggregated:
            continue
        
        print(f"\nEpoch {epoch}:")
        print(f"  {'Proj':<6} {'NuclearNorm(A+B)':>18} {'Complexity(Avg)':>18} {'Count(A/B)':>12}")
        print(f"  {'-'*6} {'-'*18} {'-'*18} {'-'*12}")
        
        for proj in PROJ_TYPES:
            stats = aggregated[epoch].get(proj)
            if stats:
                print(f"  {proj:<6} "
                      f"{stats['nuclear_norm_sum']:>18.4f} "
                      f"{stats['complexity_ratio_avg']:>18.2f} "
                      f"{stats['count_a']}/{stats['count_b']:>10}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":

    all_epoch_data = load_all_epochs(BASE_PATH, EPOCHS_TO_PLOT)
    if not all_epoch_data:
        exit(1)
    
    aggregated = aggregate_summed_by_proj(all_epoch_data, EPOCHS_TO_PLOT)
    
    for epoch in EPOCHS_TO_PLOT:
        if epoch in aggregated:
            print(f"  Epoch {epoch}:")
            for proj in PROJ_TYPES:
                stats = aggregated[epoch].get(proj)
                if stats:
                    print(f"    {proj}: nuclear_sum={stats['nuclear_norm_sum']:.4f}, "
                          f"complexity_avg={stats['complexity_ratio_avg']:.2f}")
    
    print_statistics(aggregated, EPOCHS_TO_PLOT)
    
    plot_summed_nuclear_by_epoch(
        aggregated,
        EPOCHS_TO_PLOT,
        os.path.join(SAVE_DIR, "{}.png".format(split))
    )
    
    plot_summed_complexity_by_epoch(
        aggregated,
        EPOCHS_TO_PLOT,
        os.path.join(SAVE_DIR, "complexity_ratio_sum_by_epoch.png")
    )
    
    for epoch in EPOCHS_TO_PLOT:
        plot_proj_comparison_single_epoch(
            aggregated,
            epoch,
            os.path.join(SAVE_DIR, f"epoch{epoch}_proj_comparison_sum.png")
        )
    
    plt.figure(figsize=(10, 6))
    heatmap_data = []
    
    for proj in PROJ_TYPES:
        proj_vals = []
        for epoch in EPOCHS_TO_PLOT:
            val = aggregated[epoch][proj].get('nuclear_norm_sum', 0)
            proj_vals.append(val)
        if any(v > 0 for v in proj_vals):
            heatmap_data.append(proj_vals)
    
    if heatmap_data:
        im = plt.imshow(
            np.array(heatmap_data).T,
            cmap='YlOrRd',
            aspect='auto',
            interpolation='nearest'
        )
        plt.colorbar(im, label='Nuclear Norm Sum (A+B)')
        plt.xticks(range(len(PROJ_TYPES)), PROJ_TYPES)
        plt.yticks(range(len(EPOCHS_TO_PLOT)), EPOCHS_TO_PLOT)
        plt.xlabel('Projection Type')
        plt.ylabel('Epoch')
        plt.title('Heatmap: Nuclear Norm Sum (LoRA_A+B) by Epoch & Projection')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "nuclear_norm_sum_heatmap.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
