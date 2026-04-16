import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import pickle

layer_norm = []
epoch = 0
with open("./full_grad_data.pkl", 'rb') as f:
    data = pickle.load(f)
    for key, item in data[epoch].items():
        layer_norm.append(item.tolist())

cut = 300
FIG_WIDTH = 12     
FIG_HEIGHT = 6      
DPI = 300           

max_pixels = 2**23  
assert FIG_WIDTH * DPI < max_pixels
assert FIG_HEIGHT * DPI < max_pixels

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

layer_indices = [2, 5, 10, 15, 25]
colors = ['#C44E52', '#55A868', '#9292EF', '#6239E6', '#9467bd']


for layer_idx in layer_indices:
    data = layer_norm[26-layer_idx][:cut]
    print(f"  Layer {layer_idx}: len={len(data)}, range=[{min(data):.4f}, {max(data):.4f}]")


min_len = None
plot_data = {}  

for idx, layer_idx in enumerate(layer_indices):
    data = layer_norm[26-layer_idx][:cut]
    
    if idx == 0:
        min_len = len(data)
    else:
        min_len = min(min_len, len(data))
    
    plot_data[layer_idx] = [item * 100 for item in data[:min_len]]  # 缓存
    
    plt.plot(
        range(min_len), plot_data[layer_idx],
        label=f'Layer {layer_idx}',
        color=colors[idx % len(colors)],
        linewidth=3, alpha=0.9
    )

ax = plt.gca()
plt.xlabel('Token Index', fontsize=16)
plt.ylabel(r'Token-wise Gradient Frobenius Norm $\|\mathbf{v}_{i}\|_{F}$', fontsize=16)
plt.xticks(fontsize=16, fontweight='normal')
plt.yticks(fontsize=16, fontweight='normal')
plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='gray')
plt.grid(True, linestyle='--', alpha=0.4)
plt.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')


for spine in ax.spines.values():
    spine.set_linewidth(1.8)        
    spine.set_edgecolor('#000000') 


if min_len > 100:
    axins = zoomed_inset_axes(plt.gca(), zoom=3.5, loc='upper center', borderpad=2)
    
    for layer_idx in layer_indices:
        axins.plot(range(min_len), plot_data[layer_idx],
                  color=colors[layer_indices.index(layer_idx) % len(colors)],
                  linewidth=3, alpha=0.9)
    

    sample_data = plot_data[layer_indices[0]]
    x1, x2 = 190, 230  
    y1, y2 = min(sample_data[x1:x2]) * 0.35, max(sample_data[x1:x2]) * 0.65
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.tick_params(labelsize=10)
    axins.grid(True, linestyle='--', alpha=0.3)
    
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, 
              fc="none", ec="#d62728", linewidth=1, ls='--')

save_path = "./token_gradient.png"
plt.savefig(save_path, dpi=DPI, bbox_inches='tight', format='png')
plt.close()
