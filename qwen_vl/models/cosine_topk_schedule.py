import numpy as np
import math

class LayerScheduler:
    def __init__(self, start=64, total_layers=24, min_value=1):
        self.start = start
        self.end = min_value
        self.total = total_layers
        
    def linear(self):
        return np.maximum(np.round(np.linspace(self.start, self.end, self.total)), self.end).astype(int)
        
    def cosine(self):
        idxs = np.linspace(0, 1, self.total)
        values = (self.start - self.end) / 2 * np.cos(idxs * np.pi) + (self.start + self.end) / 2
        return np.maximum(np.round(values), self.end).astype(int)
        
    def geometric(self):
        base = (self.end / self.start) ** (1 / max(1, self.total - 1))
        indices = np.arange(self.total)
        values = self.start * (base ** indices)
        return np.maximum(np.round(values), self.end).astype(int).tolist()

scheduler = LayerScheduler(start=64, total_layers=36, min_value=16) 