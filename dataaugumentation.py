import torch
import numpy as np
import torchvision.transforms as transforms
import random

# Base transforms that are safe for any dimension
class Normalize(object):
    def __init__(self, stats=None):
        self.stats = stats or {}
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if 'seismic_min' in self.stats and 'seismic_max' in self.stats:
            s_min, s_max = self.stats['seismic_min'], self.stats['seismic_max']
            seismic = 2 * (seismic - s_min) / (s_max - s_min) - 1
            
        if 'velocity_min' in self.stats and 'velocity_max' in self.stats:
            v_min, v_max = self.stats['velocity_min'], self.stats['velocity_max']
            velocity = (velocity - v_min) / (v_max - v_min)
            
        return (seismic, velocity)

class ToTensor(object):
    def __call__(self, sample):
        seismic, velocity = sample
        
        # Convert numpy arrays to tensors
        if not isinstance(seismic, torch.Tensor):
            seismic = torch.from_numpy(seismic).float()
        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float()
            
        return (seismic, velocity)

class AddNoise(object):
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        # Add noise only to seismic data
        noise = torch.randn_like(seismic) * self.noise_level * torch.std(seismic)
        seismic = seismic + noise
        
        return (seismic, velocity)

class HorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.flip_prob:
            # Get the number of dimensions
            seismic_dims = seismic.dim()
            velocity_dims = velocity.dim()
            
            # Only flip if we have at least 1 dimension
            if seismic_dims > 0:
                # Flip the last dimension
                seismic = torch.flip(seismic, dims=[-1])
            
            if velocity_dims > 0:
                # Flip the last dimension
                velocity = torch.flip(velocity, dims=[-1])
            
        return (seismic, velocity)

class VerticalFlip(object):
    def __init__(self, flip_prob=0.3):
        self.flip_prob = flip_prob
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.flip_prob:
            # Check if tensors have at least 2 dimensions before flipping vertically
            if seismic.dim() > 1:
                seismic = torch.flip(seismic, dims=[-2])
            
            if velocity.dim() > 1:
                velocity = torch.flip(velocity, dims=[-2])
            
        return (seismic, velocity)

class RandomJitter(object):
    def __init__(self, jitter_range=0.05, prob=0.4):
        self.jitter_range = jitter_range
        self.prob = prob
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.prob:
            # Add small random jitter to seismic data only
            jitter = torch.rand_like(seismic) * 2 * self.jitter_range - self.jitter_range
            seismic = seismic + jitter
            
        return (seismic, velocity)

class MaskRandom(object):
    def __init__(self, mask_prob=0.2, mask_ratio=0.1):
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.mask_prob:
            # Create mask tensor with same shape as seismic
            mask = torch.rand_like(seismic) < self.mask_ratio
            
            # Apply mask by setting values to the mean
            mean_val = seismic.mean()
            seismic = seismic.masked_fill(mask, mean_val)
            
        return (seismic, velocity)

class TimeReverse(object):
    def __init__(self, reverse_prob=0.2):
        self.reverse_prob = reverse_prob
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.reverse_prob:
            # Reverse the last dimension (time or spatial)
            seismic = torch.flip(seismic, dims=[-1])
            
        return (seismic, velocity)

class SmallAmplitudeScaling(object):
    def __init__(self, scale_range=(0.9, 1.1), prob=0.3):
        self.scale_range = scale_range
        self.prob = prob
        
    def __call__(self, sample):
        seismic, velocity = sample
        
        if torch.rand(1).item() < self.prob:
            # Randomly scale amplitude of seismic data
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            seismic = seismic * scale
            
        return (seismic, velocity)

# Updated transform pipelines with safe, robust options
transform_train = transforms.Compose([
    ToTensor(),
    Normalize(stats={'seismic_min': -0.5, 'seismic_max': 0.5, 
                    'velocity_min': 1500, 'velocity_max': 5000}),
    HorizontalFlip(flip_prob=0.5),
    # Only include VerticalFlip if your data is 2D
    # VerticalFlip(flip_prob=0.3),
    RandomJitter(jitter_range=0.03, prob=0.4),
    MaskRandom(mask_prob=0.3, mask_ratio=0.05),
    TimeReverse(reverse_prob=0.2),
    SmallAmplitudeScaling(scale_range=(0.9, 1.1), prob=0.3),
    AddNoise(noise_level=0.03)
])

transform_val = transforms.Compose([
    ToTensor(),
    Normalize(stats={'seismic_min': -0.5, 'seismic_max': 0.5, 
                    'velocity_min': 1500, 'velocity_max': 5000})
])
