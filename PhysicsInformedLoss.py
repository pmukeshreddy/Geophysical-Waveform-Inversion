import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_physics=0.0, dx=10.0, dt=0.002, normalize_data=True, smooth_data=False, 
                 smooth_sigma=0.5, enable_physics_after=20):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.dx = dx
        self.dt = dt
        self.mse_loss = nn.MSELoss()
        self.epoch_counter = 0
        self.normalize_data = normalize_data
        self.smooth_data = smooth_data
        self.smooth_sigma = smooth_sigma
        self.enable_physics_after = enable_physics_after
        
    def spatial_laplacian(self, f):
        """Computes spatial Laplacian, preserving batch and time dimensions."""
        # Input shape: [B, T, Nx, Ny] or [B, 1, Nx, Ny]
        # Output shape: [B, T, Nx-2, Ny-2] or [B, 1, Nx-2, Ny-2]
        f_xx = (f[:, :, 2:, 1:-1] - 2 * f[:, :, 1:-1, 1:-1] + f[:, :, :-2, 1:-1]) / (self.dx**2)
        f_yy = (f[:, :, 1:-1, 2:] - 2 * f[:, :, 1:-1, 1:-1] + f[:, :, 1:-1, :-2]) / (self.dx**2)
        return f_xx + f_yy
        
    def time_second_derivative(self, wavefield):
        """Computes second time derivative, operating on the time dimension.
        Input shape: [B, T, Nx, Ny]
        Output shape: [B, T-2, Nx, Ny]
        """
        # Ensure there are enough time steps
        if wavefield.shape[1] < 3:
            print("WARNING: Not enough time steps to compute second derivative")
            # Return placeholder zeros matching expected output shape
            return torch.zeros((wavefield.shape[0], 1, wavefield.shape[2], wavefield.shape[3]), 
                              device=wavefield.device)
        
        # Apply finite difference along time dimension (dim=1)
        u_tt = (wavefield[:, 2:, :, :] - 
                2 * wavefield[:, 1:-1, :, :] + 
                wavefield[:, :-2, :, :]) / (self.dt**2)
        return u_tt
    
    def smooth_wavefield(self, wavefield):
        """Apply Gaussian smoothing to the wavefield to reduce noise.
        Uses a separable Gaussian filter for efficiency.
        """
        # Skip smoothing if disabled
        if not self.smooth_data:
            return wavefield
            
        # Extract dimensions for reshaping
        B, T, H, W = wavefield.shape
        
        # Reshape to [B*T, 1, H, W] for conv2d
        wavefield_flat = wavefield.reshape(B*T, 1, H, W)
        
        # Create 1D Gaussian kernels for separable filtering
        kernel_size = max(3, int(2 * self.smooth_sigma) * 2 + 1)
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create 1D Gaussian kernel
        channels = 1
        sigma = self.smooth_sigma
        
        # Create x-direction kernel (horizontal)
        kernel_x = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / (2*sigma**2))
        kernel_x = kernel_x / kernel_x.sum()
        kernel_x = kernel_x.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        
        # Create y-direction kernel (vertical)
        kernel_y = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / (2*sigma**2))
        kernel_y = kernel_y / kernel_y.sum()
        kernel_y = kernel_y.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
        
        # Move kernels to same device as input
        kernel_x = kernel_x.to(wavefield.device)
        kernel_y = kernel_y.to(wavefield.device)
        
        # Apply separable convolution for efficiency
        # First horizontal, then vertical
        padding_x = (0, kernel_size//2, 0, kernel_size//2)
        padding_y = (kernel_size//2, 0, kernel_size//2, 0)
        
        # Pad before convolution
        wavefield_padded_x = F.pad(wavefield_flat, padding_x, mode='reflect')
        wavefield_smooth_x = F.conv2d(wavefield_padded_x, kernel_x, groups=channels)
        
        # Pad again for the vertical direction
        wavefield_padded_y = F.pad(wavefield_smooth_x, padding_y, mode='reflect')
        wavefield_smooth = F.conv2d(wavefield_padded_y, kernel_y, groups=channels)
        
        # Reshape back to original shape
        return wavefield_smooth.reshape(B, T, H, W)
        
    def normalize_wavefield(self, wavefield, per_batch=True):
        """Normalize wavefield to reduce the magnitude of derivatives.
        Can normalize globally or per batch.
        
        Compatible with older PyTorch versions that don't support tuple dims.
        """
        if not self.normalize_data:
            return wavefield
            
        # Get absolute values
        abs_wavefield = torch.abs(wavefield)
            
        if per_batch:
            # Normalize each batch separately - using sequential max operations
            # Compatible with older PyTorch versions
            B, T, H, W = wavefield.shape
            max_vals = []
            
            for b in range(B):
                # Flatten across all dimensions except batch
                batch_flat = abs_wavefield[b].reshape(-1)
                batch_max = torch.max(batch_flat)
                max_vals.append(batch_max)
                
            # Stack the max values and reshape for broadcasting
            max_vals = torch.stack(max_vals).view(B, 1, 1, 1)
            # Add small epsilon to avoid division by zero
            max_vals = torch.clamp(max_vals, min=1e-10)
            normalized = wavefield / max_vals
        else:
            # Global normalization
            max_val = torch.max(abs_wavefield)
            max_val = torch.clamp(max_val, min=1e-10)
            normalized = wavefield / max_val
            
        return normalized
        
    def wave_equation(self, predicted_velocity, seismic_data):
        """Computes the physics loss based on the wave equation.
        predicted_velocity: [B, 1, Nx_v, Ny_v] - Velocity model
        seismic_data: [B, R, T, Nx_s] or [B, R, T, Nx_s, Ny_s] - Seismic wavefield
        """
        # Step 1: Process seismic data into proper wavefield form [B, T, Nx, Ny]
        if seismic_data.dim() == 4:  # [B, R, T, Nx]
            # Average over receivers and add Ny dimension
            wavefield = seismic_data.mean(dim=1)  # [B, T, Nx]
            wavefield = wavefield.unsqueeze(-1)   # [B, T, Nx, 1]
        elif seismic_data.dim() == 5:  # [B, R, T, Nx, Ny]
            # Average over receivers
            wavefield = seismic_data.mean(dim=1)  # [B, T, Nx, Ny]
        
        # Step 2: Normalize and smooth the wavefield
        wavefield = self.normalize_wavefield(wavefield)
        wavefield = self.smooth_wavefield(wavefield)
        
        # Step 3: Spatially interpolate wavefield to match velocity grid size
        # Preserve the time dimension by handling each time step separately
        B, T = wavefield.shape[0], wavefield.shape[1]
        Nx_v, Ny_v = predicted_velocity.shape[2], predicted_velocity.shape[3]
        
        # Reshape to treat each time step as a separate "batch" for interpolation
        wavefield_flat = wavefield.reshape(B * T, 1, *wavefield.shape[2:])
        
        # Interpolate to velocity grid dimensions
        wavefield_interp = torch.nn.functional.interpolate(
            wavefield_flat,
            size=(Nx_v, Ny_v),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back to [B, T, Nx_v, Ny_v]
        wavefield = wavefield_interp.reshape(B, T, Nx_v, Ny_v)
        
        # Step 4: Compute second time derivative (along time dimension)
        u_tt = self.time_second_derivative(wavefield)  # [B, T-2, Nx_v, Ny_v]
        
        # Step 5: Compute spatial Laplacian (operated on wavefield)
        # For correct alignment, compute Laplacian of appropriate time slices
        if wavefield.shape[1] >= 3:
            # Match the time steps used in u_tt (from t=0 to t=T-2)
            laplacian_u = self.spatial_laplacian(wavefield[:, :-2, :, :])  # [B, T-2, Nx_v-2, Ny_v-2]
        else:
            # Fallback if not enough time steps
            laplacian_u = self.spatial_laplacian(wavefield)  # [B, T, Nx_v-2, Ny_v-2]
        
        # Step 6: Process velocity model (extract interior to match Laplacian output)
        vel_interior = predicted_velocity[:, :, 1:-1, 1:-1]  # [B, 1, Nx_v-2, Ny_v-2]
        
        # Ensure velocity values are reasonable (prevent division by very small values)
        # Typical seismic velocities are ~1500-6000 m/s
        # Lower bound: 100 m/s, Upper bound: 10000 m/s
        vel_interior = torch.clamp(vel_interior, min=100.0, max=10000.0)
        vel_terms = 1.0 / (vel_interior**2)  # Inverse squared velocity 
        
        # Step 7: Align dimensions for the wave equation residual
        # Crop u_tt to match laplacian_u spatial dimensions
        u_tt_cropped = u_tt[:, :, 1:-1, 1:-1]  # [B, T-2, Nx_v-2, Ny_v-2]
        
        # Final check and adjustment to ensure tensor shapes match
        if laplacian_u.shape != u_tt_cropped.shape:
            # Determine the minimum common dimensions
            min_t = min(laplacian_u.shape[1], u_tt_cropped.shape[1])
            min_h = min(laplacian_u.shape[2], u_tt_cropped.shape[2])
            min_w = min(laplacian_u.shape[3], u_tt_cropped.shape[3])
            
            # Crop to common dimensions
            laplacian_u = laplacian_u[:, :min_t, :min_h, :min_w]
            u_tt_cropped = u_tt_cropped[:, :min_t, :min_h, :min_w]
            
            # Also crop vel_terms to match spatial dimensions
            if vel_terms.shape[2] != min_h or vel_terms.shape[3] != min_w:
                vel_terms = vel_terms[:, :, :min_h, :min_w]
        
        # Broadcast vel_terms to match time dimension if needed
        if vel_terms.shape[1] == 1 and laplacian_u.shape[1] > 1:
            vel_terms = vel_terms.expand(-1, laplacian_u.shape[1], -1, -1)
        
        # Step 8: Compute wave equation residual: ∇²u - (1/v²)∂²u/∂t²
        wave_eq_residuals = laplacian_u - vel_terms * u_tt_cropped
        
        # Compute mean squared residual
        physics_loss = torch.mean(wave_eq_residuals**2)
        
        # Clamp to avoid extreme values (more conservative clamping)
        physics_loss = torch.clamp(physics_loss, max=1e4)
            
        return physics_loss
    
    def forward(self, predicted_velocity, target_velocity, seismic_data):
        """Forward pass computing both MSE and physics-informed loss.
        predicted_velocity: [B, 1, Nx_v, Ny_v] - Predicted velocity model
        target_velocity: [B, 1, Nx_t, Ny_t] - Ground truth velocity model
        seismic_data: [B, R, T, Nx_s] or [B, R, T, Nx_s, Ny_s] - Seismic data
        """
        # Resize predicted velocity to match target if needed
        if predicted_velocity.shape != target_velocity.shape:
            predicted_velocity = torch.nn.functional.interpolate(
                predicted_velocity, 
                size=(target_velocity.shape[2], target_velocity.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        predicted_velocity_physical = predicted_velocity * (5000 - 1500) + 1500

        
        # Calculate MSE loss (data-fitting term)
        mse = self.mse_loss(predicted_velocity, target_velocity)
        
        # Calculate physics loss with error handling
        physics_loss_weighted = torch.tensor(0.0, device=predicted_velocity.device)
        if self.lambda_physics > 0:
            try:
                physics_loss = self.wave_equation(predicted_velocity_physical, seismic_data)
                physics_loss_weighted = self.lambda_physics * physics_loss
                
                # Log diagnostics about the loss values
                with torch.no_grad():
                    print(f"Raw physics loss: {physics_loss.item():.2e}, Lambda: {self.lambda_physics:.2e}, "
                          f"Weighted physics loss: {physics_loss_weighted.item():.2e}, MSE loss: {mse.item():.2e}")
            except Exception as e:
                print(f"Error in physics loss calculation: {str(e)}")
                # Avoid using traceback which might not be imported
                print("Please check the loss function parameters and tensor shapes.")
                physics_loss_weighted = torch.tensor(0.0, device=predicted_velocity.device)
        
        # Calculate total loss
        total_loss = mse + physics_loss_weighted
        
        return total_loss, mse, physics_loss_weighted
    
    def update_epoch(self, mse_value=None):
        """Call this at the end of each epoch to update internal counters
        
        Args:
            mse_value: Optional current MSE loss value to help scale physics loss
        """
        self.epoch_counter += 1
        
        # Implement improved progressive physics loss weighting
        # Only start physics loss after specified number of epochs
        if self.epoch_counter == self.enable_physics_after and self.lambda_physics == 0:
            # Set initial lambda based on MSE value if provided
            if mse_value is not None:
                # Start with physics contribution at ~1% of MSE
                # Assuming normalized physics loss ~1000
                self.lambda_physics = max(1e-10, mse_value * 0.01 / 1000)
            else:
                # Default initial value if MSE unknown
                self.lambda_physics = 1e-8
                
            print(f"Enabling physics loss with initial lambda: {self.lambda_physics:.2e}")
            
        # Continue gradual increase after enabled
        elif self.epoch_counter > self.enable_physics_after and self.lambda_physics > 0:
            # Gentler increase - multiply by 1.5 every 5 epochs up to a maximum
            if self.epoch_counter % 5 == 0 and self.lambda_physics < 1e-3:
                old_lambda = self.lambda_physics
                self.lambda_physics = min(self.lambda_physics * 1.5, 1e-3)
                print(f"Updated physics loss weight: {old_lambda:.2e} -> {self.lambda_physics:.2e}")
