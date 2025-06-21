import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

    
            
def train_fno_unet_model(train_loader, val_loader, test_loader):
    # Set device and ensure it's GPU
    device = torch.device('cuda')  # Directly use CUDA without fallback
    print(f"Using device: {device}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure GPU is properly configured.")
    
    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize model on GPU
    model = FOUent(in_channels=5, out_channels=1, features=32, modes=5).to(device)
    
    # Physics-informed loss on GPU
    criterion = PhysicsInformedLoss(lambda_physics=0.01, dx=10.0, dt=0.002).to(device)
    
    # Optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'fno_unet_{timestamp}.pth')
    
    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    early_stopping_counter = 0  # Counter for epochs without improvement
    early_stopping_min_delta = 1e-4  # Minimum change to qualify as improvement
    
    # Lists to store metrics
    train_losses, train_mse_losses, train_physics_losses = [], [], []
    val_losses, val_mse_losses, val_physics_losses = [], [], []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_physics = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for seismic_data, velocity_model in pbar:
            # Move data to GPU immediately
            seismic_data = seismic_data.to(device, non_blocking=True)
            velocity_model = velocity_model.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
            
            # Forward pass
            predicted_velocity = model(seismic_data)
            
            # Compute loss
            loss, mse, physics_loss = criterion(predicted_velocity, velocity_model, seismic_data)
            
            # Backward pass and optimize
            loss.backward()
            
            # Optional: gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics - detach from computation graph
            running_loss += loss.detach().item()
            running_mse += mse.detach().item()
            running_physics += physics_loss.detach().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.detach().item(),
                'mse': mse.detach().item(),
                'physics': physics_loss.detach().item(),
                'gpu_mem': f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            })
            
            # Optional: Clear GPU cache periodically
            if torch.cuda.is_available() and (pbar.n % 50 == 0):
                torch.cuda.empty_cache()
        
        # Calculate average train losses
        train_loss = running_loss / len(train_loader)
        train_mse_loss = running_mse / len(train_loader)
        train_physics_loss = running_physics / len(train_loader)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        running_val_mse = 0.0
        running_val_physics = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for seismic_data, velocity_model in val_pbar:
                # Move data to GPU
                seismic_data = seismic_data.to(device, non_blocking=True)
                velocity_model = velocity_model.to(device, non_blocking=True)
                
                # Forward pass
                predicted_velocity = model(seismic_data)
                
                # Compute loss
                loss, mse, physics_loss = criterion(predicted_velocity, velocity_model, seismic_data)
                
                # Update statistics
                running_val_loss += loss.item()
                running_val_mse += mse.item()
                running_val_physics += physics_loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'mse': mse.item(),
                    'physics': physics_loss.item()
                })
        
        # Calculate average validation losses
        val_loss = running_val_loss / len(val_loader)
        val_mse_loss = running_val_mse / len(val_loader)
        val_physics_loss = running_val_physics / len(val_loader)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save losses for plotting
        train_losses.append(train_loss)
        train_mse_losses.append(train_mse_loss)
        train_physics_losses.append(train_physics_loss)
        val_losses.append(val_loss)
        val_mse_losses.append(val_mse_loss)
        val_physics_losses.append(val_physics_loss)
        
        # Print progress with GPU memory usage
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4e}, MSE: {train_mse_loss:.4e}, Physics: {train_physics_loss:.4e}")
        print(f"  Val Loss: {val_loss:.4e}, MSE: {val_mse_loss:.4e}, Physics: {val_physics_loss:.4e}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter when validation loss improves
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, save_path)
            print(f"  Saved new best model with validation loss: {val_loss:.4e}")
        else:
            early_stopping_counter += 1
            print(f"  EarlyStopping: {early_stopping_counter}/{patience} [best val_loss: {best_val_loss:.4e}]")
            
            # Check if early stopping criteria is met
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4e}")
                break
        
        # Clear GPU cache at the end of epoch
        torch.cuda.empty_cache()
        
        # Save loss curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss')
            plt.legend()
            plt.title('Training Progress')
            
            plt.subplot(1, 3, 2)
            plt.plot(train_mse_losses, label='Train MSE')
            plt.plot(val_mse_losses, label='Val MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.title('MSE Component')
            
            plt.subplot(1, 3, 3)
            plt.plot(train_physics_losses, label='Train Physics')
            plt.plot(val_physics_losses, label='Val Physics')
            plt.xlabel('Epoch')
            plt.ylabel('Physics Loss')
            plt.legend()
            plt.title('Physics Component')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'loss_curves_{timestamp}_epoch{epoch+1}.png'))
            plt.close()
    
    # Final evaluation on test set
    model.eval()
    running_test_loss = 0.0
    running_test_mse = 0.0
    running_test_physics = 0.0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Final Test Evaluation")
        for seismic_data, velocity_model in test_pbar:
            # Move data to GPU
            seismic_data = seismic_data.to(device, non_blocking=True)
            velocity_model = velocity_model.to(device, non_blocking=True)
            
            # Forward pass
            predicted_velocity = model(seismic_data)
            
            # Compute loss
            loss, mse, physics_loss = criterion(predicted_velocity, velocity_model, seismic_data)
            
            # Update statistics
            running_test_loss += loss.item()
            running_test_mse += mse.item()
            running_test_physics += physics_loss.item()
            
            # Update progress bar
            test_pbar.set_postfix({
                'loss': loss.item(),
                'mse': mse.item(),
                'physics': physics_loss.item()
            })
    
    # Calculate average test losses
    test_loss = running_test_loss / len(test_loader)
    test_mse = running_test_mse / len(test_loader)
    test_physics = running_test_physics / len(test_loader)
    
    print("\nTest Results:")
    print(f"  Test Loss: {test_loss:.4e}, MSE: {test_mse:.4e}, Physics: {test_physics:.4e}")
    
    # Visualization of final predictions
    visualize_results(model, test_loader, device, save_dir, timestamp)
    
    # Return model and metrics
    return model, {
        'train_loss': train_losses,
        'train_mse': train_mse_losses,
        'train_physics': train_physics_losses,
        'val_loss': val_losses,
        'val_mse': val_mse_losses,
        'val_physics': val_physics_losses,
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_physics': test_physics
    }
            
            

def visualize_results(model, test_loader, device, save_dir, timestamp):
    """Visualize some predictions from the test set"""
    model.eval()
    
    # Get batch of test data
    seismic_data, true_velocity = next(iter(test_loader))
    seismic_data = seismic_data.to(device, non_blocking=True)
    true_velocity = true_velocity.to(device, non_blocking=True)
    
    # Make predictions
    with torch.no_grad():
        predicted_velocity = model(seismic_data)
    
    # Select a few samples to visualize
    num_samples = min(4, seismic_data.shape[0])
    
    # Move data back to CPU for visualization
    seismic_data = seismic_data.cpu()
    true_velocity = true_velocity.cpu()
    predicted_velocity = predicted_velocity.cpu()
    
    plt.figure(figsize=(15, 4*num_samples))
    
    for i in range(num_samples):
        # Get data
        sample_seismic = seismic_data[i].numpy()
        sample_true = true_velocity[i].numpy()
        sample_pred = predicted_velocity[i].numpy()
        
        # Plot
        plt.subplot(num_samples, 3, i*3 + 1)
        if sample_seismic.ndim > 2:  # Multiple channels
            plt.imshow(sample_seismic[0], cmap='seismic')
            plt.title(f"Sample {i+1}: Seismic Data (Channel 0)")
        else:
            plt.imshow(sample_seismic, cmap='seismic')
            plt.title(f"Sample {i+1}: Seismic Data")
        plt.colorbar()
        
        plt.subplot(num_samples, 3, i*3 + 2)
        if sample_true.ndim > 2:  # Multiple channels
            plt.imshow(sample_true[0], cmap='viridis')
            plt.title(f"Sample {i+1}: True Velocity")
        else:
            plt.imshow(sample_true, cmap='viridis')
            plt.title(f"Sample {i+1}: True Velocity")
        plt.colorbar()
        
        plt.subplot(num_samples, 3, i*3 + 3)
        if sample_pred.ndim > 2:  # Multiple channels
            plt.imshow(sample_pred[0], cmap='viridis')
            plt.title(f"Sample {i+1}: Predicted Velocity")
        else:
            plt.imshow(sample_pred, cmap='viridis')
            plt.title(f"Sample {i+1}: Predicted Velocity")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_{timestamp}.png'))
    plt.close()

# Add GPU memory monitoring function
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"Max allocated: {max_allocated:.2f}GB / {total:.2f}GB total")

# Optimized DataLoader setup recommendation
def setup_optimized_data_loaders(dataset_train, dataset_val, dataset_test, batch_size=32):
    # Pin memory for faster GPU transfer
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,  # Adjust based on CPU cores
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader

