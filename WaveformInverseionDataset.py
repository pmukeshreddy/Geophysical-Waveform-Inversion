from torch.utils.data import Dataset, DataLoader

class WaveformInverseionDataset(Dataset):
    def __init__(self,data_pairs,transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
    def __len__(self):
        return len(self.data_pairs)
    def __getitem__(self,idx):
        seismic_path, velocity_path = self.data_pairs[idx]
        #load data
        seismic_data = np.load(seismic_path)
        velocity_model = np.load(velocity_path)

        seismic_data = torch.from_numpy(seismic_data).float()
        velocity_data = torch.from_numpy(velocity_model).float()

        #batch_size, time_steps, channels, height, width = seismic_data.shape
        time_step_idx = seismic_data.shape[0] // 2  # Middle time step
        seismic_data = seismic_data[time_step_idx]
        velocity_data = velocity_data[time_step_idx]

        #seismic_data = seismic_data.reshape(-1, channels, height, width)

        if self.transform:
            sample = (seismic_data, velocity_data)
            sample = self.transform(sample)
            seismic_data, velocity_data = sample
        return seismic_data , velocity_data
