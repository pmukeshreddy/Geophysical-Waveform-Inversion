# slice from all files and shuffles them.
class WaveformInversionDataset(Dataset):
    def __init__(self, data_pairs, num_time_steps=5, mode='train',transform=None):
        super().__init__()
        self.data_pairs = data_pairs
        self.num_time_steps = num_time_steps
        self.transform = transform
        self.slice_map = []

        print("Building a map of all possible data slices...")
        for file_idx, (seismic_path, _) in enumerate(tqdm(self.data_pairs)):
            # We only need to check the length of the seismic data
            # np.load is fast enough here, but for huge datasets, you might store lengths separately
            num_frames = np.load(seismic_path).shape[0]
            
            if num_frames >= self.num_time_steps:
                # For each possible starting point in the file, add an entry to our map
                for slice_start_idx in range(num_frames - self.num_time_steps + 1):
                    self.slice_map.append((file_idx, slice_start_idx))
        print(f"Dataset created with {len(self.slice_map)} total slices.")

    def __len__(self):
        # The length of the dataset is now the total number of slices
        return len(self.slice_map)

    def __getitem__(self, idx):
        # 1. Get the file and the specific slice start point from our map
        file_idx, slice_start_idx = self.slice_map[idx]
        seismic_path, velocity_path = self.data_pairs[file_idx]
        
        # 2. Load the data
        seismic_data = np.load(seismic_path)
        velocity_model = np.load(velocity_path)

        # 3. Extract the specific slice
        seismic_data = seismic_data[slice_start_idx : slice_start_idx + self.num_time_steps]

        # --- The rest of the preprocessing is the same ---
        if velocity_model.ndim > 2:
            velocity_model = velocity_model[velocity_model.shape[0] // 2]
        
        if seismic_data.ndim == 4:
            seismic_data = np.mean(seismic_data, axis=1)

        seismic_tensor = torch.from_numpy(seismic_data.copy()).float()
        velocity_tensor = torch.from_numpy(velocity_model.copy()).float()

        if seismic_tensor.shape[-2:] != (70, 70):
             seismic_tensor = F.interpolate(seismic_tensor.unsqueeze(0), size=(70, 70), mode='bilinear', align_corners=False).squeeze(0)
        
        if velocity_tensor.ndim == 2:
            velocity_tensor = velocity_tensor.unsqueeze(0)
        
        if velocity_tensor.shape[-2:] != (70, 70):
             velocity_tensor = F.interpolate(velocity_tensor.unsqueeze(0), size=(70, 70), mode='bilinear', align_corners=False).squeeze(0)

        if self.transform:
            seismic_tensor, velocity_tensor = self.transform((seismic_tensor, velocity_tensor))

        return seismic_tensor, velocity_tensor
