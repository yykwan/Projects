import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import spatialdata as sd
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from spatialdata.transformations import Affine
import einops
import timm
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, zarr_path, transform=None, context_radius=128):
        self.data = sd.SpatialData.from_zarr(zarr_path)
        self.transform = transform
        self.context_radius = context_radius
        
        # Filter cells with valid spatial coordinates
        valid_cells = self.data.tables['anucleus'].obs[
            ~np.isnan(self.data.tables['anucleus'].obsm['spatial'][:, 0])]
        self.cell_ids = valid_cells.index
        
        # Precompute spatial kd-tree for neighborhood queries
        self.coords = self.data.tables['anucleus'].obsm['spatial'][self.cell_ids]
        self.kdtree = KDTree(self.coords)
        
    def __len__(self):
        return len(self.cell_ids)
    
    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        center = self.data.tables['anucleus'].obsm['spatial'][cell_id]
        
        # Extract multi-scale H&E patches
        patches = []
        for scale in [64, 128, 256]:
            patch = extract_patch(self.data.images['HE_registered'], 
                                center, 
                                scale,
                                self.data.transformations['HE_registered']['global'])
            patches.append(cv2.resize(patch, (128, 128)))
        
        # Get spatial context (neighboring cells)
        neighbors = self.kdtree.query_radius([center], r=self.context_radius)[0]
        neighbor_expr = self.data.tables['anucleus'].X[neighbors]
        neighbor_coords = self.coords[neighbors] - center
        
        # Target gene expression
        target_expr = self.data.tables['anucleus'].X[cell_id]
        
        sample = {
            'patches': torch.stack([torch.tensor(p).permute(2,0,1).float() for p in patches]),
            'neighbor_expr': torch.tensor(neighbor_expr).float(),
            'neighbor_coords': torch.tensor(neighbor_coords).float(),
            'target': torch.tensor(target_expr).float(),
            'cell_id': cell_id
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class HierarchicalSpatialPredictor(nn.Module):
    def __init__(self, num_genes=460, num_neighbors=32):
        super().__init__()
        
        # Multi-scale image encoder (using pretrained ViT)
        self.image_encoder = timm.create_model('vit_base_patch16_224', 
                                             pretrained=True,
                                             num_classes=0)
        self.image_proj = nn.Linear(768, 256)
        
        # Spatial attention for neighbors
        self.pos_enc = PositionalEncodingPermute2D(2)
        self.neighbor_attn = nn.TransformerEncoderLayer(
            d_model=num_genes+2, nhead=8, dim_feedforward=512)
        
        # Fusion and prediction
        self.fusion = nn.TransformerEncoderLayer(
            d_model=256+num_genes+2, nhead=8, dim_feedforward=1024)
        self.predictor = nn.Sequential(
            nn.Linear(256+num_genes+2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, num_genes)
        )
        
    def forward(self, x):
        # Process multi-scale image patches
        img_feats = []
        for patch in einops.rearrange(x['patches'], 'b s c h w -> (b s) c h w'):
            feats = self.image_encoder(patch.unsqueeze(0))
            img_feats.append(self.image_proj(feats))
        img_feat = torch.mean(torch.stack(img_feats), dim=0)
        
        # Process neighbors with spatial attention
        neighbor_feats = torch.cat([x['neighbor_expr'], x['neighbor_coords']], dim=-1)
        pos_enc = self.pos_enc(neighbor_feats.unsqueeze(0)).squeeze(0)
        neighbor_feats = self.neighbor_attn(neighbor_feats + pos_enc)
        neighbor_feat = torch.mean(neighbor_feats, dim=0)
        
        # Fuse features and predict
        combined = torch.cat([img_feat, neighbor_feat], dim=-1)
        combined = self.fusion(combined.unsqueeze(0)).squeeze(0)
        return self.predictor(combined)

def train_with_cv(zarr_paths, num_folds=5, epochs=20):
    all_data = [SpatialTranscriptomicsDataset(p) for p in zarr_paths]
    kf = KFold(n_splits=num_folds, shuffle=True)
    
    best_models = []
    for train_idx, val_idx in kf.split(all_data):
        train_data = torch.utils.data.ConcatDataset([all_data[i] for i in train_idx])
        val_data = torch.utils.data.ConcatDataset([all_data[i] for i in val_idx])
        
        model = HierarchicalSpatialPredictor().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Training loop
            model.train()
            for batch in DataLoader(train_data, batch_size=16, shuffle=True):
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch['target'].cuda())
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in DataLoader(val_data, batch_size=32):
                    outputs = model(batch)
                    val_loss += criterion(outputs, batch['target'].cuda()).item()
            
            val_loss /= len(val_data)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_fold_{len(best_models)}.pt')
            
            scheduler.step()
        
        best_models.append(model.load_state_dict(torch.load(f'best_fold_{len(best_models)}.pt')))
    
    return best_models