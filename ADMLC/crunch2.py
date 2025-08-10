import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class GeneImputationModel(nn.Module):
    def __init__(self, input_dim=460, hidden_dim=1024, output_dim=2000):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Projection heads
        self.proj_xenium = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.proj_scrnaseq = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim*2, output_dim)
        )
        
    def forward(self, x, mode='xenium'):
        h = self.encoder(x)
        if mode == 'xenium':
            return self.proj_xenium(h), self.predictor(h)
        else:
            return self.proj_scrnaseq(h), self.predictor(h)

class ContrastiveGeneDataset(Dataset):
    def __init__(self, sc_data, xenium_data, n_neighbors=50):
        self.sc_data = sc_data
        self.xenium_data = xenium_data
        
        # Normalize data
        sc.pp.normalize_total(self.sc_data, target_sum=1e4)
        sc.pp.log1p(self.sc_data)
        
        # Find nearest neighbors between modalities
        self.knn = NearestNeighbors(n_neighbors=n_neighbors).fit(self.sc_data.X)
        
    def __len__(self):
        return len(self.xenium_data)
    
    def __getitem__(self, idx):
        xenium_expr = self.xenium_data[idx]
        
        # Find matching scRNA-seq profiles
        _, neighbor_idx = self.knn.kneighbors([xenium_expr])
        sc_expr = self.sc_data.X[neighbor_idx.squeeze()]
        
        # Randomly sample positive and negative pairs
        pos_idx = np.random.choice(neighbor_idx.squeeze())
        neg_idx = np.random.choice(len(self.sc_data))
        
        return {
            'xenium': torch.tensor(xenium_expr).float(),
            'sc_pos': torch.tensor(self.sc_data.X[pos_idx]).float(),
            'sc_neg': torch.tensor(self.sc_data.X[neg_idx]).float()
        }

def train_gene_imputation(sc_path, xenium_paths, epochs=50):
    # Load and preprocess data
    sc_data = sc.read_h5ad(sc_path)
    xenium_data = [sd.SpatialData.from_zarr(p).tables['anucleus'].X for p in xenium_paths]
    xenium_data = np.vstack(xenium_data)
    
    # Filter genes
    common_genes = [...] # Get intersection of 460 Xenium genes and scRNA-seq genes
    sc_data = sc_data[:, common_genes]
    
    # Create dataset and model
    dataset = ContrastiveGeneDataset(sc_data, xenium_data)
    model = GeneImputationModel().cuda()
    
    # Loss functions
    contrastive_loss = nn.TripletMarginLoss(margin=1.0)
    prediction_loss = nn.HuberLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs//5)
    
    # Training loop
    for epoch in range(epochs):
        for batch in DataLoader(dataset, batch_size=128, shuffle=True):
            optimizer.zero_grad()
            
            # Contrastive learning
            xenium_h, _ = model(batch['xenium'].cuda(), 'xenium')
            sc_pos_h, _ = model(batch['sc_pos'].cuda(), 'scrnaseq')
            sc_neg_h, _ = model(batch['sc_neg'].cuda(), 'scrnaseq')
            loss_contrast = contrastive_loss(xenium_h, sc_pos_h, sc_neg_h)
            
            # Prediction task
            _, preds = model(batch['xenium'].cuda(), 'xenium')
            targets = get_imputation_targets(batch['xenium']) # Function to get target genes
            loss_pred = prediction_loss(preds, targets.cuda())
            
            # Combined loss
            loss = 0.3 * loss_contrast + 0.7 * loss_pred
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return model