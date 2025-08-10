# Autoimmune Disease Machine Learning Challenge
## Goal
1. Crunch1: Predicting spatial transcriptomics data from pathologty images  
2. Crunch2: Predicting unmeasured genes in spatial transcriptomics using both pathology images and single-cell transcriptomics data
3. Crunch3: Identifying gene markers of pre-cancerous tissue regions in IBD

## Datasets Provided
1. H&E stained pathology images - Standard histological images showing tissue morphology
2. Xenium spatial transcriptomics data - High-resolution gene expression measurements for 460 genes
3. Single-cell RNA-seq data - Gene expression profiles from dissociated colon tissues
4. DAPI images and nuclear seqmentation masks - For aligning H&E and Xenium modalities

## Approach for each crunch
### Crunch 1:
#### Approach: 
1. Multi-modal Data Fusion: combines image features from H&E stains with spatial context from neighboring cells, uses both local cellular patterns and global tissue organization
2. Hierachical Architecture: 
    ```class HierarchicalSpatialPredictor(nn.Module):
        def __init__(self):
            # Image encoder (pretrained ViT)
            self.image_encoder = timm.create_mode('vit_base_patch16_224')
        
            # Spatial attention for neighbors
            self.neighbor_attn = nn.TransformerEncoderLayer()
        
            # Fusion and prediction
            self.fusion = nn.TransformerEncoderLayer()
            self.predictor = nn.Sequential(...)```
3. Key Components: 
    1. Multi-scale image patches: Extract features at 64px,128px and 256px scales
    2. Spatial attention: Models relationships between neighboring cells using their expression and coordinates
    3. Cross-validation
#### Flow:
1. Extract H&E patches centered on each nucleus
2. Encode patches using Vision Transformer
3. Aggregate features from neighboring cells with spatial attention
4. Fuse image an spatial features
5. Predict 460 dimensioinal gene expression vector
### Crunch 2:
#### Approach:
1. Cross-modal Alignment: 
    ```class GeneImputationModel(nn.Module):
        def __init__(self):
            # Shared encoder
            self.encoder = nn.Sequential(...)
            
            # Projection heads
            self.proj_xenium = nn.Sequential(...)
            self.proj_scrnaseq = nn.Sequential(...)
            
            # Prediction head
            self.predictor = nn.Sequential(...)```
2. Contrastive Learning(bring Xenium and scRNA-seq closer for similar cells and push apart for dissimilar cells): ```contrastive_loss = nn.TripletMarginLoss(margin=1.0)```
3. Combine contrastive loss with prediction loss, weighted sum (0.3 contrastive +0.7 prediction)
#### Flow:
1. Normalize both Xenium and scRNA-seq data (transform log1p)
2. Find nearest between modalities
3. Create +/- pairs for contrastive learning
4. Train with alternating batches from both modalities
### Crunch 3:
#### Integrated Approach:
    ```def identify_markers(self, method='integrated'):
        # Differential expression
        de_results = mannwhitneyu(normal_expr, dysplasia_expr)
        
        # Single-cell evidence
        sc_markers = self._analyze_sc_markers()
        
        # Pathway enrichment
        pathway_scores = self._pathway_analysis()
        
        # Combined scoring
        return self._integrated_ranking(de_results, sc_markers, pathway_scores)```
#### Integration with more than one evidece:
1. Tissue level: Mann-Whitney U test for differential expreession and calculations for fold changes
2. Validation for single cell: Use mutual info for market identification and dysplasia-annoted scRNA-seq data
## Conclusion
Although submission was not accepted, this was a fun and enriching competition.





