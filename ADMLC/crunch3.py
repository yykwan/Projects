import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from gseapy import enrichr
from scipy.stats import mannwhitneyu

class DysplasiaMarkerDetector:
    def __init__(self, crunch3_data, sc_data, pretrained_model):
        self.he_image = load_tiff(crunch3_data['he_path'])
        self.masks = {
            'normal': load_tiff(crunch3_data['mask_path']) == 1,
            'dysplasia': load_tiff(crunch3_data['mask_path']) == 2
        }
        self.sc_data = sc.read_h5ad(sc_data)
        self.model = pretrained_model
        
        # Preprocess single-cell data
        self._preprocess_sc_data()
        
    def _preprocess_sc_data(self):
        """Normalize and filter single-cell data"""
        sc.pp.filter_genes(self.sc_data, min_cells=10)
        sc.pp.normalize_total(self.sc_data, target_sum=1e4)
        sc.pp.log1p(self.sc_data)
        sc.pp.highly_variable_genes(self.sc_data, n_top_genes=5000)
        self.sc_data = self.sc_data[:, self.sc_data.var.highly_variable]
        
    def predict_tissue_expression(self):
        """Predict gene expression across entire tissue"""
        # Extract patches in sliding window
        patches = extract_sliding_patches(self.he_image, window_size=256, stride=128)
        
        # Predict expression for each patch
        self.expression_map = []
        with torch.no_grad():
            for batch in DataLoader(patches, batch_size=64):
                self.expression_map.append(self.model(batch.cuda()).cpu().numpy())
        self.expression_map = np.vstack(self.expression_map)
        
        # Assign predictions to mask regions
        self.normal_expr = self.expression_map[self.masks['normal'].flatten()]
        self.dysplasia_expr = self.expression_map[self.masks['dysplasia'].flatten()]
        
    def identify_markers(self, method='integrated'):
        """Identify marker genes using multiple approaches"""
        # Differential expression analysis
        de_results = []
        for gene in range(self.expression_map.shape[1]):
            _, pval = mannwhitneyu(self.normal_expr[:, gene], 
                                  self.dysplasia_expr[:, gene])
            fc = np.median(self.dysplasia_expr[:, gene]) / np.median(self.normal_expr[:, gene])
            de_results.append((gene, fc, pval))
        
        # Single-cell evidence
        sc_markers = self._analyze_sc_markers()
        
        # Pathway enrichment
        pathway_scores = self._pathway_analysis()
        
        # Combine evidence
        if method == 'integrated':
            return self._integrated_ranking(de_results, sc_markers, pathway_scores)
        else:
            return self._default_ranking(de_results)
    
    def _analyze_sc_markers(self):
        """Find markers in single-cell data"""
        # Compare normal vs dysplasia samples
        normal = self.sc_data[self.sc_data.obs['dysplasia'] == 'n']
        dysplasia = self.sc_data[self.sc_data.obs['dysplasia'] == 'y']
        
        # Compute mutual information
        labels = np.concatenate([np.zeros(len(normal)), np.ones(len(dysplasia))])
        expr = np.vstack([normal.X, dysplasia.X])
        mi_scores = mutual_info_classif(expr, labels)
        
        return {gene: score for gene, score in zip(self.sc_data.var_names, mi_scores)}
    
    def _pathway_analysis(self):
        """Perform pathway enrichment analysis"""
        # Get top differentially expressed genes
        top_genes = [...] # Select based on DE analysis
        
        # Run enrichment analysis
        enr = enrichr(gene_list=top_genes,
                     gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2023'])
        
        # Score genes by pathway importance
        pathway_scores = {}
        for pathway in enr.results:
            for gene in pathway['genes'].split(';'):
                pathway_scores[gene] = pathway_scores.get(gene, 0) + -np.log10(pathway['pval'])
        
        return pathway_scores
    
    def _integrated_ranking(self, de_results, sc_markers, pathway_scores):
        """Combine multiple evidence sources"""
        rankings = []
        for gene, fc, pval in de_results:
            gene_name = self.sc_data.var_names[gene]
            score = (
                0.4 * -np.log10(pval) + 
                0.3 * np.log2(fc) + 
                0.2 * sc_markers.get(gene_name, 0) + 
                0.1 * pathway_scores.get(gene_name, 0)
            )
            rankings.append((gene_name, score))
        
        # Sort and rank
        rankings.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(rankings, columns=['gene', 'score'])

def create_final_submission(detector, output_path):
    """Generate submission file with proper formatting"""
    ranking = detector.identify_markers()
    
    # Format according to challenge requirements
    submission = pd.DataFrame({
        'gene': ranking['gene'],
        'rank': range(1, len(ranking)+1)
    })
    
    # Ensure all 18,615 genes are included
    all_genes = pd.read_csv('genes_reference.csv') # Provided by challenge
    final_submission = all_genes.merge(submission, on='gene', how='left')
    final_submission['rank'] = final_submission['rank'].fillna(len(ranking)+1)
    final_submission = final_submission.sort_values('rank')[['gene', 'rank']]
    
    final_submission.to_csv(output_path, index=False)