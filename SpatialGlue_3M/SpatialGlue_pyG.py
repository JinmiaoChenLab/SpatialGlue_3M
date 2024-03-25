import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing

class Train_SpatialGlue:
    def __init__(self, 
        data,
        datatype = 'Triplet',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input  
            The default is 'Triplet'. To date, real-worlk triplet modality data is still unavailable. We define default data type as 'Triplet' temporarily.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_cell_omics3 = self.adata_omics3.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.dim_output3 = self.dim_output
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, self.dim_input3, self.dim_output3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.features_omics3, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_spatial_omics3, self.adj_feature_omics3)
            
            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            self.loss_recon_omics3 = F.mse_loss(self.features_omics3, results['emb_recon_omics3'])
            
            # correspondence loss
            self.loss_corr_omics1_omics2 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_omics2_across_recon'])
            self.loss_corr_omics1_omics3 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_omics3_across_recon'])
            
            self.loss_corr_omics2_omics1 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_omics1_across_recon'])
            self.loss_corr_omics2_omics3 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_omics3_across_recon'])
            
            self.loss_corr_omics3_omics1 = F.mse_loss(results['emb_latent_omics3'], results['emb_latent_omics3_omics1_across_recon'])
            self.loss_corr_omics3_omics2 = F.mse_loss(results['emb_latent_omics3'], results['emb_latent_omics3_omics2_across_recon'])
            
                
            loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2 + self.weight_factors[2]*self.loss_recon_omics3 \
                   + self.weight_factors[3]*self.loss_corr_omics1_omics2 + self.weight_factors[4]*self.loss_corr_omics1_omics3  \
                   + self.weight_factors[5]*self.loss_corr_omics2_omics1 + self.weight_factors[6]*self.loss_corr_omics2_omics3  \
                   + self.weight_factors[7]*self.loss_corr_omics3_omics1 + self.weight_factors[8]*self.loss_corr_omics3_omics2
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.features_omics3, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_spatial_omics3, self.adj_feature_omics3)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
                  'SpatialGlue': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha_omics3': results['alpha_omics3'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}
        
        return output
    
    
    
        
    
    
      

    
        
    
    
