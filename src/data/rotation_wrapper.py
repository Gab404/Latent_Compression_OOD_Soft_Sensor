import torch

class RotatedEncoderWrapper:
    """
    Intercepte un encodeur Raman (CDAE, CVAE, Beta-VAE) et applique une 
    rotation orthogonale aléatoire fixe (Q) à son espace latent.
    """
    def __init__(self, base_encoder, seed=42):
        self.base_encoder = base_encoder
        self.device = base_encoder.device
        self.latent_dim = 64
        
        # Génération de la matrice orthogonale Q (64x64)
        torch.manual_seed(seed)
        H = torch.randn(self.latent_dim, self.latent_dim)
        Q, R = torch.linalg.qr(H)
        Q = Q * torch.sign(torch.diag(R)) # Uniformisation de Haar
        
        self.Q = Q.to(self.device)
        print("\n" + "!"*70)
        print(" [EXPERIMENT] ⚠️ ROTATION ORTHOGONALE APPLIQUÉE À L'ESPACE LATENT ⚠️")
        print("!"*70 + "\n")

    def encode(self, x):
        out = self.base_encoder.encode(x)
        if isinstance(out, tuple):
            mu, logvar = out
            return mu @ self.Q, logvar
        return out @ self.Q

    def encode_dataframe(self, df):
        Z = self.base_encoder.encode_dataframe(df)
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)
        Z_rot = Z_tensor @ self.Q
        return Z_rot.cpu().numpy()
        
    def __getattr__(self, name):
        return getattr(self.base_encoder, name)