import pandas as pd

df = pd.read_csv("latent_activations_beta005_train.csv")  # columns: z_38, z_4
r = df["z_38"].corr(df["z_4"])
print(f"|r| = {abs(r):.6f}")
