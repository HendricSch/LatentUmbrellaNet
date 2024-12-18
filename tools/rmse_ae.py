from models.ae import Autoencoder
from dataset.geopotential_vae import GeopotentialDatasetVAE
from matplotlib import pyplot as plt
import numpy as np

model = Autoencoder.load_from_checkpoint(
    "logs/ae/version_1/checkpoints/epoch=19-step=8279.ckpt"
)
# load model to cpu
model.to("cpu")
model.eval()
model.freeze()

ds = GeopotentialDatasetVAE(mode="test", data_dir="./dataset")


sample = ds[20].unsqueeze(0)
prediction = model(sample)

x = ds[20][0].numpy()
x = (x * ds.std) + ds.mean
x_hat = prediction[0, 0].detach().numpy()
x_hat = (x_hat * ds.std) + ds.mean

rmse = np.sqrt(np.mean((x - x_hat) ** 2))

print(f"RMSE: {rmse}")

# Plot the original, reconstructed, and test
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(x, cmap="viridis")
plt.title("Original")
plt.colorbar()
plt.subplot(132)
plt.imshow(x_hat, cmap="viridis")
plt.title("Reconstructed")
plt.colorbar()

plt.show()
