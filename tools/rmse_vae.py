from models.vae import VAE
from dataset.geopotential_vae import GeopotentialDatasetVAE
from matplotlib import pyplot as plt
import numpy as np

model = VAE.load_from_checkpoint(
    "logs/vae/version_7/checkpoints/epoch=19-step=8279.ckpt"
)
# load model to cpu
model.to("cpu")
model.eval()
model.freeze()

ds = GeopotentialDatasetVAE(mode="test", data_dir="./dataset")


sample = ds[10].unsqueeze(0)
prediction, encoder_out = model(sample)

x = ds[10][0].numpy()
x = (x * ds.std) + ds.mean
x_hat = prediction[0, 0].detach().numpy()
x_hat = (x_hat * ds.std) + ds.mean

rmse = np.sqrt(np.mean((x - x_hat) ** 2))

print(f"RMSE: {rmse}")

print(encoder_out.shape)

test = encoder_out[0, 1].detach().numpy()
print(test.shape)

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
plt.subplot(133)
plt.imshow(test)
plt.title("Encoder output")
plt.colorbar()

plt.show()
