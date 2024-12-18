from models.vqvae import VQVAE
from dataset.geopotential_vae import GeopotentialDatasetVAE
from matplotlib import pyplot as plt
import numpy as np

model = VQVAE.load_from_checkpoint(
    "logs/vqvae/version_11/checkpoints/epoch=23-step=9135.ckpt"
)
# load model to cpu
model.to("cpu")
model.eval()
model.freeze()

ds = GeopotentialDatasetVAE(mode="test", data_dir="./dataset")

encoder_outputs = []

for i in range(16):
    sample = ds[100 + i].unsqueeze(0)
    prediction, encoder_output, _ = model.forward(sample)

    x = ds[100 + i][0].numpy()
    x = (x * ds.std) + ds.mean
    x_hat = prediction[0, 0].detach().numpy()
    x_hat = (x_hat * ds.std) + ds.mean

    rmse = np.sqrt(np.mean((x - x_hat) ** 2))

    print(f"RMSE: {rmse}")

    encoder_output = encoder_output[0].detach().numpy().T
    encoder_output_min = encoder_output.min()
    encoder_output_max = encoder_output.max()
    encoder_output = (encoder_output - encoder_output_min) / (
        encoder_output_max - encoder_output_min
    )
    encoder_outputs.append(encoder_output)


plt.figure(figsize=(12, 6))

for i, encoder_output in enumerate(encoder_outputs):
    plt.subplot(4, 4, i + 1)
    plt.imshow(encoder_output, cmap="magma")
    plt.title(f"Latent space {i}")
    plt.colorbar()

plt.show()


# Plot the original, reconstructed, and test
# plt.figure(figsize=(12, 6))
# plt.subplot(131)
# plt.imshow(x, cmap="viridis")
# plt.title("Original")
# plt.colorbar()
# plt.subplot(132)
# plt.imshow(x_hat, cmap="viridis")
# plt.title("Reconstructed")
# plt.colorbar()

# plt.show()
