# LatentUmbrellaNet

LatentUmbrellaNet (LUN) ist ein zweistufiges Wettervorhersagemodell:

1) Ein Autoencoder (VAE, VQ-VAE) komprimiert ERA5-Wetterdaten in einen latenten Raum und rekonstruiert diese wieder in den Pixelraum.
2) Ein Vorhersagenetz (UNet oder AFNO) sagt im latenten Raum den nächsten Zeitschritt vorher. Die latente Vorhersage wird anschließend vom Decoder zurück in den Pixelraum transformiert.

Der Datensatz wird direkt aus dem ARCO-ERA5 Zarr-Store auf Google Cloud (öffentlich) in Batches gestreamt. Das Projekt verwendet PyTorch Lightning für Training und Logging sowie Dask/xarray/xbatcher für effiziente Datenpipelines.

## Voraussetzungen und Installation

Empfohlen: Python 3.12+, CUDA-fähige GPU (Training), ausreichend RAM/VRAM.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS (Bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Datenzugriff (ARCO-ERA5)

Die Daten werden direkt aus dem öffentlichen Zarr-Store geladen:

- Bucket: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- Zugriff: anonym (kein Token nötig)

Der Dataloader in `data/era5.py` nutzt `xarray` + `xbatcher`, um 4D Batches (Zeit × Variablen × Lon × Lat) zu erzeugen. Für Training/Validierung wird standardmäßig 1979–2022 bzw. 2023–2024 verwendet.

## Training des Autoencoders (VAE, VQ-VAE)

Konfiguration prüfen/anpassen (z. B. `configs/autoencoder/kl-f8-disc.yaml`). Dann:

```bash
python train.py
```

Parameter wie Batchgröße, Epochen, Lernrate etc. sind in der YAML-Datei (`config`-Schlüssel) definiert.

## Erzeugen eines latenten Zarr-Datastores

Nach dem (oder mit einem vorhandenen) Autoencoder können latente Darstellungen als Zarr-Store erzeugt werden. Pfade sind im Skript konfiguriert (`LATENT_STORE_PATH`, Checkpoint/Config):

```bash
python data/generate_latent_datastore.py
```

## Training des Vorhersagenetzes (U-Net/AFNO)

Das Vorhersagemodell lernt, aus zwei aufeinanderfolgenden latenten Zeitpunkten die nächste latente Darstellung vorherzusagen:

```bash
python train_predictionmodel.py
```

Standardmäßig wird das Attention-U-Net Modell verwendet. AFNO ist ebenfalls enthalten (siehe `models/predictionnet.py` und `models/afnonet.py`).


## Inferenz mit LatentUmbrellaNet

Die Kombination aus Autoencoder und Vorhersagenetz ist in `models/latent_umbrella_net.py` implementiert. Beispiel:

```python
import torch
from models.latent_umbrella_net import LatentUmbrellaNet

lun = LatentUmbrellaNet(
    vae_ckpt_path="checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt",
    vae_config_path="configs/autoencoder/kl-f8-disc.yaml",
    prediction_net_type="unet",  # oder "afno"
    prediction_net_ckpt_path="checkpoints/prediction-model-val_loss=0.01241.ckpt",
    device="cuda",
)

x_1 = torch.randn(1, 69, 1440, 721)
x_2 = torch.randn(1, 69, 1440, 721)

# Ein Schritt (gibt (1, 69, 1440, 720) zurück)
forecast = lun(x_1, x_2, forecast_steps=1)
print(forecast.shape)
```

## Evaluation (Baselines und Modelle)

`evaluation.py` bietet mehrere Routinen und speichert Ergebnisse als CSV unter `evaluation/`:

- Persistence: `eval_persistence(forecast_steps, rounds, save_to_csv)`
- Climatology: `eval_climatology(sample_size, rounds, save_to_csv)`
- VAE-Rekonstruktionsfehler: `eval_vae_error(rounds, save_to_csv)`
- LUN + UNet: `eval_lun_unet(forecast_steps, rounds, save_to_csv)`
- LUN + AFNO: `eval_lun_fourcastnet(forecast_steps, rounds, save_to_csv)`

## Konfigurationen (YAML)

Die YAML-Dateien unter `configs/` steuern Daten- und Modellparameter. Wichtige Felder unter `config`:

- data: `in_channels`, `out_channels`, `x`, `y`
- model: `z_channels`, `embed_dim`, `channels`, `channel_mult`, `num_res_blocks`, `attention`, `num_heads`, `norm_groups`
- training: `batch_size`, `epochs`, `learning_rate`, `lr_scheduler`
- loss: `reconstruction_loss` ("l1" | "mse" | "rmse"), `kl_weight`, `discriminator_weight`, `discriminator_start_steps`

Pfade zu Checkpoints und Zarr-Stores werden direkt in den Skripten (`train.py`, `train_predictionmodel.py`, `data/generate_latent_datastore.py`) oder beim Erstellen des `LatentUmbrellaNet` übergeben.


## Quellen

- Stable Diffusion / Latent Diffusion:
  - Rombach, A., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
  - CompVis, Stable Diffusion Repository: [github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

- FourCastNet / AFNO:
  - Pathak, J., Subramanian, S., Harrington, P., et al. (2022). FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators. [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
  - NVLabs, AFNO-Transformer Repository: [github.com/NVlabs/AFNO-transformer](https://github.com/NVlabs/AFNO-transformer)

- PatchGAN Discriminator:
  - Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks ("pix2pix"). [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)