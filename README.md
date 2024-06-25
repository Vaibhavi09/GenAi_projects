## Anime Face Generator using DCGAN

This project uses Deep Convolutional Generative Adversarial Networks (DCGAN) to generate anime faces. Below are the steps to set up the environment, download the dataset, and train the model.

### Installation

First, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/anime-dcgan.git
cd anime-dcgan
pip install -r requirements.txt
```

### Dataset

The dataset used for this project is the Anime Face Dataset, which can be downloaded from Kaggle.

1. Download the dataset:

```python
import opendatasets as od

dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
od.download(dataset_url)
```

2. Verify the downloaded files:

```python
import os

DATA_DIR = './animefacedataset'
print(os.listdir(DATA_DIR))
print(os.listdir(DATA_DIR+'/images')[:10])
```

### Training

The training process involves defining the discriminator and generator models, moving them to the GPU (if available), and then training them using the specified hyperparameters.

1. Define the Discriminator and Generator:

```python
import torch
import torch.nn as nn

# Discriminator
discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)

# Generator
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)
```

2. Move models to the GPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = discriminator.to(device)
generator = generator.to(device)
```

3. Define training functions:

```python
def train_discriminator(real_images, opt_d):
    opt_d.zero_grad()
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = nn.functional.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = nn.functional.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt_g):
    opt_g.zero_grad()
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = nn.functional.binary_cross_entropy(preds, targets)
    loss.backward()
    opt_g.step()
    return loss.item()
```

4. Training loop:

```python
from tqdm.notebook import tqdm

def fit(epochs, lr):
    torch.cuda.empty_cache()
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            loss_g = train_generator(opt_g)
        print(f"Epoch [{epoch+1}/{epochs}], loss_g: {loss_g:.4f}, loss_d: {loss_d:.4f}, real_score: {real_score:.4f}, fake_score: {fake_score:.4f}")
        save_samples(epoch+1, fixed_latent, show=False)
    return losses_g, losses_d, real_scores, fake_scores

lr = 0.0002
epochs = 25
history = fit(epochs, lr)
```

5. Save generated samples:

```python
from torchvision.utils import save_image

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent)
```

### Logging and Tracking

Use Jovian to log hyperparameters and track the training process:

```python
import jovian

jovian.reset()
jovian.log_hyperparams(lr=lr, epochs=epochs)
```

### Results

Generated images will be saved in the `generated` directory after each epoch. Below are some sample generated images from different epochs:

- Epoch 1:
  ![Epoch 1](generated/generated-images-0001.png)
- Epoch 5:
  ![Epoch 5](generated/generated-images-0005.png)
- Epoch 10:
  ![Epoch 10](generated/generated-images-0010.png)

### Acknowledgements

- Kaggle for the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- [PyTorch](https://pytorch.org/) for the deep learning framework
