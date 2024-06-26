# Anime Face Generation using DCGAN

This project focuses on generating anime faces using a Deep Convolutional Generative Adversarial Network (DCGAN). The network is trained on a dataset of anime faces to produce realistic and high-quality anime face images.

## Project Overview

### Introduction

Generative Adversarial Networks (GANs) are a class of neural networks designed by Ian Goodfellow and his colleagues in 2014. They consist of two neural networks, the generator and the discriminator, which compete against each other to improve the overall performance. This project utilizes a special type of GAN, the Deep Convolutional GAN (DCGAN), to generate anime faces.

### Dataset

The dataset used for this project is the Anime Face Dataset from Kaggle, which contains over 63,000 images of anime faces. The dataset is loaded and preprocessed to ensure optimal training performance.

### Network Architecture

#### Discriminator
The discriminator is a Convolutional Neural Network (CNN) that classifies images as real or fake. It uses Leaky ReLU activations and batch normalization to stabilize training.

#### Generator
The generator is a transposed CNN that generates new images from random noise vectors. It uses ReLU activations and batch normalization to produce realistic images.

### Training

The training process involves alternating between training the discriminator and the generator. The discriminator learns to distinguish between real and fake images, while the generator learns to create images that are indistinguishable from real ones. The networks are trained on a GPU to speed up the training process.

### Results

The generated images improve in quality as the training progresses. Sample images are saved at regular intervals to visualize the generator's progress.


![generated-images-0025](https://github.com/Vaibhavi09/GenAi_projects/assets/88539906/b84d8a32-aece-45e2-a19a-0cb7a0229bac)






### Dependencies

- Python
- PyTorch
- torchvision
- numpy
- matplotlib
- opendatasets
- jovian

### How to Run

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the dataset from Kaggle and place it in the appropriate directory.
4. Run the Jupyter notebook to start training the model.

### Jupyter Notebook

The project includes a Jupyter notebook that contains the complete code for training the DCGAN. It includes data loading, preprocessing, model definition, training loop, and sample generation.

### Contributions

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the MIT License.

### Acknowledgments

- Kaggle for providing the Anime Face Dataset.
- PyTorch and torchvision for the deep learning framework and utilities.
- Jovian for experiment tracking and management.


