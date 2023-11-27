# GM-model-server

## Overview

GM-model-server is a server application designed for handling the training and generation tasks related to Generative Models, specifically using DCGAN (Deep Convolutional Generative Adversarial Networks). 
It provides functionalities for downloading datasets, training models, generating images, and sending evaluation metrics.

## Features

- **Dataset Handling**: Download datasets from specified URLs and extract them.
- **Model Training**: Train DCGAN models on the downloaded datasets.
- **Image Generation**: Generate images using trained models.
- **Metric Evaluation**: Calculate and send evaluation metrics such as accuracy, FID, and LPIPS.

## Getting Started

To get started with GM-model-server, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/GM-model-server.git
   cd GM-model-server
   ```

2. **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the Server**:
  ```bash
  uvicorn main:app --reload --port your_port_number 
  ```

## Configuration
Adjust the server configuration in the .env file.
