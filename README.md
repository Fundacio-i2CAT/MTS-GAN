# MTS-GAN

## Description
This repository contains the official implementation of Mamba Time-Series GAN (MTS-GAN), a Generative Adversarial Network (GAN) designed for multivariate time-series generation.  
The project also includes a reimplementation of TTS-GAN as a baseline model for comparison.  
MTS-GAN has been evaluated on two open-source datasets, resulting in five different experimental scenarios.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MTS-GAN.git
   cd MTS-GAN
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

3. Initialize MLflow for experiment tracking:
   ```bash
   mlflow ui
   ```
   By default, MLflow will start at http://localhost:5000, if the is changed please change it too in the config.yaml

## Usage
1. Datasets  
   Place datasets in the `data/` directory.  
   The expected data format is:
   ```
   [num_sequences, sequence_length, num_features]
   ```
    And parquet format is required.

2. Configurations  
   10 experimental configurations are provided in the `configs/` directory.  5 for MTS-GAN and 5 for TTS-GAN
   Example YAML configuration: `configs/config_ett96.yaml`

3. Training  
   To train a model, execute:
   ```bash
   python3 main.py --config configs/config_ett96.yaml
   ```
4. Inference
   To make inference of a model:
   ```bash
   python3 utils/inference.py \
    --tracking_uri http://0.0.0.0:5000 \
    --run_id <YOUR_RUN_ID> \
    --num_samples 10
   ```

## Repository Structure
```
MTS-GAN/
│
├── architecture/        # Network architectures (tts_gan, mts_gan)
├── core/                # Training, validation, and model logic
├── data/                # Datasets and data loaders
├── utils/               # Utility scripts (logging, metrics, visualization)
├── configs/             # YAML configuration files for different scenarios
├── main.py              # Entry point for training
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```


## Authors and Acknowledgment
Developed by Santiago Escuder (I2CAT Foundation).  
Acknowledgments to the authors of the [TTS-GAN](https://github.com/imics-lab/tts-gan) and [Mamba](https://github.com/state-spaces/mamba) repositories for their foundational contributions and open-source releases, which served as the basis for this implementation.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Support
For questions, issues, or collaboration inquiries, please contact:  
santiago@escuder.i2cat.net