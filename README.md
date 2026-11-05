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
   - Place your datasets in the `data/` directory.
   - The expected data format is:
     [num_sequences, sequence_length, num_features]
     and the files must be in Parquet format.

   - To use the ETT dataset:
     1. Download the raw dataset from the GitHub repository: [ETTh1](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv)
     2. Place it under:
        data/ETT/ETT_raw/
     3. Then run the processing script:
        python3 data/ett_processing.py
 
   - To use the UniMiB‑SHAR dataset:
     1. Download the raw dataset from:
        [UniMiB‑SHAR](https://www.sal.disco.unimib.it/technologies/unimib-shar/)
        or from [Kaggle] (https://www.kaggle.com/datasets/alirezacman/unimib-shar)
     2. Place it under:
        data/UniMiB‑SHAR/UniMiB‑SHAR_raw/
     3. Then run the processing script:
        python3 data/unimib_processing.py

   - These scripts will convert and save the datasets into the correct format for training your network.

2. Configurations  
   10 experimental configurations are provided in the `configs/` directory.  5 for MTS-GAN and 5 for TTS-GAN
   Example YAML configuration: `configs/mts-gan/ETT_96.yaml`

3. Training  
   To train a model, execute:
   ```bash
   python3 main.py --config configs/mts-gan/ETT_96.yaml
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
Developed by Santiago Escuder Folch (I2CAT Foundation) and Oriol Graupera-Serra (I2CAT Foundation).  
Acknowledgments to the authors of the [TTS-GAN](https://github.com/imics-lab/tts-gan) and [Mamba](https://github.com/state-spaces/mamba) repositories for their foundational contributions and open-source releases, which served as the basis for this implementation.


# Source
This code has been developed within the ExtremeXP project (EXPerimentation driven and user eXPerience oriented analytics for eXtremely Precise outcomes and decisions), 
which has received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No. 101093164.

More information about the project is available at: https://cordis.europa.eu/project/id/101093164/reporting

# Copyright
This code has been developed by Fundació Privada Internet i Innovació Digital a Catalunya (i2CAT).
i2CAT is a *non-profit research and innovation centre* that  promotes mission-driven knowledge to solve business challenges, co-create solutions with a transformative impact, empower citizens through open and participative digital social innovation with territorial capillarity, and promote pioneering and strategic initiatives.
i2CAT *aims to transfer* research project results to private companies in order to create social and economic impact via the out-licensing of intellectual property and the creation of spin-offs.
Find more information of i2CAT projects and IP rights at https://i2cat.net/tech-transfer/

# License
This code is licensed under the terms GNU Affero General Public License v3.0 (*AGPLv3*). Information about the license can be located at [link](https://www.gnu.org/licenses/agpl-3.0.html).
If you find that this license doesn't fit with your requirements regarding the use, distribution or redistribution of our code for your specific work, please, don’t hesitate to contact the intellectual property managers in i2CAT at the following address: techtransfer@i2cat.net

# Citation
If you find this repo useful, please cite our paper via
```bibtex
@article{folch2026mts,
  title={MTS-GAN: Synthetic multivariate time series generation with the Mamba architecture},
  author={Folch, Santiago Escuder and Graupera-Serra, Oriol and Calvo, Albert and Mu{\~n}oz, Borja and Guti{\'e}rrez, Gorka and Escrig, Josep and Compasti{\'e}, Maxime},
  journal={Knowledge-Based Systems},
  pages={115996},
  year={2026},
  publisher={Elsevier}
}
```
## Support
For questions, issues, or collaboration inquiries, please contact:  
santiago.escuder@i2cat.net
oriol.graupera@i2cat.net
albert.calvo@i2cat.net
