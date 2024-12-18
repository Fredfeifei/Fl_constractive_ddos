# **FL_Contrastive_DDoS**

## 🚀 Overview
This project focuses on detecting Distributed Denial-of-Service (DDoS) attacks using Federated Learning and Contrastive Learning techniques. The **CICIDS-2019 DDoS** dataset serves as the foundation for our analysis and model training.

---

## 📂 Dataset
- **Source**: [CICIDS-2019 DDoS Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
- The raw data is preprocessed to clean and engineer features, making it ready for training and evaluation.

---

## 🛠️ How to Use

### 1️⃣ Preprocess the Data
Run the `data_preprocessing.py` script to clean the raw dataset and perform feature engineering.

### 2️⃣ Train the Encoder
Train the encoder using the Variational Autoencoder (VAE) model. This step extracts the latent representation 𝑧(𝑥)
z(x) for the input data.

### 3️⃣ Fine-Tune the Model
Fine-tune the model using contrastive learning techniques to enhance its performance.

```bash
python data_preprocessing.py

python main_VaE.py

python main_constractive.py


Raw Data (CICIDS-2019)
       |
Data Cleaning & Preprocessing
       |
     VAE Training
(Encoder learns latent z(x))
       |
Calculate z(x)
(Local Prototypes --> Global Prototype)
       |
  New Attack Data
       |
Compare z(x) with Global Prototype
       |
   Inference & Detection
