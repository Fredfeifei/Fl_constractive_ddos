# **FL_Contrastive_DDoS**

## 🚀 Overview
This project focuses on detecting Distributed Denial-of-Service (DDoS) attacks using Federated Learning and Contrastive Learning techniques. The **CICIDS-2019 DDoS** dataset serves as the foundation for our analysis and model training.

---

## 📂 Dataset
- **Source**: [CICIDS-2019 DDoS Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
- The raw data is preprocessed to clean and engineer features, making it ready for training and evaluation.

---

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

## 🛠️ How to Use

### 1️⃣ Preprocess the Data
Run the `data_preprocessing.py` script to clean the raw dataset and perform feature engineering.

```bash
python data_preprocessing.py


