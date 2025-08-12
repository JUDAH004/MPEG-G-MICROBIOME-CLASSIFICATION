# 🧬 MPEG-G-MICROBIOME-CLASSIFICATION
This project aims to build a classification model that can predict microbiome samples by body site

---
## Table of Contents
- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Description](#data-description)
- [Pipeline Workflow](#pipeline-workflow)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
    - [3. Feature Engineering](#3-feature-engineering)
    - [4. Modeling](#4-modeling)
    - [5. Model Explainability](#5-model-explainability)
    - [6. Deployment](#6-deployment)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [README.md Summary](#In'Summary)
- [Citation](#citation)
- [Contact](#contact)

---

## Project Overview

This project aims to classify human microbiome samples by body site (Mouth, Nasal, Skin, Stool) using 16S rRNA gene profiles stored in MPEG-G format, along with participant metadata and cytokine profiles. The pipeline covers data decoding, feature extraction, exploratory analysis, model training (RandomForest, XGBoost, LightGBM), explainability (SHAP, LIME), and deployment via a Gradio web app.

---

## Business Understanding

- **Problem:** Accurately determine the body site origin of a microbiome sample using genomic and metadata features.
- **Objectives:**
    - Build a robust machine learning classifier for microbiome sample origin.
    - Decode MPEG-G files to FASTQ format for analysis.
    - Extract and engineer relevant microbial features.
    - Evaluate models using accuracy, precision, recall, F1-score, confusion matrix, ROC, and AUC.
    - Provide model explainability for transparency.
    - Deploy the best model for interactive use.

- **Stakeholders:** Bioinformatics researchers, clinical microbiologists, genomics data scientists.

---

## Data Description

- **Source:** [Zindi MPEG-G Microbiome Classification Challenge](https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge)
- **Files:**
    - `TrainFiles.zip`, `TestFiles.zip`: MPEG-G compressed 16S rRNA sequences (`.mgb` files).
    - `Train.csv`, `Test.csv`: Sample metadata (with/without labels).
    - `Train_Subjects.csv`: Participant info (age, sex, insulin sensitivity).
    - `cytokine_profiles.csv`: Immune marker data.
    - `SampleSubmission.csv`: Submission template.

- **Classes:** Nasal, Mouth, Skin, Stool.

---

## Pipeline Workflow

### 1. Data Preparation

- **Decompression:**  
  Decode `.mgb` files to `.fastq` using Docker and Genie, then compress to `.fastq.gz`.
- **Feature Extraction:**  
  Extract read-level features (average read length, GC content, N content, read count) from FASTQ files using Biopython.
- **Metadata Merge:**  
  Merge sequence features with tabular metadata for modeling.

### 2. Exploratory Data Analysis

- Visualize sample type distribution and files per subject.

### Sample Type Distribution 

<img width="424" height="280" alt="58f2b56a-cdd1-4825-ba56-46c8d18e01c5" src="https://github.com/user-attachments/assets/0f041240-4849-4700-af92-090da19aab3d" />


### Distribution of files per subject 

<img width="606" height="387" alt="1e03f4fb-d79d-4d35-8675-601191485602" src="https://github.com/user-attachments/assets/063c53ca-733c-44c3-bf7c-e3fc82a12036" />

> . The plot above shows the number of files each subject had. In that case, when a subject is included, **four samples** are taken. As seen, most subjects overall had **20-25 samples** taken over the research period; however, some had more than 25 samples taken, like the few subjects who had more than **+ 400 samples** taken.These extreme cases may represent participants who were followed up more intensively, perhaps due to **medical interest** or their **availability**.This also highlights the dedication of these participants and the importance of the research in capturing detailed, longitudinal microbiome data.

- Explore feature distributions and missing values.
- Dimensionality reduction (t-SNE, PCA) for data visualization.

### t-SNE Projection of Microbiome Features
<img width="571" height="424" alt="d45b9f43-2086-4495-ba3d-32e22706c4dc" src="https://github.com/user-attachments/assets/1357de61-49ff-4f1c-ab34-496cc6a10ec0" />

 > . The t-SNE projection captures **non-linear relationships**, grouping microbiome samples into tighter, more distinct local clusters by body site. It excels at showing local similarities but does not preserve global distances.

 > . The t-SNE plot shows **tight local clusters** of **same-class points** (indicating strong local similarity in microbiome composition).

 > . It uses the **perplexity parameter** (like a balance between local and global structure) to decide how many neighbors to consider when building the low-dimensional map.

 > . The t-SNE plot is great for visualizing **high-dimensional biological data** with subtle patterns.

### PCA Projection of Microbiome Features
<img width="568" height="424" alt="55dad154-ef17-4858-a701-434031225e1e" src="https://github.com/user-attachments/assets/589015d2-2c27-4f9e-968a-0187b5f0bf2c" />

> . PCA captures **linear relationships**, overall **variance structure** and is useful for **feature interpretability**

> . **PC1 (Principal Component 1)** captures the direction of maximum variance in the dataset; **PC2** captures the second-highest variance orthogonal to PC1

> . The points are color-coded by body site class.

> . In the plot above, the spread along PC1 and PC2 suggests there is some separation, but many points **overlap**—meaning linear projections don’t fully distinguish classes.

> . The clusters are elongated and overlapping, which is common in biological data due to **subtle differences** between microbiomes.

### 3. Feature Engineering

- Handle missing values (mean/mode imputation).
- One-hot encode categorical variables.
- Align train/test feature columns.
- Standardize features for modeling.

### 4. Modeling

- **RandomForestClassifier:**
  Hyperparameter tuning via GridSearchCV.  
  Validation accuracy: ~71.9%

<img width="568" height="424" alt="0d57ccc3-c335-4e5c-9172-2006a1074bf2" src="https://github.com/user-attachments/assets/86274ab3-0226-4d86-a829-ec23b400adfc" />

>   . The tuned Random Forest model achieved an overall **accuracy** of **71.9%** on the validation set.
    
>  . Performance across classes was generally strong, with **Skin (F1 = 0.83)** and **Mouth (F1 = 0.77)** showing the highest predictive accuracy, indicating clear feature separation for these categories.

>   . **Nasal (F1 = 0.64)** had the weakest performance, suggesting greater overlap with other classes or less distinctive features.

>   . **Stool** performed fairly well **(F1 = 0.65)**

>   . The **macro-average F1-score (0.72)** and **weighted-average F1-score (0.72)** indicate that the model handles class balance reasonably well.

- **XGBoost:**  
  Hyperparameter tuning via GridSearchCV.  
  Validation accuracy: ~86.9%

<img width="568" height="424" alt="02a10b24-5a30-4bce-ad4a-c414da051506" src="https://github.com/user-attachments/assets/b3a77644-aabc-4463-9668-aa837ee5fd7d" />

> . The tuned XGBoost model achieved an **accuracy** of **86.9%** on the validation set.

> . **Skin (F1 = 0.91)** and Mouth **(F1 = 0.88)** were predicted with the highest consistency.

> . **Stool (F1 = 0.86)** also performed well, benefiting from high **recall (0.90)**.

> . **Nasal (F1 = 0.82)** had slightly lower **recall (0.78)**, suggesting some misclassification with other classes, but still demonstrated robust precision. 

> .  **Macro** and **weighted averages** (**both 0.87**) confirm balanced performance across all classes, indicating that the model generalizes well without favoring a specific class.

- **LightGBM:**  
  Hyperparameter tuning via GridSearchCV.  
  Validation accuracy: ~86.4%

<img width="568" height="424" alt="e193e2ac-0f88-4612-8a74-9a107a626585" src="https://github.com/user-attachments/assets/21879cfa-fd9d-4d37-b26d-142b46361d93" />

>  . The tuned LightGBM model achieved an **accuracy** of **86.4%**

>  . **Skin (F1 = 0.90)** had the highest performance with excellent **recall (0.91)**

>  . **Mouth (F1 = 0.87)** and **Stool (F1 = 0.86)** also showed **balanced precision** and **recall**

>  . **Nasal (F1 = 0.83)** again had **slightly lower recall (0.79)**, indicating occasional confusion with other classes

>  .  **Macro** and **weighted** averages **(0.86–0.87)** confirm balanced predictions across all categories, with no strong bias toward a single class

- **Evaluation Metrics:**  
  Accuracy, precision, recall, F1-score, ROC curves, AUC.

### 5. Model Explainability

- **SHAP:**
   Feature importance visualization for each model.

#### **SHAP(SHapley Additive exPlanations) Explainer**
<img width="714" height="676" alt="5d675b3f-565f-49e0-a5d7-a5e4d2b3cbff" src="https://github.com/user-attachments/assets/b9f3f4e1-e945-4fbc-af27-ace2bca08671" />

  > **Feature Importance (SHAP/RandomForest Plot)**

> . The plot above shows that your most influential features used by the RandomForest model in predicting microbiome sample type (Mouth, Nasal, Skin, Stool) are:

| Feature                  | Notes                                                                             |
| ------------------------ | --------------------------------------------------------------------------------- |
| **avg\_read\_length**    | Most influential — possibly reflects sequencing protocol or sample complexity. |
| **gc\_content**          | Strongly indicative of microbial composition.                                     |
| **read\_count**          | A proxy for sequencing depth, which can hint at biomass or complexity.            |
| **Adj\_age**             | Possibly reflects host factors affecting microbiome.                              |
| Others (OGTT, BMI, etc.) | Minor contributions — possibly useful in ensemble or stacked models.              |

> . Biological features (cytokines, metadata) have low influence compared to sequence-derived stats. This suggests the sequencing features carry the bulk of predictive signal. 

#### **SHAP(SHapley Additive exPlanations) Explainer**
<img width="711" height="676" alt="2b96c442-50ab-4b20-b954-0d2f5bfea38e" src="https://github.com/user-attachments/assets/8d6c5f76-0b63-4c88-a3ea-a08639ccf1aa" />
  
  > **Feature Importance (SHAP/XGBoost Plot)**

> . The plot above shows that your most influential features used by the XGBoost model in predicting microbiome sample type (Mouth, Nasal, Skin, Stool) are:

| Feature                  | Notes                                                                             |
| ------------------------ | --------------------------------------------------------------------------------- |
| **avg\_read\_length**    | Most influential — possibly reflects sequencing protocol or sample complexity. |
| **gc\_content**          | Strongly indicative of microbial composition.                                     |
| **read\_count**          | A proxy for sequencing depth, which can hint at biomass or complexity.            |
| **Adj\_age**             | Possibly reflects host factors affecting microbiome.                              |
| Others (OGTT, BMI, etc.) | Minor contributions — possibly useful in ensemble or stacked models.              |

> . Biological features (cytokines, metadata) have low influence compared to sequence-derived stats. This suggests the sequencing features carry the bulk of predictive signal.

#### **SHAP(SHapley Additive exPlanations) Explainer**
<img width="711" height="676" alt="a250778e-3ec1-4344-8d1a-5a9a838e690c" src="https://github.com/user-attachments/assets/8a27567f-8adf-4eda-9fb4-2ff7f87a32d2" />

  > **Feature Importance (SHAP/LighGBM Plot)**

 > . The plot above shows that your most influential features used by the model(LightGBM - Tuned) in predicting microbiome sample type (Mouth, Nasal, Skin, Stool) are:

| Feature                  | Notes                                                                             |
| ------------------------ | --------------------------------------------------------------------------------- |
| **avg\_read\_length**    | Most influential — possibly reflects sequencing protocol or sample complexity. |
| **gc\_content**          | Strongly indicative of microbial composition.                                     |
| **read\_count**          | A proxy for sequencing depth, which can hint at biomass or complexity.            |
| **Adj\_age**             | Possibly reflects host factors affecting microbiome.                              |
| Others (OGTT, BMI, etc.) | Minor contributions — possibly useful in ensemble or stacked models.              |


 > . Biological features (cytokines, metadata) have low influence compared to sequence-derived stats. This suggests the sequencing features carry the bulk of predictive signal.

- **LIME:**  
  Interactive explanations in Gradio app.

---

## KeyTakeaway_for_the_validation_Set_Performance_in_each_model

Validation Set Performance (RandomForest):
Accuracy: 0.7194
Classification Report:
              precision    recall  f1-score   support

       Mouth       0.77      0.77      0.77       119
       Nasal       0.61      0.66      0.64       142
        Skin       0.82      0.83      0.83       158
       Stool       0.68      0.62      0.65       162

    accuracy                           0.72       581
   macro avg       0.72      0.72      0.72       581
weighted avg       0.72      0.72      0.72       581

Validation Set Performance (XGBoost - Tuned):
Accuracy: 0.8692
Classification Report:
              precision    recall  f1-score   support

       Mouth       0.88      0.88      0.88       119
       Nasal       0.87      0.78      0.82       142
        Skin       0.91      0.91      0.91       158
       Stool       0.82      0.90      0.86       162

    accuracy                           0.87       581
   macro avg       0.87      0.87      0.87       581
weighted avg       0.87      0.87      0.87       581

Validation Set Performance (LightGBM - Tuned):
Accuracy: 0.8640
Classification Report:
              precision    recall  f1-score   support

       Mouth       0.87      0.86      0.86       119
       Nasal       0.87      0.79      0.83       142
        Skin       0.89      0.91      0.90       158
       Stool       0.83      0.90      0.86       162

    accuracy                           0.86       581
   macro avg       0.87      0.86      0.86       581
weighted avg       0.86      0.86      0.86       581


### 6. Deployment

- **Gradio Web App:**  
  Interactive UI for predictions and LIME explanations.
  - Enter top feature values and optional extra features.
  - View predicted class, probability chart, and LIME explanation.

This is the link to the gradio(deployment) UI :
https://9e87d9862b20f2f7df.gradio.live

---

## Results

- **Best Model:** Tuned XGBoost (`xgboost_tuned_model.pkl`)
    - Accuracy: 86.9%
    - Most influential features: `avg_read_length`, `gc_content`, `read_count`, `Adj_age`
    - Biological metadata (cytokines, etc.) had lower influence than sequence-derived features.

- **Deployment:**  
  Gradio app for user-friendly predictions and model interpretability.

---

## How to Run

1. **Install Dependencies**
    ```sh
    pip install pandas numpy scikit-learn xgboost lightgbm shap gradio lime biopython tqdm matplotlib seaborn joblib
    ```

2. **Decode MPEG-G Files**
    - Install Docker.
    - Run Genie container to decode `.mgb` files to `.fastq` (see notebook for code)
    Use Docker and Genie as shown in the notebook

3. **Run the Notebook**
    - Open [MPEG_G_Microbiome_Classification.ipynb] in Jupyter or VS Code.
    - Execute cells step-by-step to preprocess data, train models, and deploy the app.

4. **Launch Gradio App**
    - At the end of the notebook, run:
      ```python
      demo.launch(share=True)
      ```
    - Access the web interface for predictions and explanations.

---

## Dependencies

- Python 3.7+
- pandas, numpy, scikit-learn, xgboost, lightgbm, shap, gradio, lime, biopython, tqdm, matplotlib, seaborn, joblib
- Docker (for Genie MPEG-G decoding)

---

## **In'Summary**

### Overview
This project builds a machine learning pipeline to classify human microbiome samples by body site (Mouth, Nasal, Skin, Stool) using 16S rRNA gene profiles stored in MPEG-G format. It leverages metadata and cytokine profiles, and explores federated learning for privacy-aware collaboration.

### Workflow

1. **Data Preparation**
   - Decode `.mgb` (MPEG-G) files to `.fastq.gz` using Docker and Genie.
   - Extract microbial features (read length, GC content, nucleotide distribution) from FASTQ files.
   - Merge sequence features with tabular metadata (Train.csv, Test.csv, Train_Subjects.csv, cytokine_profiles.csv).

2. **Exploratory Data Analysis**
   - Visualize class distributions, sequencing depth, and quality metrics.
   - Dimensionality reduction using t-SNE and PCA.

3. **Modeling**
   - Train and tune RandomForest, XGBoost, and LightGBM classifiers.
   - Evaluate models using accuracy, precision, recall, F1-score, confusion matrix, ROC, and AUC.
   - Use SHAP for feature importance and model explainability.

4. **Deployment**
   - Deploy the best model (XGBoost) using Gradio for interactive web-based predictions and LIME explanations.

### Usage

1. **Environment Setup**
   - Install required Python packages: `Bio`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `gradio`, `lime`, etc.
   - Use Docker to run Genie for MPEG-G decoding.

2. **Run the Pipeline**
   - Follow the notebook steps to preprocess data, train models, and evaluate performance.
   - Save the best model as `xgboost_tuned_model.pkl`.

3. **Web App**
   - Launch the Gradio app for predictions and explanations:
     ```python
     demo.launch(share=True)
     ```

### Files

- `MPEG_G_Microbiome_Classification.ipynb`: Main notebook with all code and analysis.
- `train_processed.csv`, `test_processed.csv`, `train_labels.csv`: Processed data for modeling.
- `xgboost_tuned_model.pkl`: Saved best model.
- `Other Files/`: Contains raw and metadata files.
[
### Results

- **Best Model:** Tuned XGBoost (accuracy: 86.9%)
- **Key Features:** avg_read_length, gc_content, read_count, Adj_age
- **Deployment:** Interactive Gradio app with LIME explanations](https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge)

---

### Citation

Challenge data from [Zindi: MPEG-G Microbiome Classification Challenge](https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge).

---

## Contact

For questions or collaboration or for an open issue, please contact the project owners

- **JUDAH SAMUEL**- Judahsamuel.19@gmail.com
- **DOREEN KAHARE**- 
- **YVONNE KARINGE**- karingeyvonne@gmail.com

---


