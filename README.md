# Robust Methods for Detecting LLM Generated Text

## Aim: 
To develop a robust system for acurrately detecting LLM-generated text by applying reliable machine learning techniques and maintaining a reproducible, automated development workflow usign Git and Jenkins. 

## Objectives:
- Design and implement models capable of distinguishing human-written and LLM-generated text with high accuracy. 
- Build a well-versioned codebase using Git to support collaboration, traceability, and controlled experimentation. 
- Set up Jenkins pipelines to automate testing, model evaluation, and deployment. 
- Ensure robustness by validating the system across diverse datasets and different LLM outputs. 

## Preprocessing Steps: 
1. Data Loading & Cleaning:
The dataset is loaded from a CSV file using Pandas. Irrelevant columns such as id and prompt_id are removed, and duplicate rows are checked to ensure data quality. 

2. Exploratory Text Analysis: 
Basic linguistic features are computed from the raw text, including word count, average sentence length, and vocabulary diversity. These features help analyze structural difference between human-written and LLM-generated text. Their distriutions are visualized for insight. 

<img src = "Answer Length Distribution (Words).png" width = "500">

<img src = "Average Sentence Length Distribution.png" width = "500">

<img src = "Vocabulary Diversity Distribution.png" width = "500">

3. Text Cleaning:
Each text is normalised by converting to lowercase, removing digits, punctuation, and special characters. This reduces noise and ensures consistent input for feature extraction. 

4. Feature Extraction (TF-IDF):
Cleaned text is transformed into numerical features using TF-IDF with unigrams and bigrams. This captures word importance and contextual patterns useful for classification. 

5. Class Distribution Analysis:
The class labels (human vs generated) are visualized to identify imblance in the dataset.

<img src = "Class Distribution (Before Resampling).png" width = "500">

6. Resampling:
SMOTE combined with TOmek links is applied to balance the dataset by oversampling the minority class and cleaning overlapping samples, improving model robustness. 

<img src = "Class Distribution (After Resampling).png" width = "500">

7. Train-Test Split:
The balanced data is split into training and testing sets to allow fair evaluation. 

8. Saving Preprocessed Data:
The final processed datasets are saved using joblib for reuse in model training and deployment pipelines. 