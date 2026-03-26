# Early Prediction of ICU Patient Mortality Using Machine Learning

Prepared for  
UMBC Data Science Master’s Degree Capstone  
Instructor: Dr. Chaojie (Jay) Wang  

**Author:**  
Siva Ram Potluri  

**GitHub Repository:**  
https://github.com/SivaRam47925/UMBC-DATA606-Capstone  

**LinkedIn Profile:**  
https://www.linkedin.com/in/siva-ram-potluri-59777b26a  

**PowerPoint Presentation:**  
(To be added in final submission)

**YouTube Video Presentation:**  
(To be added in final submission)

---

## 2. Background

Intensive Care Units (ICUs) provide critical care to patients with life-threatening conditions that require continuous monitoring and advanced medical interventions. Despite advancements in healthcare technology and clinical decision support systems, ICU mortality remains a significant challenge due to patient comorbidities, severity of illness, and delayed recognition of physiological deterioration.

This project focuses on developing machine learning models to predict hospital mortality among ICU patients using structured clinical data. The dataset includes demographic characteristics, ICU admission details, vital signs, laboratory measurements, comorbidities, and severity indicators such as APACHE-related variables.

Early prediction of mortality risk is critical because it can:

- Assist clinicians in identifying high-risk patients  
- Support resource prioritization and ICU management  
- Enable earlier medical intervention  
- Improve patient outcomes through data-driven decision support  

Traditional severity scoring systems such as **APACHE (Acute Physiology and Chronic Health Evaluation)** provide rule-based mortality risk estimates based on worst-case physiological measurements. However, these systems have limitations in capturing complex relationships between variables.

Machine learning models can:

- Capture nonlinear relationships  
- Learn interactions between features  
- Handle high-dimensional data  

This project evaluates whether machine learning techniques can enhance mortality prediction performance beyond traditional scoring approaches.

---

### Research Questions

1. Can machine learning models accurately predict hospital mortality among ICU patients using clinical and demographic data?  
2. Which features contribute most significantly to mortality prediction?  
3. How does model performance differ when using:
   - APACHE-only features  
   - Non-APACHE clinical features  
   - Combined feature sets  
4. Which classification algorithm achieves the best predictive performance for this task?  

---

## 3. Data

### Data Source

The dataset used in this project is obtained from Kaggle:

**WiDS Datathon 2020 ICU Mortality Prediction Dataset**

https://www.kaggle.com/competitions/widsdatathon2020/data

The original dataset contains approximately:

- **91,713 ICU patient records**  
- **186 variables**

Due to GitHub file size limitations, a **20,000-row representative sample** (`training_sample.csv`) is used in this repository for exploratory data analysis.

---

### Data Size

- Full dataset size: ~63 MB  
- Sample dataset size: < 20 MB  

---

### Data Shape

Sample dataset:

- ~20,000 rows  
- 186 columns  

Final modeling dataset:

- Reduced number of rows after sampling  
- Reduced number of columns after feature selection  

Each row represents a single ICU patient encounter.

---

### Time Period

The dataset consists of de-identified ICU encounters collected over multiple years. Clinical measurements primarily reflect the **first 24 hours of ICU admission**.

---

### Unit of Observation

Each row corresponds to a **single ICU patient admission encounter**.

---

### Target Variable

**hospital_death**

Binary outcome variable:

- `0` = Survived  
- `1` = Died  

This variable serves as the label for the machine learning classification models.

---

### Feature Variables

#### Demographics
- age  
- gender  
- ethnicity  
- body mass index (BMI)  

#### ICU Admission Characteristics
- ICU type  
- admission source  
- pre-ICU length of stay  

#### Vital Signs
- heart rate  
- systolic and diastolic blood pressure  
- respiratory rate  
- temperature  
- oxygen saturation  

#### Laboratory Measurements
- creatinine  
- blood urea nitrogen (BUN)  
- glucose  
- lactate  
- electrolytes  
- white blood cell count  

#### Severity Indicators
- APACHE-derived physiological measurements  
- APACHE severity indicators  

#### Comorbidities
- diabetes mellitus  
- cirrhosis  
- immunosuppression  
- AIDS  

---

### Important Data Considerations

The dataset includes both:

- **APACHE features** (severity-based clinical indicators)  
- **Non-APACHE features** (raw physiological measurements)  

In this project:

- Top APACHE features were selected using **Random Forest feature importance**
- These were matched with **available physiologically related non-APACHE variables**

However:

Not all APACHE features have corresponding non-APACHE variables.

As a result:

- The final dataset contains **8 APACHE features**
- And a **smaller set of matched non-APACHE features**

---

### Data Preprocessing

The dataset undergoes the following preprocessing steps:

- Missing value imputation  
- Removal of identifier columns  
- Removal of APACHE prediction outputs (to prevent data leakage)  
- Feature selection using Random Forest  
- Creation of final reduced dataset  

---

### Data Dictionary (Selected Variables)

| Column Name | Data Type | Description | Possible Values |
|------------|----------|-------------|----------------|
| age | Numeric | Patient age at ICU admission | 0–100 |
| gender | Categorical | Patient gender | Male, Female |
| bmi | Numeric | Body Mass Index | Continuous |
| pre_icu_los_days | Numeric | Length of stay before ICU | Continuous |
| heart_rate_apache | Numeric | Heart rate used in APACHE score | Continuous |
| d1_heartrate_max | Numeric | Maximum heart rate (Day 1) | Continuous |
| aids | Binary | AIDS comorbidity indicator | 0 = No, 1 = Yes |
| hospital_death | Binary | Hospital mortality outcome | 0 = Survived, 1 = Died |

---

## 4. Exploratory Data Analysis (EDA)

### Overview

Exploratory Data Analysis was conducted to understand the dataset structure, identify missing values, and explore relationships between features and the target variable.

---

### Missing Value Analysis

The dataset contains significant missing values, which is common in ICU datasets due to selective clinical testing and real-world clinical data collection.

Missing values were handled using **median imputation** during preprocessing.

---

### Target Distribution

The target variable is imbalanced, with more survivors than deaths.

This motivated the use of:

- ROC-AUC  
- precision  
- recall  
- F1-score  

instead of relying only on accuracy.

---

### Feature Distributions

Key variables such as creatinine, BUN, and glucose showed skewed distributions, reflecting variability in patient conditions and severity levels.

---

### Feature vs Target Relationships

Boxplots revealed:

- Higher abnormal values for patients who died  
- Strong separation in kidney-related variables such as creatinine and BUN  

---

### Data Cleaning and Preparation

The following steps were performed:

1. Removed identifier columns  
2. Removed APACHE prediction outputs to avoid data leakage  
3. Selected top APACHE features using Random Forest  
4. Matched APACHE features with available non-APACHE features  
5. Created the final reduced dataset  

---

### Final Dataset

The final dataset includes:

- Selected APACHE features  
- Available matched non-APACHE features  
- Target variable (`hospital_death`)  

---

### Key Findings

- The dataset is high-dimensional and contains missing data  
- The target variable is imbalanced  
- APACHE features are highly predictive  
- Combining APACHE and non-APACHE features improves performance  

---

## 5. Modeling

### Overview

Machine learning models were trained using three feature configurations:

1. APACHE-only  
2. Non-APACHE-only  
3. Combined  

The following models were evaluated:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

---

### Data Preparation

- Missing values handled using median imputation  
- Features standardized using StandardScaler  
- Data split into 80% training and 20% testing sets  
- Stratified sampling used to preserve class distribution  

Additionally, class imbalance was addressed using class weighting in models such as Logistic Regression and Random Forest, ensuring that minority class predictions were not ignored.

---

### Evaluation Metrics

The models were evaluated using:

- ROC-AUC (primary metric)  
- Precision  
- Recall  
- F1-score  

ROC-AUC was selected as the primary evaluation metric because it provides a threshold-independent measure of model performance, which is especially important for imbalanced datasets.

---

### Results

| Feature Set | Model | ROC-AUC | Precision | Recall | F1 Score |
|-------------|-------|--------:|----------:|-------:|---------:|
| APACHE only | Logistic Regression | 0.8055 | 0.2146 | 0.7214 | 0.3308 |
| APACHE only | Random Forest | 0.8438 | 0.5283 | 0.1592 | 0.2447 |
| APACHE only | Gradient Boosting | 0.8576 | 0.6957 | 0.2123 | 0.3253 |
| Non-APACHE only | Logistic Regression | 0.7020 | 0.1627 | 0.5287 | 0.2488 |
| Non-APACHE only | Random Forest | 0.6977 | 0.2776 | 0.0954 | 0.1420 |
| Non-APACHE only | Gradient Boosting | 0.7644 | 0.7161 | 0.0701 | 0.1277 |
| Combined | Logistic Regression | 0.8096 | 0.2165 | 0.7378 | 0.3348 |
| Combined | Random Forest | 0.8510 | 0.7205 | 0.1466 | 0.2436 |
| Combined | Gradient Boosting | **0.8631** | 0.6990 | 0.2186 | 0.3330 |

---

### Interpretation

The modeling results show several important patterns:

- The combined feature set performed best overall  
- Gradient Boosting achieved the highest ROC-AUC (0.8631)  
- Logistic Regression achieved the highest recall  
- Random Forest achieved the highest precision  
- APACHE-only models outperformed non-APACHE-only models  
- Non-APACHE features provided additional value when combined  

---

### Final Model

**Gradient Boosting with Combined Features** was selected as the final model.

---

### Conclusion

- Machine learning models can effectively predict ICU mortality  
- Combining APACHE and non-APACHE features improves performance  
- Gradient Boosting provides the best overall predictive capability  

Overall, this project demonstrates the effectiveness of machine learning in supporting clinical decision-making for ICU mortality prediction.

---

### Limitations

One limitation of this study is the presence of missing data and the lack of direct non-APACHE counterparts for some variables.

---

### Next Steps

- Deploy the model using Streamlit  
- Finalize presentation materials  
- Submit the completed project  
