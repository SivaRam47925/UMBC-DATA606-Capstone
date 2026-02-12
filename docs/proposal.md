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

# 2. Background

Intensive Care Units (ICUs) provide critical care to patients with life-threatening conditions that require continuous monitoring and advanced medical interventions. Despite advancements in healthcare technology and clinical decision support systems, ICU mortality remains a significant challenge due to patient comorbidities, severity of illness, and delayed recognition of physiological deterioration.

This project focuses on developing machine learning models to predict hospital mortality among ICU patients using structured clinical data. The dataset includes demographic characteristics, ICU admission details, vital signs, laboratory measurements, comorbidities, and severity indicators such as APACHE-related variables.

Early prediction of mortality risk is critical because it can:

- Assist clinicians in identifying high-risk patients  
- Support resource prioritization and ICU management  
- Enable earlier medical intervention  
- Improve patient outcomes through data-driven decision support  

Traditional severity scoring systems like APACHE provide rule-based mortality risk estimates. However, machine learning models can identify nonlinear relationships and complex interactions among high-dimensional clinical variables. This project evaluates whether machine learning techniques can enhance mortality prediction performance beyond traditional scoring approaches.

---

## Research Questions

1. Can machine learning models accurately predict hospital mortality among ICU patients using clinical and demographic data?  
2. Which features contribute most significantly to mortality prediction?  
3. How does model performance differ when APACHE-related features are included versus excluded?  
4. Which classification algorithm achieves the best predictive performance for this task?  

---

# 3. Data

## Data Source

The dataset used in this project is obtained from Kaggle:

**WiDS Datathon 2020 ICU Mortality Prediction Dataset**

The original dataset contains approximately **91,713 ICU patient records** and **186 variables**.

Due to GitHub file size limitations, a **20,000-row representative sample** (`training_sample.csv`) is used in this repository for exploratory data analysis.

---

## Data Size

- Full dataset size: ~63 MB  
- Sample dataset size: < 20 MB  

---

## Data Shape

Sample dataset:

- ~20,000 rows  
- 186 columns  

Each row represents a single ICU patient encounter.

---

## Time Period

The dataset consists of de-identified ICU encounters collected over multiple years. Exact calendar years are not explicitly provided. Clinical measurements primarily reflect the first 24 hours of ICU admission.

---

## Unit of Observation

Each row corresponds to a single ICU patient admission encounter.

---

## Target Variable

**hospital_death**

Binary outcome variable:

- `0` = Survived  
- `1` = Died  

This variable serves as the label for the machine learning classification models.

---

## Feature Variables

Potential predictor variables include:

### Demographics
- age  
- gender  
- ethnicity  
- body mass index (BMI)  

### ICU Admission Characteristics
- ICU type  
- admission source  
- pre-ICU length of stay  

### Vital Signs
- heart rate  
- systolic and diastolic blood pressure  
- respiratory rate  
- temperature  
- oxygen saturation  

### Laboratory Measurements
- creatinine  
- blood urea nitrogen (BUN)  
- glucose  
- lactate  
- electrolytes  
- white blood cell count  

### Severity Indicators
- APACHE-derived physiological measurements  
- APACHE predicted mortality probability  

### Comorbidities
- diabetes mellitus  
- cirrhosis  
- immunosuppression  
- AIDS  

These variables will undergo preprocessing steps including:

- Missing value imputation  
- Encoding of categorical variables  
- Feature selection  
- Model training and evaluation using appropriate classification metrics (e.g., ROC-AUC, recall, F1-score)

---

## Data Dictionary (Selected Variables)

| Column Name | Data Type | Description | Possible Values |
|-------------|-----------|------------|----------------|
| age | Numeric | Patient age at ICU admission | 0–100 |
| gender | Categorical | Patient gender | Male, Female |
| bmi | Numeric | Body Mass Index | Continuous |
| pre_icu_los_days | Numeric | Length of stay before ICU | Continuous |
| heart_rate_apache | Numeric | Heart rate used in APACHE score | Continuous |
| d1_heartrate_max | Numeric | Maximum heart rate (Day 1) | Continuous |
| aids | Binary | AIDS comorbidity indicator | 0 = No, 1 = Yes |
| hospital_death | Binary | Hospital mortality outcome | 0 = Survived, 1 = Died |

Due to the large number of variables (186), only representative features are shown above. A complete data dictionary will be included in the final report.

---
