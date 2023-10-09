# <p align ="center"> Breast Cancer Predection </p>
##  Title and Author
- Project Title - **Breast Cancer Predection**
- Prepared for UMBC Data Science Master Degree Capstone by **Dr Chaojie (Jay) Wang**
- Author - **Kurre Sai Chand**
- Author's GitHub profile - https://github.com/saichandkurre
- Author's LinkedIn progile - www.linkedin.com/in/sai-chand-kurre
- PowerPoint presentation file -
- YouTube video -
##  Background
<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="BreastCancer_Awareness.jpg" alt="Breast Cancer Awareness" width="400">
    </p>
</div> 

**Breast Cancer**
- Breast cancer is the most common cancer in women, accounting for one in every four cancer diagnoses in the United States and killing over 40,000 people each year. It is also, after lung cancer, the second leading cause of cancer death in women. Early detection of breast cancer is critical and can improve survival chances.
- Breast cancer is the most common cancer, and it is one of the most commonly reported skin cancer types in recent years. The infographics above provide a clear picture of this cancer and its impact on the modern world. Breast Cancer is a current hot-buttom issue in the world of healthcare. The main reason could be our modern, sedentary lifestyle. This type of cancer can affect both men and women, but according to scientific research, women are twice as likely as men to develop breast cancer. That is why it is critical to detect cancer in its early stages.
- Breast Cancer arises in the lining cells (epithelium) of the ducts (85%) or lobules (15%) in the glandular tissue of the breast. Initially, the cancerous growth is confined to the duct or lobule (“in situ”) where it generally causes no symptoms and has minimal potential for spread (metastasis).
- Cancer, as most of us know, is defined as uncontrolled cell growth in a specific area. An expert will classify the cells as malignant or benign based on an imaging procedure known as Fine Needle Aspiration. But how can machine learning be used to diagnose breast cancer? That is the burning question of the hour. Cell characteristics are measured from Fine Needle aspiration images using image processing techniques or manual measurements, and these characteristics are used to classify the cells as benign or cancerous.
<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="CancerFacts.jpg" alt="Facts About Cancer" width="400">
    </p>
</div> 

- Can we develop a predictive model that accurately classifies tumors as malignant (M) or benign (B) based on the given features (radius_mean, texture_mean, perimeter_mean, etc.)?
- Are there any strong correlations between pairs of features?

## Data
- **Data Source** : The data is found from UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Data Size** : size of our data is 128 KB
- **Data Shape** : Our Data Set consists of 570 rows and 33 columns
- Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- **Data Dictionary** :
- 1 ID number
- 2 Diagnosis (M = malignant, B = benign)
- 3-33) Ten real-valued features are computed for each cell nucleus:
    - a) radius (mean of distances from center to points on the perimeter)
    - b) texture (standard deviation of gray-scale values)
    -  c) perimeter
    - d) area
    - e) smoothness (local variation in radius lengths)
    - f) compactness (perimeter^2 / area - 1.0)
    - g) concavity (severity of concave portions of the contour)
    - h) concave points (number of concave portions of the contour)
    - i) symmetry
    - j) fractal dimension
- The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 columns.
- id is of datatype int64, diagnosis is of data type object and the rest of other 30 parameters are of Float64.
- **Target For ML Model** : Diagnosis
## 4. Exploratory Data Analysis (EDA)
 ### 4.1 Data Cleaning 
  #### 4.1.1 Dropping the unwanted columns
- In the DataFrame we have 1 columns they are Unnamed: 32.  
- Unnamed: 32 is the unwanted columns so that is deleted.

#### 4.1.2 Checking & removing the duplicate rows from DataFrame
- We have checked whether any 2 columns have the same value but there are none. Checking the duplicate rows in the DataFrame. It is observed that there are no multiple rows

 ### 4.2 visualization
- Box Plot for value counts of Malignant and Benign to know the distribution of data
  
  <div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Boxplot.png" alt="value counts of Malignant and Benign" width="200">
    </p>
</div> 

- Histogram of Radius Mean for Benign and Malignant Tumors
  
  <div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Histogram.png" alt="Histogram of Radius Mean for Benign and Malignant Tumors" width="200">
    </p>
</div> 

- Box Plot of Radius Mean and Texture Mean for Benign and Malignant Tumors

<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Boxplot_Radiusmean&Texturemean.png" alt="Box Plot of Radius Mean and Texture Mean" width="200">
    </p>
</div> 
 
- violin Plot of for first 10 rows to know more about the data

<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="violinplot.png" alt="violin Plot of for first 10 rows" width="200">
    </p>
</div> 
