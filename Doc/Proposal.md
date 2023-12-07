# <p align ="center"> Breast Cancer Prediction </p>
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

**Why Does it Matter?**
- For Early Detection and treatment which helps in increasing the survival rates and less aggressive treatment methods can be applied.
- Prevention and the risk reduction and targeted screening programs for high-risk populations might reduce the death rate.

**Research questions?**
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
##	Potential Features/Predictors
- All columns except the target variable may have potential to be utilised as feature columns in machine learning models.
##	Exploratory Data Analysis (EDA)
- **Import Libraries**
  - Imported all thenecessary libraries such as  pandas, matplotlib, seaborn, plotly, and numpy.
  
- **Loading Data_set**
  - loaded the dataset which is in csv format into the jupyter Notebook as df, as data frame.
   
- **Cleaning Data**
	- knowing the datatypes 
	- knowing the shape of data
	- knowing all the column names
	- knowing the stastistical details about our data
	- Checking the null values
	- Deleting unwanted columns(Unnamed: 32 is the empty columns so that is deleted).
	- Checking whether any rows or columns are identical (no two columns or rows have same value)
	- Checking the number of categeorical values (there are 2 categorical values Bengin and melanin)

 
## Feature Engineering
-	Replace the values in the 'diagnosis' column to numerical labels.'Benign' is replaced with 0 and 'Malignant' is replaced with 1.
-	Generate meta data, This purpose is to provide a summary of important information about the columns in a given DataFrame. This summary aids in comprehending the dataset's structure and characteristics.
	### Scaling
	- feature scaling itself does not directly prevent overfitting, it plays a crucial role in maintaining consistency between the training and testing datasets, improving model stability, and influencing the regularization process. These factors collectively contribute to creating models that are less prone to overfitting, leading to better generalization to unseen data.
  ### Normalization
 - Normalization is a statistical and machine learning data preprocessing technique that rescales numerical variables to a standard range. Normalization is the process of transforming a dataset's features to have a similar scale. This is important in various machine learning algorithms because it ensures that no single feature dominates due to its larger scale, thereby preventing biases in the model's learning process.
  -	Normalization methods vary, but one popular approach is Min-Max normalization, which scales the data to a fixed range, typically [0, 1]. The following is how Min-Max normalization works:
	-	1. Find the minimum (min) and maximum (max) values of the feature to be normalized.
	-	2. For each value in the feature, apply the following formula: normalized_value = (original_value - min)/(max - min)
	-	3. Using this formula, the original values are scaled between 0 and 1. If the original value is the minimum value, the normalized value is zero; if it is the maximum value, the normalized value is one.
  ### Standardization
= Standardization is another data preprocessing technique used in machine learning and statistics. In contrast to normalization, standardization rescales features to have the properties of a standard normal distribution with a mean of 0 and a standard deviation of 1. This is also known as z-score normalization or standardization.
- Calculate the mean (μ) and standard deviation (σ) of the feature.
- For each value in the feature, apply the following formula: Standardized_value = (original_value - μ)/σ
- he original values are scaled based on how far they deviate from the mean in this formula. A positive standardized value indicates that the original value is greater than the mean, whereas a negative value indicates that the original value is less than the mean.
	
### Principle Component Analysis
- Principal Component Analysis (PCA) is a technique for reducing dimensionality that is widely used in machine learning and data analysis. Its main goal is to keep as much information as possible while reducing the number of features (or dimensions) in a dataset. The original features are transformed into a new set of uncorrelated features known as principal components by PCA. These principal components are orthogonal to each other and are linear combinations of the original features.
- 1. Variance: PCA seeks to maximize data variance along the new dimensions. High variance indicates that the data points are dispersed and provide useful information.
- 2. Orthogonality: The primary components are orthogonal, which means they are uncorrelated. This ensures that the new features capture a variety of data aspects.
- 3. Eigenvalues and Eigenvectors:PCA entails determining the eigenvalues and eigenvectors of the original data's covariance matrix. Eigenvalues represent the amount of variance explained by each principal component, whereas eigenvectors represent the component's direction in the original feature space.
#### Advantages of PCA:
- 1. Dimensionality Reduction
- 2. Noise Reduction
- 3. Visualization
	- Split the DataFrame into X (features) and y (target) and Initialize PCA with the specified number of components.
	- Fit PCA on the feature matrix (X) and gets the principle components and then Creates a DataFrame to store the principal components.
	- Initializes a dictionary to store top features for each principal component and Loop through each principal component then Select the top 'top_n' features.
	- Next, Creates a list of selected features by combining top features from all principal components and Calculates PCA variance explained and cumulative variance explained.
	- Plotting for both PCA variance and cumulative variance and the Creates a new DataFrame with selected features and the target column.
	<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="ScreePlot.png" alt="Pca Variance" width="700">
    </p>
</div> 
<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Cumulative_variance.png" alt="Pca Variance" width="700">
    </p>
</div>
	- After PCA the	Selected Columns are : ['symmetry_worst', 'concave points_mean', 'texture_worst', 'fractal_dimension_mean', 'fractal_dimension_worst', 'concavity_se', 'texture_se', 'smoothness_mean', 'smoothness_se', 'symmetry_mean']
	
### visualization
- Box Plot for value counts of Malignant and Benign to know the distribution of data
  
  <div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Boxplot.png" alt="value counts of Malignant and Benign" width="700">
    </p>
</div> 

- Histogram of Radius Mean for Benign and Malignant Tumors
  
  <div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Histogram.png" alt="Histogram of Radius Mean for Benign and Malignant Tumors" width="700">
    </p>
</div> 

- Histogram of Smoothness Mean for Benign and Malignant Tumors
  
  <div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="HistogramSmoothness.png" alt="Histogram of smoothness Mean for Benign and Malignant Tumors" width="700">
    </p>
</div> 


- Box Plot of Radius Mean and Texture Mean for Benign and Malignant Tumors

<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Boxplot_Radiusmean&Texturemean.png" alt="Box Plot of Radius Mean and Texture Mean" width="700">
    </p>
</div> 
 
- violin Plot of for first 10 rows to know more about the data

<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="violinplot.png" alt="violin Plot of for first 10 rows" width="500">
    </p>
</div> 

- Heat map for our final data

<div style="display: flex; align-items: center;">
  </div>
  <div style="flex: 1;">
    <p align="center">
    <img src="Heatmap.png" alt="H" width="700">
    </p>
</div> 

## Machine Learning Models

- In the pursuit of accurate and discerning cancer prediction, a comprehensive approach is undertaken through the employment of six distinct machine learning models, each meticulously selected for its unique algorithmic prowess. Logistic Regression, renowned for its simplicity and interpretability, sets the initial foundation, leveraging a linear approach to discern the intricate patterns within our dataset. K-Nearest Neighbor, a method of nuanced proximity analysis, contributes its discriminating capability, promising to unveil the latent structures in the complex web of cancer-related attributes. Meanwhile, the Random Forest Classification algorithm, an ensemble of decision trees, offers a robust and versatile framework capable of capturing intricate relationships and minimizing overfitting.

- Furthermore, the formidable Support Vector Classification, armed with its adeptness in high-dimensional spaces, endeavors to carve optimal hyperplanes, intricately separating the intricacies of cancer manifestation. The Decision Tree Classification, with its intuitive branching logic, aspires to decipher the underlying decision rules embedded within the intricate tapestry of our dataset. Gaussian Naive Bayes, relying on the foundational principles of Bayes' theorem and Gaussian distributions, emerges as an eloquent method for probabilistic classification, particularly adept in scenarios where the independence assumption is reasonably maintained.

-This multifaceted ensemble of algorithms converges toward a singular objective: to meticulously discern the optimal predictive model for our cancer dataset. The comparative analysis, anchored in the pursuit of accuracy, will serve as the crucible wherein these algorithms are scrutinized, revealing their respective merits and intricacies. 

- The dataset, meticulously curated for our cancer prediction endeavor, undergoes a judicious partitioning, allocating 70% of its intricacies to the training domain and reserving the remaining 30% for the rigorous crucible of testing.
 
- In crafting a robust framework for our machine learning endeavors, we judiciously harnessed the formidable arsenal within the sklearn library. Focusing on optimal data utilization, the train_test_split function from sklearn.model_selection Embracing the imperative of feature standardization, the StandardScaler from sklearn.preprocessing was enlisted to ensure a harmonious convergence of features within the machine learning models.

- For our predictive models, a curated selection of algorithms was orchestrated. The venerable LogisticRegression from sklearn.linear_model lent its linear interpretability, while the KNeighborsClassifier from sklearn.neighbors infused an element of nuanced proximity analysis. The ensemble prowess of the RandomForestClassifier from sklearn.ensemble offered a robust approach, navigating the intricate web of cancer-related attributes.

- In the realm of support vector classification, the SVC from sklearn.svm wielded its prowess in high-dimensional spaces, discerning optimal hyperplanes amidst the intricacies of cancer manifestation. The inherently intuitive decision logic of the DecisionTreeClassifier from sklearn.tree was harnessed to decipher underlying decision rules. Meanwhile, the probabilistic elegance of the GaussianNB from sklearn.naive_bayes was employed for nuanced probabilistic classification, embracing the Gaussian assumption.

- In the aftermath of model predictions, a meticulous scrutiny of performance was facilitated by the judicious employment of metrics gleaned from sklearn.metrics. The triumvirate of accuracy_score, classification_report, and confusion_matrix emerged as indispensable companions, offering insights into the accuracy scores, detailed classification reports, and the intricate nuances encapsulated within the confusion matrix.

- I primarily use my personal laptop as my development environment. Additionally, for collaborative projects and when I need access to more computational resources, I leverage Google Colab. I also utilize GitHub CodeSpaces for its seamless integration with version control and collaborative coding. 

- To measure and compare the performance of the models, I will employ a combination of relevant evaluation metrics tailored to the nature of the problem and the characteristics of our dataset. These metrics will include accuracy, precision, recall, F1 score, and the confusion matrix.

## Application of the Trained Model

- Embarking on the forefront of healthcare technology, we have meticulously developed a user-friendly webpage leveraging the cutting-edge capabilities of the Streamlit platform. This dynamic interface serves as a gateway for precise breast cancer prediction, harnessing the formidable prowess of the Random Forest Classifier algorithm. By inputting essential details such as radius, texture, smoothness, and compactness, users can promptly receive real-time predictions distinguishing between benign and malignant conditions.

- Our webpage transcends mere diagnostic capabilities; it extends a compassionate hand by providing personalized guidance. In addition to the predictive insights, users gain access to a wealth of knowledge. The platform suggests tailored treatment methods, empowering individuals with informed choices on their journey to recovery. Furthermore, a curated selection of suggested foods is presented, strategically designed to bolster resilience and complement the specific needs dictated by the nature of the cancer.

- This holistic approach not only bridges the diagnostic gap but also cultivates a supportive ecosystem, underscoring our commitment to empowering individuals with the tools and knowledge necessary to navigate the complexities of breast cancer. Our Streamlit-powered webpage stands as a testament to the seamless fusion of advanced technology, medical insights, and empathetic care, forging a pathway toward informed decisions and resilient health outcomes.

## Conclusion

- In our comprehensive project, we harnessed the predictive capabilities of six state-of-the-art machine learning algorithms, including Logistic Regression, K-Nearest Neighbor, Random Forest Classification, Support Vector Classification, Decision Tree Classification, and Gaussian Naive Bayes. These algorithms collectively underwent rigorous evaluation for breast cancer prediction within our Streamlit-powered webpage. Users can seamlessly input specific details, and our system dynamically utilizes these algorithms to provide real-time predictions, distinguishing between benign and malignant conditions.

- The integration of multiple machine learning models underscores our commitment to achieving the highest predictive accuracy and robustness. Through a meticulous evaluation process, we have ensured that our predictive tool not only excels in diagnostic precision but also goes beyond by offering personalized treatment suggestions and dietary recommendations. 

## Limitations

- While our project represents a significant advancement in breast cancer prediction and personalized guidance, it's crucial to acknowledge its limitations. Firstly, the predictive accuracy of our models is contingent on the quality and representativeness of the training data. Any biases or inadequacies in the dataset could impact the model's performance and generalizability.

- our system relies on the assumption of feature independence, In real-world scenarios, features may exhibit complex interdependencies that this assumption does not capture.

- our platform's predictive capabilities are contingent on the completeness and accuracy of the input data provided by users. Incomplete or inaccurate information may lead to suboptimal predictions.

- our current system focuses exclusively on breast cancer prediction and provides general treatment and dietary recommendations. It doesn't account for individual variations in patient history, genetic factors, or other co-existing health conditions, limiting the personalization of the guidance offered.

## Leasons Learned

- we've learned the paramount importance of meticulous data preprocessing. The quality and representativeness of our training data profoundly impact the predictive accuracy of our models. We recognize the need for thorough data exploration, cleansing, and validation to mitigate biases and enhance the robustness of our system.

- Each algorithm carries its strengths and limitations, and our iterative experimentation process has underscored the significance of aligning model choices with the unique characteristics of our dataset.  

- We have learned to embrace an agile development mindset, allowing for flexibility and adaptability. As we encountered unforeseen challenges or opportunities for improvement, our ability to pivot and iterate has been a key driver of progress.

## Future Research

-  Future research endeavors could delve deeper into refining the personalization aspect of our predictive models. Incorporating individual patient histories, genetic data, and lifestyle factors could further tailor treatment and dietary recommendations, ensuring a more precise and patient-centric approach.

- Exploring the integration of advanced technologies, such as deep learning or neural networks, could unlock new dimensions in predictive accuracy and feature representation.

-  The incorporation of real-time data streams, such as continuous monitoring of health metrics or wearable device data, could enhance the dynamic nature of our predictive models.

- our current project focuses on breast cancer, there is ample room for extending the predictive models to other types of cancer. 

- Future research should also address the ethical implications and regulatory considerations associated with deploying predictive healthcare models. 

## References

- https://statisticsbyjim.com/basics/principal-component-analysis/

- https://www.geeksforgeeks.org/understanding-logistic-regression/

- https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- https://scikit-learn.org/stable/modules/svm.html

- https://www.coursera.org/articles/decision-tree-machine-learning
