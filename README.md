# Customer Propensity to Buy Dataset
## Data Science and Machine learning Project

## Unsupervised Learning
### Introduction and Objectives
#### The primary objective of this analysis is to perform customer segmentation through clustering methods, specifically utilizing K-Means and Gaussian Mixture Models (GMM). The dataset used for this analysis comprises 607,056 observations and includes 25 features capturing detailed customer interaction behaviors, such as clicks, basket activities, and device preferences. 

### Data Preparation and Feature Engineering
#### The dataset used for clustering analysis consisted of over 607,000 observations and 25 behavioral features, combining customer interactions from both training and testing samples. These features covered a wide range of digital behaviors, such as basket interactions, promotional clicks, account and delivery actions, and device usage. After merging the datasets, a quality check confirmed that there were no missing values across any columns, indicating a complete and reliable dataset for modeling purposes.
<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/acc52ea3-fd88-4d69-b167-179bc3ddf71a" />

<img width="2608" height="2090" alt="image" src="https://github.com/user-attachments/assets/298bb68f-586e-4a37-a920-2bb0ceeab714" />

#### To begin exploratory analysis, numeric and binary features were identified. The binary features, consisting of actions encoded as 0s and 1s, were visualized to better understand the customer behavior distribution across key events. As shown in Distribution of Features (Figure 1), most users did not perform high-intent actions such as basket_add_detail or ordered, highlighting significant behavioral imbalance and sparsity in conversion-related activities. 
#### Next, low-variance features were removed to enhance the quality of correlation analysis. A correlation heatmap (Figure 2) was generated to identify multicollinearity among variables. The heatmap revealed several strong positive correlations between basket-related actions and between sign-in, checkout, and order completion events. These findings motivated the next step: feature engineering through behavioral score aggregation.
#### Three composite behavioral scores were created—engagement_score, intent_score, and conversion_score—each aggregating related features into a single dimension to summarize customer engagement, shopping intent, and conversion behavior. Redundant features used to compute these scores were dropped, along with optional device-type indicators, which were deemed less relevant for the clustering task.
