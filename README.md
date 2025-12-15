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

### Clustering Methodology

#### To segment customers based on behavioral scores, clustering was performed using both K-Means and Gaussian Mixture Models (GMM). Prior to clustering, all engineered features were standardized to eliminate the influence of scale discrepancies, ensuring fair treatment of each variable during distance-based computations.
#### For K-Means, the optimal number of clusters was determined using two evaluation techniques: the Elbow Method and the Silhouette Method. The Elbow Method (Figure 3) evaluated the within-cluster sum of squares (WSS) for cluster counts ranging from 2 to 10. A distinct 'elbow' was observed at the point where adding more clusters yielded diminishing returns in reducing WSS, indicating an appropriate cluster number.
<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/b3aa910c-2e89-43dd-bbfd-5e83e348c4a4" />

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/56591eae-deb7-48df-b43f-90232d618fc6" />

#### The Silhouette Method (Figure 4) validated optimal cluster selection by measuring cohesion and separation. Unlike K-Means, Gaussian Mixture Models (GMM) flexibly model clusters of varying shapes, sizes, and orientations using probabilistic labels, effectively capturing overlapping or non-spherical clusters.

### Results and Visualizations

#### Following the application of K-Means clustering with the optimal number of clusters set to 8, the model segmented users into distinct behavioral groups. K-Means Cluster Distribution (Figure 5) revealed a heavily skewed structure, with Cluster 3 representing the majority of users—categorized as passive or silent users—while other clusters varied in size, capturing more specific behavioral profiles.

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/8dfa03a7-9169-4bd5-99a2-fffc00defbce" />

#### The clustering quality was quantitatively evaluated using the average silhouette score, which was 0.843 , indicating well-separated and cohesive clusters. Further insights into the structure and spread of these segments are visible in the 3D Cluster Plot (Figure 6) across the three behavioral scores: engagement, intent, and conversion.

<img width="838" height="622" alt="image" src="https://github.com/user-attachments/assets/ce047f5a-a0f9-468c-92cb-5df7ce8e87ff" />

#### To understand the composition of each segment, a dual-axis visualization was created to display the number of customers per cluster alongside their average feature scores. As shown in Cluster Distribution with Average Feature Values (Figure 7), each cluster exhibited a unique behavioral profile, ranging from “Low-Engagement Browsers” and “Window Shoppers” to “Committed Buyers” and “Indecisive Visitors.

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/94966982-5255-4bef-98cf-d9c9fe94f39e" />

#### The GMM model was then applied to the same scaled data. The number of optimal clusters and model shape parameters were determined using the Bayesian Information Criterion (BIC), as visualized in BIC for GMM Model Selection (Figure 8). A GMM model with 9 clusters was selected based on this criterion. The 3D visualization of GMM clusters (Figure 9) showed more fluid boundaries between groups, with each customer assigned to a cluster probabilistically.

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/7bffe143-55c4-4974-857c-9f1a3ffcecf7" />

<img width="1564" height="898" alt="image" src="https://github.com/user-attachments/assets/d05f1d9c-650a-4cb8-b2ea-388c0d27d03a" />

#### The average silhouette score for the GMM approach was 0.891, slightly outperforming K-Means in terms of cohesion and separation. 

#### Finally, the combined visualization in GMM Cluster Distribution with Average Feature Values (Figure10) effectively captured the size and behavioral characteristics of each segment. Together, the results from both clustering methods offered a robust foundation for personalized marketing strategies and customer engagement planning.

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/3fa1dcef-3ab9-4af6-9f3f-cc10056416a1" />

### Insights and Business Implications

#### The clustering analysis yielded actionable customer segments with distinct behavioral profiles, enabling personalized marketing strategies.
#### For instance, Low-Engagement Browsers and Window Shoppers can be targeted with awareness campaigns or personalized offers to encourage deeper engagement. Majority Silent Users, being the largest group, may be better served with reactivation efforts or deprioritized in resource-intensive campaigns.
#### Committed Buyers and High-Value Customers should be prioritized for loyalty programs and upselling, while Indecisive Visitors may benefit from urgency tactics like limited-time discounts.
#### GMM clusters, including Mass Market, Potential Buyers, and Loyal Customers, provide additional nuance for strategic targeting. Probabilistic assignment allows marketers to tailor actions based on confidence levels in customer classification















