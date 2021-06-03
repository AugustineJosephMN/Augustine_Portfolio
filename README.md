# Augustine_Portfolio
## My Data Science Portfolio

# Project 1.
Movie reccomentation system having both colaboratory and contentbased filtering.
Flask is used as front end. The movie-name is given and the most similar or co-related movie from Imdb dataset is choosen.
# Project 2.
Contact-lens.
Identify the eyeball and calculate the area of eyeball using open cv and place lens of different color.
The yolov3 is used to detect the eyeballs from face.
# Project 3.
# Feature-Engineering
## Different types of Missing Data.
### Missing Completely at Random, MCAR:
A variable is missing completely at random (MCAR) if the probability of being missing is the same for all the observations. When data is MCAR, there is absolutely no relationship between the data missing and any other values, observed or missing, within the dataset. In other words, those missing data points are a random subset of the data.
There is nothing systematic going on that makes some data more likely to be missing than other.
### Missing Data Not At Random(MNAR): Systematic missing Values
There is absolutely some relationship between the data missing and any other values, observed or missing, within the dataset.
### Missing At Random(MAR)
#### Mean/ MEdian /Mode imputation
Mean/median imputation has the assumption that the data are missing completely at random(MCAR).
We solve this by replacing the NAN with the most frequent occurance of the variables
##### Advantages
Easy to implement(Robust to outliers)
Faster way to obtain the complete dataset
##### Disadvantages
Change or Distortion in the original variance
Impacts Correlation
#### Random Sample Imputation
Aim: Random sample imputation consists of taking random observation from the dataset and we use this observation to replace the nan values
When should it be used? It assumes that the data are missing completely at random(MCAR)
##### Advantages
Easy To implement
There is less distortion in variance
##### Disadvantage
Every situation randomness wont work
#### Capturing NAN values with a new feature
It works well if the data are not missing completely at random
##### Advantages
Easy to implement
Captures the importance of missing values
##### Disadvantages
Creating Additional Features(Curse of Dimensionality)
#### Arbitrary Value Imputation
This technique was derived from kaggle competition It consists of replacing NAN by an arbitrary value.
##### Advantages
Easy to implement
Captures the importance of missingess if there is one
##### Disadvantages
Distorts the original distribution of the variable
If missingess is not important, it may mask the predictive power of the original variable by distorting its distribution
Hard to decide which value to use
### How To Handle Categroical Missing Values
#### Frequent Category Imputation
##### Compute the frequency with every feature
##### Advantages
Easy To implement
Fater way to implement
##### Disadvantages
Since we are using the more frequent labels, it may use them in an over respresented way, if there are many nan's
It distorts the relation of the most frequent label
#### Adding a variable to capture NAN
### Handle Categorical Features
#### One Hot Encoding
#### Count Or Frequency Encoding
##### Advantages
Easy To Use
Not increasing feature space
##### Disadvantages
It will provide same weiMean Encodingght if the frequencies are same
#### Target Guided Ordinal Encoding
Ordering the labels according to the target
Replace the labels by the joint probability of being 1 or 0
#### Mean Encoding
#### Probability Ratio Encoding
Probability of Survived based on Cabin--- Categorical Feature.
Probability of Not Survived---1-pr(Survived).
pr(Survived)/pr(Not Survived).
Dictonary to map cabin with probability.
replace with the categorical feature.
### Transformation of Features
#### Why Transformation of Features Are Required?
Linear Regression---Gradient Descent ----Global Minima
Algorithms like KNN,K Means,Hierarichal Clustering--- Eucledian Distance
Every Point has some vectors and Directiom
Deep Learning Techniques(Standardization, Scaling) 1.ANN--->GLobal Minima, Gradient 2.CNN 3.RNN
0-255 pixels
### Types Of Transformation
##### Normalization And Standardization
##### Scaling to Minimum And Maximum values
##### Scaling To Median And Quantiles
##### Guassian Transformation Logarithmic Transformation,Reciprocal Trnasformation,Square Root Transformation,Exponential Trnasformation,Box Cox Transformation
### Standardization
Bring all the variables or features to a similar scale. standarisation means centering the variable at zero. z=(x-x_mean)/std
### Min Max Scaling (### CNN)---Deep Learning Techniques
Min Max Scaling scales the values between 0 to 1. X_scaled = (X - X.min / (X.max - X.min).
### Robust Scaler
It is used to scale the feature to median and quantiles Scaling using median and quantiles consists of substracting the median to all the observations, and then dividing by the interquantile difference. The interquantile difference is the difference between the 75th and 25th quantile:
#### IQR = 75th quantile - 25th quantile
#### X_scaled = (X - X.median) / IQR
### BoxCOx Transformation
The Box-Cox transformation is defined as:
T(Y)=(Y exp(λ)−1)/λ
where Y is the response variable and λ is the transformation parameter. λ varies from -5 to 5. In the transformation, all values of λ are considered and the optimal value for a given variable is selected.

# Project 4.
# Feature-Selection-Techniques
## Dropping constant features
Removing the features which have constant features which are actually not important for solving the problem statement.
### Variance Threshold 
Feature selector that removes all low-variance features.This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning.
## Correlation
Removing the features which are highly correlated and also used to check the corelation between Input features and Target output.
## Mutual Information In Classification Problem Statements
MI Estimate mutual information for a discrete target variable.
Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances.
#### Inshort
A quantity called mutual information measures the amount of information one can obtain from one random variable given another.
The mutual information between two random variables X and Y can be stated formally as follows:
I(X ; Y) = H(X) – H(X | Y) Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits.
### Mutual Information In Regression Problem Statements
Estimate mutual information for a continuous target variable.
Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances
Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable.
#### Inshort
A quantity called mutual information measures the amount of information one can obtain from one random variable given another.
The mutual information between two random variables X and Y can be stated formally as follows:
I(X ; Y) = H(X) – H(X | Y) Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits.
## Difference Between Information Gain And Mutual Information
I(X ; Y) = H(X) – H(X | Y) and IG(S, a) = H(S) – H(S | a)
As such, mutual information is sometimes used as a synonym for information gain. Technically, they calculate the same quantity if applied to the same data.
##  Chisquare Test
Compute chi-squared stats between each non-negative feature and class.
This score should be used to evaluate categorical variables in a classification task.
This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative features such as booleans or frequencies (e.g., term counts in document classification), relative to the classes.
Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification. The Chi Square statistic is commonly used for testing relationships between categorical variables.
It compares the observed distribution of the different classes of target Y among the different categories of the feature, against the expected distribution of the target classes, regardless of the feature categories.
