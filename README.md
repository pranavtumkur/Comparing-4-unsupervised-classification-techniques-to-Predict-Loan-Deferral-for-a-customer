# Predicting Loan Deferral using Classification ML Models

The aim of this project is to compare the evaluation accuracies of 4 different unsipervised classification techniques, viz.:

1. K Nearest Neigbours (KNN)
2. Decisison Trees
3. Support Vector Machines
4. Logistic Regression

![232](https://user-images.githubusercontent.com/65482013/84423831-b604bb00-ac3c-11ea-8133-404a72e2a6e7.jpg)


## The Dataset
This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

We download the dataset using the wget method as an OS Command, from the Amazon S3 Cloud Storage: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv

## Pre-processing and Cleaning

First, we look at the day of the week people get the loan. We observe that people who get the loan at the end of the week dont pay it off, so we use Feature binarization to set a threshold values less then day 4. Next we convert Categorical features to numerical values; we convert records with sex as male to 0 and female to 1. We convert the education field to numerical values using One Hot Encoding. Finally, we normalize the data using Data Standardization which gives data zero mean and unit variance.

## Analyzing each of the ML models:

#### K Nearest Neighbor(KNN)
First, we will find the best k to build the model with the best accuracy. For that, we will split our train_loan.csv into train and test to find the best __k__. We calculate the accuracy score of KNNs for different Ks and plot model accuracy for different number of Neighbors.

#### Decision Tree
For visualization of the Decision Tree we import two libraries: graphviz and pydotplus. We then construct an image of the actual decision tree in terms as a flowchart using all features (independent variables).

#### Support Vector Machine
We iterate for different kernels like Linear, Polynomial, Radial basis function (RBF) and Sigmoid to arrive at the one which best fits the data curve.

#### Logistic Regression
For this model, we use a User Defined Fun-ction to plot the coloured confusion matrix (without normalization so that we can analyze the distribution of True Positives/ False Positives/ True Negatives/ False Negatives 

## Model Evaluation using Test set
We finally evaluate the accuracy of each of these models against a separate test set (on which the model was not trained), to check the out-of-fold prediction capability of the models. We do this using by calculating 2 indicative parameters: __Jaccard Index__ & __F1 score__. Additionally we also plot the precision and recall values (True Positives/ False Positives/ True Negatives/ False Negatives) for each model to get better visibility on the cases where it is failing. For the Logistic Regression model, we use an adiditional parameter called the Log Loss.

# Conclusion
Lastly, we report the accuracy of the built models using different evaluation metrics:

| Algorithm          | Jaccard | F1-score | LogLoss |
|--------------------|---------|----------|---------|
| KNN                | 0.72    | 0.72     | NA      |
| Decision Tree      | 0.77    | 0.78     | NA      |
| SVM                | 0.72    | 0.72     | NA      |
| LogisticRegression | 0.75    | 0.76     | 0.47    |

In general we see high sensitivity/recall values (F1 scores) in our models for the head of 'PAIDOFF'. This seems to be because of a higher number of entries which have been paid off in the source dataset. Therefore to predict default, we need more test data. The same can be verified via the confusion matrix which shows higher number of false negatives (i.e for 'Collection' loan repayment status)

Though we see that the Decision Tree Algorithm provides the best output, other algorithms are not far behind. These models can be improved further using different solvers and higher number of iterations to make them slightly better.
