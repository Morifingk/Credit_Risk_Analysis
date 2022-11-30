# Credit_Risk_Analysis
Supervised Machine Learning - Credit Risk Analysis

Because good loans outnumber risky loans, credit risk is an inherently unbalanced classification problem. As a result, it is neccesary to use a variety of techniques to train and evaluate models with unbalanced classes. For this purpose we will build and evaluate models with resampling using the imbalanced-learn and scikit-learn libraries.

I will also use the RandomOverSampler and SMOTE algorithms to oversample the credit card credit dataset from LendingClub, a peer-to-peer lending services company, and the ClusterCentroids algorithm to undersample the data. The SMOTEENN algorithm will then be used to perform a combinatorial approach of over- and undersampling. Then, to predict credit risk, I will compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier. After that, I will assess the performance of these models and conclude whether they should be used to predict credit risk.

Overview of the analysis: Explain the purpose of this analysis.

The objective of this analysis is to apply six different unbalanced machine learning models to the same dataset and compare the results obtained. The dataset used contains the credit applications for the first quarter of 2019, obtained from a personal credit company called LendingClub. The original data file contains information on 115,675 applications.

These are the six unbalanced machine learning models that will be used in this analysis:

#	Type of Algorithm	Name of Algorithm
1	Oversampling Algorithm	Naive Random Oversampling
2	Oversampling Algorithm	SMOTE Oversampling
3	Undersampling Algorithm	Cluster Centroid
4	Combination (Over and Under) Sampling Algorithm	SMOTEENN
5	Ensemble Learner	Balanced Random Forest Classifier
6	Ensemble Learner	Easy Ensemble AdaBoost Classifier
Files used for the study

The study of the data is done with two Python code files, using Jupyter Notebooks:

Code File #1: credit_risk_resampling.ipynb

Code File #2: credit_risk_ensemble.ipynb

In the dataset there is a column named loan_status which contains the following values: Charged Off, Current, Fully Paid, In Grace Period, Issued, Late (16-30 days), Late (31-120 days).

For this analysis, the applications were classified into two groups:

The low risk group, composed of the applications marked as Current

The high risk group which is composed of applications marked with these values (In Grace Period, Late (16-30 days), Late (31-120 days)).

Applications marked as Issued, Fully Paid and Charged Off were ignored.

With these operations the dataset was reduced from 115,675 applications to 68,817.

Once the dataset was cleaned and organized for our analysis, the data was split into Training and Testing and the analysis could start.

New Functions Learned for creating Training and Testing datasets:

get.dummies(X)

Since the Machine Learning models only work with numbers we need to convert all the cells with low risk and high risk to numbers. The get_dummies() function is used to convert categorical variable into dummy/indicator variables. In this case all the 'low risk' were changed to 0 and the high risk to 1

train_test_split(X, y, random_state=1)

This is a function from the sklearn Python library. The train_test_split function is for splitting a single dataset for two different purposes: training and testing. The training subset is for building the model. The testing subset is for using the model on unknown data to evaluate the performance of the model.

After the Training and Testing groups are created, we started testing the Machine Learning Algorithms.

1.1 Oversampling Algorithms

Over sampling and under sampling are techniques used in data mining and data analytics to modify unequal data classes to create balanced data sets. Over sampling and under sampling are also known as resampling.

When one class of data is the underrepresented minority class in the data sample (in our case this would be the high risk category), over sampling techniques maybe used to duplicate these results for a more balanced amount of positive results in training. Over sampling is used when the amount of data collected is insufficient. A couple of popular over sampling technique are the Naive Random Oversampling and SMOTE (Synthetic Minority Over-sampling Technique), which creates synthetic samples by randomly sampling the characteristics from occurrences in the minority class.

Naive Random Oversampling and SMOTE Oversampling

New Functions Learned for Oversampling datasets:

RandomOverSampler(random_state=1)

Random oversampling can be implemented using the RandomOverSampler class. The class can be defined and takes a sampling_strategy argument that can be set to “minority” to automatically balance the minority class with majority class or classes. This means that if the majority class had 1,000 examples and the minority class had 100, this strategy would oversampling the minority class so that it has 1,000 examples.

fit_resample(X,y)

The fit_resample method is used in conjunction with the RandomOverSampler to resample the data and targets into a dictionary with a key-value pair of data_resampled and targets_resampled.

LogisticRegression(solver='lbfgs', random_state=1)

This is part of the Python library sklearn. The solver argument is set to lbfgs, which is the default setting. The random_state is specified so that anyone will be able to reproduce the same results as they run the code.

model.predict(X_test) and balance_accuracy_score(y_test,y_pred)

This is the model used to create predictions and generate the accuracy score of the model

confusion_matrix(y_test, y_pred)

This is another function from the sklearn Python Library. It creates a 3x3 table (or a matrix), where all the information about false positives and false negatives is displayed. It is used to evaluate the precision of the model.

classification_report_imbalanced

This function comes from the imblearn Python Library. It builds a classification report based on metrics used with imbalanced datasets. Specific metrics have been proposed to evaluate the classification performed on an imbalanced dataset. This report compiles the state-of-the-art metrics:

Accuracy Score: accuracy can be described as the ratio of true predictions (positive and negative) to the sum of the total number of positive and negative samples. Accuracy score = number of correct prediction / total number of predictions

Precision: Is the measure of how reliable a positive classification is. A low precision is indicative of a large number of false positives. Equation: Precision = TP/(TP + FP)

Recall: Is the ability of the classifier to find all the positive samples. A low recall is indicative of a large number of false negatives. Equation: Recall = TP/(TP + FN)

Specificity: Specificity is the recall of negative values. It answers the question “Of all of my negative predictions, what proportion of them are correct?”. This may be important in situations where examining the relative proportion of false positives is necessary.

F1 Score: Is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0 (3). Equation: F1 score = 2(Precision * Sensitivity)/(Precision + Sensitivity)

Geometric Mean: A less common metric that is somewhat analogous to the F1 score is the G-Mean. This is often cast in two different formulations, the first being the precision-recall g-mean, and the second being the sensitivity-specificity g-mean. They can be used in a similar manner to the F1 score in terms of analyzing algorithmic performance.

IBA: index balanced accuracy of the geometric mean: This is a metric used for evaluating learning processes in two-class imbalanced domains. The method combines an unbiased index of its overall accuracy and a measure about how dominant is the class with the highest individual accuracy rate.

SUP: Support:The number of occurrences of each label in y_true.

1.2 Undersampling Algorithms

Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.

Cluster Centroid Algorithms

New Functions Learned for Undersampling datasets:

ClusterCentroids(random_state=1)

Random oversampling can be implemented using the RandomOverSampler class. The class can be defined and takes a sampling_strategy argument that can be set to “minority” to automatically balance the minority class with majority class or classes. This means that if the majority class had 1,000 examples and the minority class had 100, this strategy would oversampling the minority class so that it has 1,000 examples.

fit_resample(X,y)

The fit_resample method is used in conjunction with the RandomOverSampler to resample the data and targets into a dictionary with a key-value pair of data_resampled and targets_resampled.

1.3 Combination (Over and Under) Sampling Algorithms

Oversampling methods duplicate or create new synthetic examples in the minority class, whereas undersampling methods delete or merge examples in the majority class. Both types of resampling can be effective when used in isolation, although can be more effective when both types of methods are used together.

SMOTEENN algorithm

Combine SMOTE with Edited Nearest Neighbor (ENN) using Python to balance the dataset

New Functions Learned for SMOTEENN datasets:

SMOTEENN(random_state=0)

The SMOTE method can generate noisy samples by interpolating new points between marginal outliers and inliers. This issue can be solved by cleaning the space resulting from over-sampling. In this regard the Edited Nearest-Neighbours (ENN) is one of the cleaning methods available.

1.4 Ensemble Learners

Ensemble learning is a general meta approach to machine learning that seeks better predictive performance by combining the predictions from multiple models.

Balanced Random Forest Classifier

Random forest algorithm is one of the most popular and potent supervised machine learning algorithms capable of performing both classification and regression tasks.

New Functions Learned for Balanced Random Forest Classifier datasets:

BalancedRandomForestClassifier(n_estimators=100, random_state=1)

This algorithm creates a forest with several decision trees. The more trees in the forest, the more robust the prediction is and hence higher accuracy to model multiple decision trees.



Easy Ensemble AdaBoost classifier

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

New Functions Learned for Easy Ensemble AdaBoost classifier datasets:

EasyEnsembleClassifier(n_estimators=100, random_state=1)

AdaBoost also called Adaptive Boosting is a technique in Machine Learning used as an Ensemble Method. The most common algorithm used with AdaBoost is decision trees with one level that means with Decision trees with only 1 split. These trees are also called Decision Stumps.



2️ Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

Algorithm Name	Accuracy	Precision Score 
Low Risk	Precision Score 
High Risk	Recall Score 
Low Risk	Recall Score 
High Risk
Naive Random Oversampling	64.3%	99.7%	1.02%	62.3%	66.3%
Smote Oversampling	65.3%	99.7%	1.2%	69.2%	61.3%
Cluster Centroid Undersampling	54.4%	99.5%	0.67%	39.5%	69.3%
SMOTEENN Over and Under Sampler	64.5%	99.7%	0.98%	56.7%	72.2%
Balanced Random Forest Classifier	78.9%	99.8%	3.2%	87.4%	70.3%
Easy Ensemble AdaBoost Classifier	91.03%	99.9%	7.4%	93.5%	88.5%
Screenshots

⚠️ NOTE: Please click on any image to zoom
🔽 Naive Random Oversampling 🔽
  
🔽 Smote Oversampling 🔽
  
🔽 Cluster Centroid Undersampling 🔽
  
🔽 SMOTEENN Over and Under Sampler 🔽
  
🔽 Balanced Random Forest Classifier 🔽
  
🔽 Easy Ensemble AdaBoost Classifier 🔽
  
3️⃣ Recap: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.



The group of Resampling Models has the lowest scores for all the calculated parameters. This makes it hard to trust them for bank operations.

For their part, the Ensemble Models have come up with numbers that look good.

The High Risk results must be used to choose the best machine learning model for this dataset. This is because the bank needs to avoid bad loans as much as possible to make money. In this way, the Easy Ensemble AdaBoost Classifier model is the clear winner, since it got the highest Accuracy score of 91.03 % and the highest Precision Score for High Risk of 7.4 %. But this model also got the highest score for High Risk in the Recall Score category, which is something to think about.

There are 115,675 applications in the original dataset, but only 526 are High Risk (either Late (16-30 days), Late (31-120 days) or In Grace Period). This is 0.45 % of the total, while 99.5 % of all loan applications are for Current loans. The original data is heavily imbalanced or skewed toward Low Risk, which could affect the results of the Machine Learning algorithms by making clusters from too few real High Risk applications. So, the models need to be tested on a much bigger set of data before either of them can be recommended.

4️ References

Module 17: Supervised Machine Learning, https://courses.bootcampspot.com/courses/1145/pages/17-dot-0-1-introduction-to-machine-learning, ©️ 2020-2021 Trilogy Education Services, Web 21 May 2022.

Stack Overflow: scikit learn output metrics.classification_report into CSV/tab-delimited format, https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format

GitHub: [BUG] 'BalancedBaggingClassifier' and 'EasyEnsembleClassifier' objects have no attribute 'n_features_in_' #872, https://github.com/scikit-learn-contrib/imbalanced-learn/issues/872

W3Resource.com: Pandas: Data Manipulation - get_dummies() function, https://www.w3resource.com/pandas/get_dummies.php

BitDegree.org: Splitting Datasets With the Sklearn train_test_split Function, https://www.bitdegree.org/learn/train-test-split

TechTarget.com: over sampling and under sampling, https://www.techtarget.com/whatis/definition/over-sampling-and-under-sampling

MachineLearningMastery.com: Random Oversampling and Undersampling for Imbalanced Classification, https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

GeeksForGeeks.com: Imbalanced-Learn module in Python, https://www.geeksforgeeks.org/imbalanced-learn-module-in-python/

glemaitre.github.io: imblearn.metrics.classification_report_imbalanced, http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html

TowardsDataScience.com: Guide to Classification on Imbalanced Datasets, https://towardsdatascience.com/guide-to-classification-on-imbalanced-datasets-d6653aa5fa23

Springer Link: Index of Balanced Accuracy: A Performance Measure for Skewed Class Distributions, https://link.springer.com/chapter/10.1007/978-3-642-02172-5_57

ImbalancedLearn.org: ClusterCentroids, https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html

MachineLearningMastery.com: How to Combine Oversampling and Undersampling for Imbalanced Classification, https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/

TowardsDataScience.com: Imbalanced Classification in Python: SMOTE-ENN Method, https://towardsdatascience.com/imbalanced-classification-in-python-smote-enn-method-db5db06b8d50

Medium.com: Surviving in a Random Forest with Imbalanced Datasets, https://medium.com/sfu-cspmp/surviving-in-a-random-forest-with-imbalanced-datasets-b98b963d52eb

AnalyticsVidhya.com: Introduction to AdaBoost Algorithm with Python Implementation, https://www.analyticsvidhya.com/blog/2021/03/introduction-to-adaboost-algorithm-with-python-implementation/
