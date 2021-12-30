# Predict Attrition on IBM HR Analytics Dataset
This project aims at predicting employee attrition using IBM HR Analytics dataset. 
The script implements 9 Machine Learning algorithms with CARET package in order to solve a binary classification problem. 
Algorithms' plots, results and resampling plots are exported as output files. The project also provides a PowerBI dashboard for data visualization.

Project's files are:
1. *WA_Fn-UseC_-HR-Employee-Attrition.csv* is the source dataset available [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset);
2. *figs* folder contains plots of confusion matrices, ROC curves, correlation, models' performances and variables' importance;
3. *1_IBM_HR_Analytics_Preprocessing.R* is the first R script which imports the dataset and proprocess it;
4. *2_IBM_HR_Analytics_Classification.R* is the second R script which performs 9 Machine Learning algorithms using CARET package;
5. *IBM HR Preprocessed.xlsx* is the output dataset after preprocessing;
6. *Models' results.xlsx* tables all algorithms' performance indicators which are taken into account;
7. *Time Elapsed.png* plots how algorithms' total elapsed time is distributed;
8. *Console output.txt* stores total script's time elapsed and detailed information about algorithms' execution;
9. *IBM_HR_Analytics_Dashboard.pbix* is a PowerBI dashboard for data visualization and analysis.

You can find further insights [here](http://inseaddataanalytics.github.io/INSEADAnalytics/groupprojects/January2018FBL/IBM_Attrition_VSS.html#business_problem).
