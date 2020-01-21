# Hello Everyone
In this project, we will learn the effect of feature selection on the decision tree with Python. I did my studies using the <a href="https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping"> Forest Type Mapping </a> dataset. 

## What do you need?
You must install the following libraries.
+ Scikit-learn
+ Pandas
+ Seaborn

## What's in the project?
+ First of all, the decision tree was trained with all the features we had.
  + The accuracy of the trained decision tree was tested and reported with test data.
  + Features were visualized.
  + Important features were selected.
  + New training and test data have been created.
+ The decision tree was retrained with new training data.
  + The accuracy of the trained decision tree was tested and reported with new test data.
+ Finally, correlation matrix and heatmap were created with new training data.

## Important Features Selection
![](https://i.ibb.co/FKnCbbb/Figure-1.png)

The photo above shows the effects of the features used for training on the classification. As you can see, the first 6 attributes have a greater impact on the classification than others. For this reason, these 6 features will be used in the new data set.

## Reporting
When we compare the two decision trees we have, we see that the accuracy rate of the second decision tree is higher. For this reason, feature selection is very important when training our models.

![](https://i.ibb.co/Rvr87Wd/Classification-Report.png)

## Heatmap
The heat map we have seen bellow has been created with the correlation matrix of the new training data. If we encountered values close to 1 in the table, we could remove one of the 2 features from the dataset.

![](https://i.ibb.co/8j64nHp/Figure-2.png)
