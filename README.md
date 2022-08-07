# XAI - Dimensionality Reduction
#### Are they useful? Is there a difference using  projections for pre-processing vs. post-hoc embeddings? PCA, TSNE, UMAP
Submission for XAI - Univeristy of Konstanz Sumer Semester 2023 
***
# Introduction to Dimensionality Reduction
In this section we will do a short introduction on what **dimensionality reduction** is, the main methods and where is it useful. 
(keep in mind that the words, attributes, features and dimensions refer to the same thing so they will be used interchangably throughout this paper. Also the term **DR** will be referring to Dimensionality Reduction  )
Data is the  main component for any machine learning task. We humans are able to te percieve a maximum of 3 dimension but on the real word data the number of dimensions can go up to milions. [Example](https://archive.ics.uci.edu/ml/datasets/URL+Reputation)
For us humans it is impossible to imagine this many features. While there are high dimensional visualization techniuqes such as [PCP](https://www.researchgate.net/publication/282524473_Evaluation_of_Parallel_Coordinates_Overview_Categorization_and_Guidelines_for_Future_Research) that can help in this case, there are still drawbacks to it. Another method used to help us is **dimensionality reduction**. 
==Dimensionality Reduction is the process of reducing the dimensions, but the key principle is that we want to retain the variation from the original dataset as much as possible.==
In a machine learning task pipeline, dimensionality reduction can be used in the pre processing step or post. 
We will discuss further what the difference is in both of these cases.
A downside of dimensionality reduction is that we have to give up variance in the original data. Later we will discuss how can we choose the appropriate number of features we want. While this may seem like a big problem, dimensionality reduction brings us more "goodies". 
- **Performance** : Obviously since we are removing the number of dimensions, less calculations have to be made, therefore faster processing time.
- **Date Visualization** : Reducing the number of dimensions to 2 or 3 can help us plot the data points wich is very important for data visualization tasks.
- **Mitigate overfitting** : When we have a large ammounts of features the model can become more easily overfitted. When we apply
- **Multicollinearity** : ["Multicollinearity is a statistical concept where several independent variables in a model are correlated"](https://www.investopedia.com/terms/m/multicollinearity.asp) PCA (a method for DR)can eliminate multicollinearity between features.
- **Averts from [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)** 
- **Remove noise in data** : During the process of DR, unimportant data points will be removed.
 
Dimensionality Reduction methods can be diveded into 2 groups:
 - **Linear techniques** - This group of methods linearly project the data to lower dimensional space. They preserve global structure
 - **Non-Linear techniques** - Techniques preserve local neighborhood

## Exploring the methods and doing experiments 
In this section we will have a closer look to the DR methods alongside with small snippet codes whith experiments to explain what is happening behind inside the method.
Firstly before diving into the methods I will talk about the dataset for the experiments.
The dataset I choose is called ["Palmer Penguins Dataset"](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris/data)
![](https://editor.analyticsvidhya.com/uploads/96124penguins.png)
A sample of this dataset is as follows
![](https://editor.analyticsvidhya.com/uploads/35133Screenshot%20(8).png)
For pre processing steps we turn the nominal data into numerical categories, normalize the values and dropping any datapoint with null values.

### PCA 
Principal Component analysis a a fairly old and common technique, it dates back to 1901. It is a linear technique. It finds data representation that retains the maximum **nonredundant** and **uncorrelated** information.
The steps to caluclate PCA are the following
 1. Substract mean
 2. Calculate covariance matrix
 3. Calculate Eigenvector Eigenvalue
 4. Forming a feature vector
 5. Deriving new data set

using sklearn library we can easily use PCA whith the following:
```
pca = PCA(n_components=2)
components = pca.fit_transform(data)
```
Which gives us the output 
![](https://gcdnb.pbrd.co/images/2JlElaPov38o.png?o=1)
We can clearly see that DR helped us visualize a high dimensional data into a 2D plot. Now the data tells us that there are 5 clusters of penguins and on each clusters there a distinction between the males and females.

For my first experiment on helping understand PCA I would like to present how easily 


