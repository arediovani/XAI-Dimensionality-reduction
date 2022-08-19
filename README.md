# XAI - Dimensionality Reduction
#### Are they useful? Is there a difference using  projections for pre-processing vs. post-hoc embeddings? PCA, TSNE, UMAP
Submission for XAI - Univeristy of Konstanz Sumer Semester 2022
***
# Introduction to Dimensionality Reduction
In this section we will do a short introduction on what **dimensionality reduction** is, the main methods and where is it useful. 
(keep in mind that the words, attributes, features, and dimensions refer to the same thing so they will be used interchangeably throughout this paper. Also, the term **DR** will be referring to Dimensionality Reduction  )
Data is the main component of any machine learning task. We humans are able to te percieve a maximum of 3 dimensions but on the real word data, the number of dimensions can go up to millions. [Example](https://archive.ics.uci.edu/ml/datasets/URL+Reputation)
For us humans, it is impossible to imagine these many features. While there are high-dimensional visualization techniques such as [PCP](https://www.researchgate.net/publication/282524473_Evaluation_of_Parallel_Coordinates_Overview_Categorization_and_Guidelines_for_Future_Research) that can help in this case, there are still drawbacks to it. Another method used to help us is **dimensionality reduction**. 
==Dimensionality Reduction is the process of reducing the dimensions, but the key principle is that we want to retain the variation from the original dataset as much as possible.==
In a machine learning task pipeline, dimensionality reduction can be used in the pre-processing step or post. 
We will discuss further what the difference is in both of these cases.
A downside of dimensionality reduction is that we have to give up variance in the original data. Later we will discuss how can we choose the appropriate number of features we want. While this may seem like a big problem, dimensionality reduction brings us more "goodies". 
- **Performance** : Obviously since we are removing the number of dimensions, fewer calculations have to be made, therefore faster processing time.
- **Date Visualization** : Reducing the number of dimensions to 2 or 3 can help us plot the data points which is very important for data visualization tasks.
- **Mitigate overfitting** : When we have a large amount of features the model can become more easily overfitted. When we apply
- **Multicollinearity** : ["Multicollinearity is a statistical concept where several independent variables in a model are correlated"](https://www.investopedia.com/terms/m/multicollinearity.asp) PCA (a method for DR)can eliminate multicollinearity between features.
- **Averts from [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)** 
- **Remove noise in data** : During the process of DR, unimportant data points will be removed.
 
Dimensionality Reduction methods can be divided into 2 groups:
 - **Linear techniques** - This group of methods linearly project the data to lower dimensional space. They preserve the global structure
 - **Non-Linear techniques** - Techniques preserve local neighborhood

## Exploring the methods and doing experiments 
In this section, we will have a closer look at the DR methods alongside with small snippet codes with experiments to explain what is happening behind the method.
Firstly before diving into the methods I will talk about the dataset for the experiments.
The dataset I choose is called ["Palmer Penguins Dataset"](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris/data)
![](https://editor.analyticsvidhya.com/uploads/96124penguins.png)
A sample of this dataset is as follows
![](https://editor.analyticsvidhya.com/uploads/35133Screenshot%20(8).png)
For pre processing steps we turn the nominal data into numerical categories, normalize the values, and drop any datapoint with null values.

### PCA 
Principal Component analysis a fairly old and common technique, it dates back to 1901. It is a linear technique. It finds data representation that retains the maximum **nonredundant** and **uncorrelated** information.
The steps to calculate PCA are the following
 1. Subtract mean
 2. Calculate covariance matrix
 3. Calculate Eigenvector Eigenvalue
 4. Forming a feature vector
 5. Deriving new data set

using sklearn library we can easily use PCA with the following:
```
pca = PCA(n_components=2)
components = pca.fit_transform(data)
```
Which gives us the output 
![](https://gcdnb.pbrd.co/images/2JlElaPov38o.png?o=1)
We can clearly see that DR helped us visualize high-dimensional data into a 2D plot. Now the data tells us that there are 5 clusters of penguins and in each cluster there is a distinction between the males and females.

For my first experiment on helping understand PCA, I would like to demonstrate how easily PCA can be skewed if there are noisy or outlier data.
for this, I removed the step that deletes outliers.
![](https://gcdnb.pbrd.co/images/W6uguAEmAN6H.png?o=1)
As we can see outliers have a massive impact on the final output of PCA. As we explained, for each dimension there is a principal component that shows the variance in that dataset. 
If we use the following method from Sklearn, we can have access to the variance of each dimension
```
exp_var_cumul = np.cumsum(components.explained_variance_ratio_)
```
We get this output
![](https://gcdnb.pbrd.co/images/tcghUlBIjiyu.png?o=1)
for a chosen k number of components, PCA will retain the k principal components with the lowest values (sorted from low to high)

### T-sne
t-SNE was introduced in 2008 by Laurens van der Maaten and Geoffrey Hinton in [this paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
Unlike PCA, t-SNE is a nonlinear dimensionality reduction method. This means that t-sne can differentiate data that cannot be differentiated by any straight line.
Below is a figure of what happens during t-SNE.
![](https://gcdnb.pbrd.co/images/nBbIvbCNMIEE.png?o=1)
In the first step, t-SNE determines the similarity of the points. The similarity is defined in the original paper as follows **"similarity of datapoint xⱼ to datapoint xᵢ is the conditional probability p_{j|i}, that xᵢ would pick xⱼ as its neighbor"**
In step one we calculate the distance between the black point and all other points, we do the same thing in the second step with another point. We do this for all the points.
For each datapoint, we plot the distance to a normal distribution curve. We portray this action in step 3.
Any datapoint that is close to the chosen datapoint (in the first case the black datapoint) means there is a high similarity. All data points that have a high distance will be further from the center of the curve. We scale all the similarities so they all add up to 1.
We use the following formula proposed on the paper.
![](https://miro.medium.com/max/263/1*1gBOzGPwWEN4L_HhYLN-VQ.png)
An important note that we need to add for step 4 is that the width of the curve depends on the cluster density near the chosen datapoint. A cluster that contains data points close to each other will have a narrower curve, while clusters that don't have dense data points will have a wider curve.
We do this because we want to receive similarity scores for clusters without regarding the density of that cluster.
In step number 5 we see a similarity matrix. Using this matrix we can plot which data points are similar to each other. 
When a datapoint is similar to another data point (using the distribution) we set that value as "high similarity" or in this case just 1, and vice versa. 
The final objective is to plot the high dimension into a lower one. 
So in step 5 t-SNE randomly plots the points into a lower dimension, in each iteration the points are pushed and pulled based on their similarity matrix. The result that we want to achieve is the similarity matrix on the right.
**Let's dive deeper on what the hyperparameters are doing**
t-SNE has 3 main hyperparameters
1. Initial Dimensions: The number of dimensions that t-SNE will work with. t-SNE still relies on PCA to remove principal components with low variance to speed up the processing time.
2. perplexity (this hyperparameter is complex so we will go into detail below)
3. iterations: The maximum number during the DR process, default value ranges to 1000

Perplexity is an important and complex hyperparameter. A simple way to define it would be as follows:
**A perplexity is a target number of neighbors for our central point**. For a mathematics fanatic, this is the formula: ![](https://miro.medium.com/max/223/1*3_qQH7KjQR89ymcDk0Y5Yw.png)
This also explains that the variance(σ) is dependent on the Gaussian distribution alongside the total number of data points around it.
A question asked by prof. Strobelt was "Why does t-SNE show clusters even when the data is distributed in a gaussian noise". Judging by our observations of testing results, the reason this happends is due to the naature of how perplexity treats density and distance between clusters. We mentioned before that Gaussian distribution is used to determine the similarity of the data points, given this the shape, density and distance between clusters dont mean anything. With a small number of perplexity, there will be a narrow distribution, which means t-SNE will group up datapoints that are close to each other and plot them as clusters. But if we give a larger number of iterations and a larger perplexity, t-SNE will show the random noisy data once again.
```
tsne = TSNE(n_components=2,n_iter=1000,perplexity=80)
projections = tsne.fit_transform(data)
```
the output of this code:
![](https://gcdnb.pbrd.co/images/c433YZug7sCC.png?o=1)
As we explained in the upper parts, perplexity is a hyper parameter, a lower number will give you more noise since the t-distribution is capturing the local embedding, but if we give it a higher perplexity it will try to preserve the global structure.
![high number of perplexity](https://gcdnb.pbrd.co/images/0kQgxQACE4zW.png?o=1)
t-sne with a high number of perplexity, as we can see it is showing us the same global features as PCA.
![](https://gcdnb.pbrd.co/images/t7RMHID5qncG.png?o=1)
t-sne with a low number of perplexity. This demonstrates how the lower number of perplexity will give us mora noisy data.

We showed 2 main methods of DR and how important they are. 
Let's now demonstrate their effectiveness in a real-world scenario.
Dimensionality reduction can be used in many ways, but the two most prominent are before or after the black box model as shown below
![](https://i.ibb.co/wccgx95/Desktop.png)
DR can be used to do feature extraction before entering the black box model. Or it can be used after the model to plot the data points.
## Experiment 2 - Improving Accuracy
For an experiment, we decided to prove that DR reduction can help extract features and improve accuracy.
**experiment is under preprocess.ipynub file**.
The dataset is the same, and we use the same preprocessing steps.
We are using regression classification to determine the gender of the penguin based on the data.
Before PCA:
```
model = LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, data, labels, scoring='accuracy', cv=cv, n_jobs=-1)
```
**Accuracy: 0.826 (0.055)**
After PCA:
```
steps = [('pca', PCA(n_components=6)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, data, labels, scoring='accuracy', cv=cv, n_jobs=-1)
```
**Accuracy: 0.902 (0.044)**
## Visualizing Word embedding.
DR can be used after the model in the Machine Learning Pipeline to visualize the separability of the data.
Our Experiment focuses on turning a high dimensionality vector such as a sentence vector into a 2D plot. 
[Experiment here](https://www.kaggle.com/code/valternamazani/bert-vis/edit)

## Conclusion
In this Explainable, we demonstarted and explained how useful Dimensionality Reduction is. Not only that, we also dove deeper into how the meain methods work on the inside. As a conclusion for this Explainable we would like to address the reader that Dimensionality Reduction should be carefully examined on what output is trying to insert inside your model. The hyperparameters are a tool that you will have to tinker for a long time, and we hope this paper helped you gain just a little bit of insight on what is happening in tuning of the hyperparameters.
