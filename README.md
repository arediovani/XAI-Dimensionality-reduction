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
- **Date Visualization** : Reducing the number of dimensions to 2 or 3 can help us plot the data points wich is very important for data visualization tasks.
- **Mitigate overfitting** : When we have a large ammounts of features the model can become more easily overfitted. When we apply

