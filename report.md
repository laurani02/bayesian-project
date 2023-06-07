### <p style="text-align: center;">SF1935 Probability Theory and Statistics with Application to Machine Learning</p>
# <p style="text-align: center;">Short report on project assignment</p>
### <p style="text-align: center;">Bayesian linear regression</p>
## <p style="text-align: center;">Leo Svanemar, Laura Nilsson</p>
### <p style="text-align: center;">2023-05-27</p>
____

## Introduction - objectves and scope
In this assignment, Bayesian linear regression and Maximum Likelihood Estimaton (MLE) are used to estimate the parameters of a linear model. The assignment consists of two tasks. In the first task, the parameters are estimated using Bayesian linear regression. In the second task, the parameters are estimated using Bayesian linear regression and MLE. The results are compared and discussed.

**Goals**

- Task 1
  - Implement Bayesian linear regression with one-dimensional input
  - Obtain the posterior probability
  - Examine prior and posterior over w
  - Examine how adding more data points and varying the noise level affects the accuracy

- Task 2
  - Implement Bayesian linear regression and MLE in a multidimensional input space
  - Examine how varying the noise level and weight parameters affects the model accuracy
  - Examine how training data and test data are affected by those
  - Use batch learning to estimate the parameters



## Method
For both tasks, Python was used. The code is available in the files [`warmup.ipynb`](https://github.com/laurani02/bayesian-project/blob/main/warmup.ipynb) and [`bayesian.ipynb`](https://github.com/laurani02/bayesian-project/blob/main/bayesian.ipynb). The imported and used libraries are scipy, numpy and matplotlib. While SciPy is used in task 1 for multivariate normal distribution and distance calculation, NumPy and Matplotlib are used in both tasks for calculations and visualisation.

## Results and discussion

### Task 1
____

In task 1, Bayesian linear regression is implemented with one-dimensional input. The posterior probability is obtained and the prior and posterior over w are examined. The effect of adding more data points and varying the noise level on the accuracy is examined.

The results from task 1 show how adding more datapoints to the likelihood calculation generates models with an increased likelihood of displaying the true parameter values. This is evident as the posterior distributions narrow down with more datapoints. **This is shown in plot X** The linear models align better resulting in a decreasing risk of overfititng or underfitting. Additionally, adding more data points reduces risks the model relying on potential outliers or individual datapoints as well as increases likelihood of capturing underlying patterns leading to more accurate predictions. Combined, this increased information reduces bias and variance by bringing the model closer to the true underlying function. Furthermore when performing the modelling on different values of noise variance (σ²), it is evident that the more noise, i.e. the less precise the data points are, the less precise the posterior distributions will be reflecting the increased uncertainty between the model and the true parameter values.

</br>


![Task 1.1](1.4.1.png "Task 1.4.1") </br>
*1.4.1: Posterior distribution over w for two data points.*


![Task 1.2](1.4.11.png) </br>
*1.4.2: Posterior distribution over w for seven data points.*

</br>

Do we want this:
The posterior distribution becomes more concentrated and localized the more datapoints you add. The linear models show less variability as more datapoints are added.

</br>

### Task 2
____

In task 2, Bayesian linear regression and MLE are implemented in a multidimensional input space. The effect of varying the noise level and weight parameters on the model accuracy is examined. The effect of testing the Bayesian model on training data versus test data is also examined. 

**The Data**
Before model development, the data was plotted to get an overview. In plot 2.1, the data is plotted and the location of the test and training data is visible. The data where |x1| > 0.3 and |x2| > 0.3 is kept for test. The rest of the data is used for training.


![2.1](2.2.png)   
*A plot of the data, where the location of test vs training data is visible. Data where |x1| > 0.3 and |x2| > 0.3 is kept for test.*


**Comparing Frequentist and Bayesian approach**

A comparsion between the frequentist approach MLE and the Bayesian is done by comparing the mean squared error (MSE), the squared difference between the true and predicted values, when tested. A lower MSE value implies that the model is more accurate. In addition to different noise levels (σ^2), the Bayesian model has different uncertainty levels (α) and the results are compared for different combinations of noise and uncertainty levels.  

The differences in predictive performance between the two approaches appear to be minimal. While the Baysian approach had a slightly lower MSE than the frequentist approach when noise was set to σ² = 0.2 (either α = 3.0 or α = 0.7) and σ² = 0.4 ( for all α ), the differences between the models were not significant. Only in the case of σ² = 0.6, the frequentist approach had a lower MSE than the Bayesian approach for all uncertinty levels.  

A general trend is that the MSE increases with increasing noise level. This is expected as the noise level is the variance of the noise and the noise is added to the data. The higher the noise level, the more the data is affected and the less accurate the model becomes. Not much can be said about how the different values of the uncertainty parameter α makes the model behave since it has differed between tests.

**Batch learning**
  
The model learned from the available training data. As visible in plot 2.1, the training and test data is not evenly distributed in the room. When testing the Bayesian model on the training vs test data, naturally the model has very different predictions for the two types. 


## Final remarks

The lab was confusing at first when getting introduced to the new modelling but it was not all too unfamiliar, as both authors had some knowledge about statistical and probability modelling as well as how to write some machine learning code from previous courses and the rest of SF1935. The simluation outcomes are based on small datasets and due to the discrepancies between training and testing data outcomes cannot be expected to 100% adhere to the theory which is reasonable. 
