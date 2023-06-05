### <p style="text-align: center;">SF1935 Probability Theory and Statistics with Application to Machine Learning</p>
# <p style="text-align: center;">Short report on project assignment</p>
### <p style="text-align: center;">Bayesian linear regression</p>
## <p style="text-align: center;">Leo Svanemar, Laura Nilsson</p>
### <p style="text-align: center;">2023-05-27</p>
____

## Introduction
In this assignment, Bayesian linear regression and Maximum Likelihood Estimaton (MLE) is used to estimate the parameters of a linear model. The assignment consists of two tasks. In the first task, the parameters are estimated using Bayesian linear regression. In the second task, the parameters are estimated using Bayesian linear regression and MLE. The results are compared and discussed.

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

## Result

### Task 1
____
**1.1**

![Task 1.1](1.1.png) </br>
*1.1: Prior distribution over w.*

</br>

**1.2**

![Task 1.2](1.2.png) </br>
*1.2: Posterior distribution over w for single data point.*

</br>


**1.3**

![Task 1.3](1.3.png) </br>
*1.3: 5 model samples drawn from posterior obtained in 1.2.*

</br>

**1.4**

![Task 1.4.1](1.4.1.png "Task 1.4.1") </br>
*1.4.1: Posterior distribution over w for two data points.*

![Task 1.4.2](1.4.2.png) </br>
*1.4.2: 5 model samples drawn from posterior obtained in 1.4.1*

![Task 1.4.3](1.4.3.png) </br>
*1.4.3: Posterior distribution over w for three data points.*

![Task 1.4.4](1.4.4.png) </br>
*1.4.4: 5 model samples drawn from posterior obtained in 1.4.3*

![Task 1.4.5](1.4.5.png) </br>
*1.4.5: Posterior distribution over w for four data points.*

![Task 1.4.6](1.4.6.png) </br>
*1.4.6: 5 model samples drawn from posterior obtained in 1.4.5*

![Task 1.4.7](1.4.7.png) </br>
*1.4.7: Posterior distribution over w for five data points.*

![Task 1.4.8](1.4.8.png) </br>
*1.4.8: 5 model samples drawn from posterior obtained in 1.4.7*

![Task 1.4.9](1.4.9.png) </br>
*1.4.9: Posterior distribution over w for six data points.*

![Task 1.4.10](1.4.10.png) </br>
*1.4.10: 5 model samples drawn from posterior obtained in 1.4.9*

![Task 1.4.11](1.4.11.png) </br>
*1.4.11: Posterior distribution over w for seven data points.*

![Task 1.4.12](1.4.12.png) </br>
*1.4.12: 5 model samples drawn from posterior obtained in 1.4.11*

</br>

**1.5**

The posterior distribution becomes more concentrated and localized the more datapoints you add. The linear models show less variability as more datapoints are added.

</br>

**1.6**           

|$\sigma ^2$ = 0.1                |$\sigma^2$ = 0.4                 |$\sigma ^2$ = 0.8               |
|---------------------------------|---------------------------------|--------------------------------|
|![Task 1.4.11](1.4.11.png) </br> |![Task 1.6.2](1.6.2.png) </br>   | ![Task 1.6.4](1.6.4.png) </br> |
|![Task 1.6.1](1.6.1.png)  </br>  |![Task 1.6.3](1.6.3.png) </br>   | ![Task 1.6.5](1.6.5.png) </br> |

### Task 2
____
**2.1 och 2.2**  
  
![2.1](2.1.png)  
*The plotted data.*
  
![2.2](2.2.png)   
*Another angle, where the location of test vs training data is visible. The division is |x1| > 0.3 and |x2| > 0.3 is test.*

</br>

**2.3 Model fitting with MLE, sigma ∈ {0.2, 0.4, 0.6}**


Mimimun square error (MSE) </br>

MSE (σ² = 0.2) = 0.1971901417530678 </br>
MSE (σ² = 0.4) = 0.4491506427837673 </br>
MSE (σ² = 0.6) = 0.556552760694397 </br>

</br>

**2.4 Bayesian linear regression with different values of the uncertainty parameter of the Gaussian prior over the weight parameters (α ∈ {0.7, 1.5, 3.0})**
  
|σ² = 0.2 </br>                                | σ² = 0.4                                    |σ² = 0.6
|----------------------------------------------|---------------------------------------------|---------------------------------------------
|MSE (σ² = 0.2, α = 0.7) = 0.1971494682678761  |MSE (σ² = 0.4, α = 0.7) = 0.3981424796062423 |MSE (σ² = 0.6, α = 0.7) = 0.6091655155964246
|MSE (σ² = 0.2, α = 1.5) = 0.22452106396164226 |MSE (σ² = 0.4, α = 1.5) = 0.400761638968057  |MSE (σ² = 0.6, α = 1.5) = 0.6000864367300098
|MSE (σ² = 0.2, α = 3.0) = 0.18572319863608935 |MSE (σ² = 0.4, α = 3.0) = 0.4069250692467875 |MSE (σ² = 0.6, α = 3.0) = 0.6091084391212728  

</br>

**2.5 A comparsion between the frequentist approach MLE and the Bayesian can be made by comparing the MSE.**

**MLE**

MSE (σ² = 0.2) = 0.1971901417530678  
MSE (σ² = 0.4) = 0.4491506427837673  
MSE (σ² = 0.6) = 0.556552760694397  

**Bayesian**

MSE (σ² = 0.2, α = 0.7) = 0.1971494682678761  
MSE (σ² = 0.2, α = 1.5) = 0.22452106396164226  
MSE (σ² = 0.2, α = 3.0) = 0.18572319863608935  
MSE (σ² = 0.4, α = 0.7) = 0.3981424796062423  
MSE (σ² = 0.4, α = 1.5) = 0.400761638968057  
MSE (σ² = 0.4, α = 3.0) = 0.4069250692467875  
MSE (σ² = 0.6, α = 0.7) = 0.6091655155964246  
MSE (σ² = 0.6, α = 1.5) = 0.6000864367300098  
MSE (σ² = 0.6, α = 3.0) = 0.6091084391212728  

</br>

**2.6 Generate predictions also for the training data and compare the uncertainty between the training and test data samples. How does uncertainty (standard deviation) and the quality (mean) of these 2 predictions change with varying α and σ.**

σ² = 0.2, α = 0.7</br>
Bayesian - Training Data</br>
Std: 0.7876788308751822</br>
Mean: 0.01906156446596578</br>
Bayesian - Test Data</br>
Std: 1.0407432265081926</br>
Mean: 0.029264653171258218</br>

σ² = 0.2, α = 1.5</br>
Bayesian - Training Data</br>
Std: 0.8142590931095135</br>
Mean: 0.0054173364556214425</br>
Bayesian - Test Data</br>
Std: 1.0758455310388912</br>
Mean: 0.01498431885304853</br>

σ² = 0.2, α = 3.0</br>
Bayesian - Training Data</br>
Std: 0.8074109491034741</br>
Mean: -0.0354671661765233</br>
Bayesian - Test Data</br>
Std: 1.0667830726130483</br>
Mean: -0.026857178186670293</br>

σ² = 0.4, α = 0.7</br>
Bayesian - Training Data</br>
Std: 0.8362360388406769</br>
Mean: -0.03247183396026418</br>
Bayesian - Test Data</br>
Std: 1.1048732936631402</br>
Mean: -0.023214406001312315</br>

σ² = 0.4, α = 1.5</br>
Bayesian - Training Data</br>
Std: 0.8039373853285245</br>
Mean: -0.0045987907860633535</br>
Bayesian - Test Data</br>
Std: 1.0622195101496643</br>
Mean: 0.005502790301335658</br>

σ² = 0.4, α = 3.0</br>
Bayesian - Training Data</br>
Std: 0.7626958263365937</br>
Mean: -0.008495965609520558</br>
Bayesian - Test Data</br>
Std: 1.0077022567743306</br>
Mean: -0.0004560067631212476</br>

σ² = 0.6, α = 0.7</br>
Bayesian - Training Data</br>
Std: 0.7450415957809041</br>
Mean: -0.008494992020619851</br>
Bayesian - Test Data</br>
Std: 0.9843989345876711</br>
Mean: 0.0006871280427911734</br>

σ² = 0.6, α = 1.5</br>
Bayesian - Training Data</br>
Std: 0.7676579896833775</br>
Mean: -0.0036604083407384045</br>
Bayesian - Test Data</br>
Std: 1.0142651917051924</br>
Mean: 0.004859680921884332</br>

σ² = 0.6, α = 3.0</br>
Bayesian - Training Data</br>
Std: 0.786802911307473</br>
Mean: 0.01810248009176131</br>
Bayesian - Test Data</br>
Std: 1.039557899736239</br>
Mean: 0.02668229540784155</br>

## Discussion

The results from task 1 show how adding more datapoints to the likelihood calculation generates models with an increased likelihood of displaying the true parameter values. This is evident as the posterior distributions narrow down with more datapoints, and the linear models align better resulting in a decreasing risk of overfititng or underfitting. Additionally, adding more data points reduces risks the model relying on potential outliers or individual datapoints as well as increases likelihood of capturing underlying patterns leading to more accurate predictions. Combined, this increased information reduces bias and variance by bringing the model closer to the true underlying function. Furthermore when performing the modelling on different values of noise variance (σ²), it is evident that the more noise, i.e. the less precise the data points are, the less precise the posterior distributions will be reflecting the increased uncertainty between the model and the true parameter values. 

Discussing the results of task 2, for σ² = 0.2, the Bayesian Linear regression model performs best when α = 3.0, with an MSE of ~0.1857. This indicates that a higher uncertainty parameter combined with a lower noise variance leads to better predictions.

Comparing Frequentist and Bayesian Linear regression the standard deviation represents the spread or uncertainty of the predictions. In general, the standard deviation tends to be similar between the frequentist and Bayesian approaches for each combination of sigma_squared and alpha.

The mean represents the average prediction and those values differ slightly between the frequentist and Bayesian approaches for each combination of sigma_squared and alpha.

Overall, the differences between the frequentist and Bayesian approaches in terms of standard deviation and mean are relatively small. The Bayesian approach accounts for uncertainty by incorporating the prior distribution over the weight parameters, while the frequentist approach relies solely on the maximum likelihood estimation. However, in this specific scenario, the differences in predictive performance between the two approaches appear to be minimal.

Moreover runnig an algorithm on its test data can lead to overfitting as evident in the last task and is not recommended as the model recognises it all.

Lastly, the training data only incorporate a small middle part of the data set and therefore is not an optimal representation for the entirety of it. Therefor running the test data on the model trained on less than optimal train data can result in inaccurate results which can affect the comparisons between the modelling approaches.    
