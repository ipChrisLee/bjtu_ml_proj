这里记录一些关于这个项目的笔记。

首先是项目的定位：

* 这是一个很明显的Few-shot Learning，数据量只有50年，每年12个月，每个月9组数据，需要预测6个月的9组数据。
* 是一个时间序列回归预测问题。
* 数据具有周期性。
  * 可能可以构造多个模型做。



项目相关想法：

* 可能需要考虑多个模型，然后建立评价指标去分析。
* LSTM?



dalao同学的建议：

* py：
  * 用SVM改核函数实现。（比如高斯核函数）
  
    查了下，似乎确有相关资料，放在了参考里了。
  
* QYQ：
  * 用统计的方法做，不建议用网络的方法做。
  * 当成一个周期量，然后分析方差之类的。
  * 多个模型的做法会有些复杂。



关于时间序列：

* [知乎：时间序列分解1](https://zhuanlan.zhihu.com/p/322273740)
* [R_时间序列分解](https://rstudio-pubs-static.s3.amazonaws.com/673856_acde89f3b63c4d40b0e8708ab249f032.html)



参考论文和博客：

* [CSDN:时间序列（arima）+支持向量机（svm）+优化=组合预测](https://blog.csdn.net/u014356002/article/details/53163684)



# 研究结论

## 关于数据性质

跑了下可视化，可以跑src/visualize.py，看vi_img下的图。

大多数数据其实都没有季节性。。。甚至可以说只有"Natural Gas Electric Power Sector CO2 Emissions"具有非常明显的季节性。

之后需要进一步使用定量的方法确定是不是具有季节性/周期性。

大部分数据的趋势也不是很明朗。。。



TBD:

* 定型/定量分析数据的季节性和周期性。
* 之后尝试对每一个单独数据用简单的时间序列分析法做预测（至少对于周期性/季节性比较强的部分可以做）。



## 数据周期性检测的方法  
傅里叶变换
傅里叶变换是一种将时域、空域数据转化为频域数据的方法，任何波形（时域）都可以看做是不同振幅、不同相位正弦波的叠加
对于一条具备周期性的时间序列，它本身就很接近正弦波，所以它包含一个显著的正弦波，周期就是该正弦波的周期，而这个正弦波可以通过傅里叶变换找到，它将时序数据展开成三角函数的线性组合，得到每个展开项的系数，就是傅里叶系数。傅里叶系数越大，表明它所对应的正弦波的周期就越有可能是这份数据的周期。  


自相关系数
自相关系数（Autocorrelation Function）度量的是同一事件不同时间的相关程度，不同相位差（lag）序列间的自相关系数可以用 Pearson 相关系数计算
当序列存在周期性时，遍历足够多的相位差，一定可以找到至少一个足够大的自相关系数，而它对应的相位差就是周期。所以对于检测时序周期来说，只需找到两个自相关系数达到一定阈值的子序列，它们起始时间的差值就是我们需要的周期。+

数据相关性的研究和评价：
pandas相关系数-DataFrame.corr()方法，该方法用来计算DataFrame对象中所有列之间的相关系数  
DataFrame.corr(method)  
参数说明：  
method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}  
pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。  
kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据  
spearman：非线性的，非正太分析的数据的相关系数min_periods：样本最少的数据量  
本实验中适合使用pearson相关系数来衡量数据间的相关性，相关系数的绝对值越大，相关性越强：相关系数越接近于1或-1，相关度越强，相关系数越接近于0，相关度越弱。  

以下是pearson相关系数绝对值大于0.7的数据组合  
Coal Electric Power Sector CO2 Emissions  and  Petroleum Coke Electric Power Sector CO2 Emissions 0.6688  
Coal Electric Power Sector CO2 Emissions  and  Total Energy Electric Power Sector CO2 Emissions 0.9399  
Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions and Residual Fuel Oil Electric Power Sector CO2 Emissions 0.7504  
Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions and  Petroleum Electric Power Sector CO2 Emissions 0.7985  
Petroleum Coke Electric Power Sector CO2 Emissions   and Total Energy Electric Power Sector CO2 Emissions 0.7635  
Residual Fuel Oil Electric Power Sector CO2 Emissions  and Petroleum Electric Power Sector CO2 Emissions 0.9951  
