# Titanic-using-machine-learning-with-logistic-regression
Logistic regression is employed in this analysis to foretell which passengers will survive the sinking of the Titanic.

# Introduction

According to Wikipedia, RMS Titanic was a British passenger liner, operated by the White Star Line, which sank in the North Atlantic Ocean on April 15, 1912, after striking an iceberg during her maiden voyage from Southampton, England, to New York City, United States. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it the deadliest sinking of a single ship up to that time. It remains the deadliest peacetime sinking of an ocean liner or cruise ship.

![cover photo](https://user-images.githubusercontent.com/41892582/224569114-eea0afa0-ae66-4633-88e5-2a27466c4e12.jpg)

By taking into account each passenger's sex, age, fare, ticket class, and the number of siblings or parents who were also on board, we hope to create a binary logistic regression model that will represent the pattern of survivors on the Titanic.
and we'll use it to forecast whether some people will survive or not, whose data we won't include in our model.

**Loading libraries**

Firstly I will start by loading some packages that I will use during the analysis

~~~
library(tidyverse)
library(Hmisc)
library(rms)
library(mice)
library(caret)
~~~

**Getting the data**
~~~
##Getting the data from Hmisc and rms libraries
getHdata(titanic3)
##assign it to a variable called "titanic"
titanic<-titanic3
~~~

**Exploration of the data**
~~~
##the structure of the data
glimpse(titanic)
~~~

~~~
Rows: 1,309
Columns: 14
$ pclass    <fct> 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st, 1st,…
$ survived  <labelled> 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, …
$ name      <labelled> "Allen, Miss. Elisabeth Walton", "Allison, Master. Hudson Trevor", "A…
$ sex       <fct> female, male, female, male, female, male, female, male, female, male, male…
$ age       <labelled> 29.0000, 0.9167, 2.0000, 30.0000, 25.0000, 48.0000, 63.0000, 39.0000,…
$ sibsp     <labelled> 0, 1, 1, 1, 1, 0, 1, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, …
$ parch     <labelled> 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, …
$ ticket    <labelled> "24160", "113781", "113781", "113781", "113781", "19952", "13502", "1…
$ fare      <labelled> 211.3375, 151.5500, 151.5500, 151.5500, 151.5500, 26.5500, 77.9583, 0…
$ cabin     <fct> B5, C22 C26, C22 C26, C22 C26, C22 C26, E12, D7, A36, C101, , C62 C64, C62…
$ embarked  <fct> Southampton, Southampton, Southampton, Southampton, Southampton, Southampt…
$ boat      <fct> 2, 11, , , , 3, 10, , D, , , 4, 9, 6, B, , , 6, 8, A, 5, 5, 5, 4, 8, , 7, …
$ body      <labelled> NA, NA, NA, 135, NA, NA, NA, NA, NA, 22, 124, NA, NA, NA, NA, NA, NA,…
$ home.dest <labelled> "St Louis, MO", "Montreal, PQ / Chesterville, ON", "Montreal, PQ / Ch…
~~~
Below is the description of our data variables from "kaggle.com"

**survival**	Survival	0 = No, 1 = Yes

**pclass**	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

**sex**	Sex	

**Age**	Age in years	

**sibsp**	number of siblings / spouses aboard the Titanic	

**parch**	number of parents / children aboard the Titanic	

**ticket**	Ticket number	

**fare**	Passenger fare	

**cabin**	Cabin number	

**embarked**	Port of Embarkation  C = Cherbourg, Q = Queenstown, S = Southampton

~~~
##Eliminate the variables that we wouldn't be using in our analysis.
titanic<-titanic%>%
  select(-name,-ticket,-cabin,-embarked,-boat,-body,-home.dest)
~~~

~~~
##summarize the data
summary(titanic)
~~~

~~~
 pclass       survived         sex           age              sibsp            parch      
 1st:323   Min.   :0.000   female:466   Min.   : 0.1667   Min.   :0.0000   Min.   :0.000  
 2nd:277   1st Qu.:0.000   male  :843   1st Qu.:21.0000   1st Qu.:0.0000   1st Qu.:0.000  
 3rd:709   Median :0.000                Median :28.0000   Median :0.0000   Median :0.000  
           Mean   :0.382                Mean   :29.8811   Mean   :0.4989   Mean   :0.385  
           3rd Qu.:1.000                3rd Qu.:39.0000   3rd Qu.:1.0000   3rd Qu.:0.000  
           Max.   :1.000                Max.   :80.0000   Max.   :8.0000   Max.   :9.000  
                                        NA's   :263                                       
      fare        
 Min.   :  0.000  
 1st Qu.:  7.896  
 Median : 14.454  
 Mean   : 33.295  
 3rd Qu.: 31.275  
 Max.   :512.329  
 NA's   :1 
~~~

~~~
##basic description of the data
describe(titanic)
~~~

~~~
titanic 

 7  Variables      1309  Observations
----------------------------------------------------------------------------------------------
pclass 
       n  missing distinct 
    1309        0        3 
                            
Value        1st   2nd   3rd
Frequency    323   277   709
Proportion 0.247 0.212 0.542
----------------------------------------------------------------------------------------------
survived : Survived 
       n  missing distinct     Info      Sum     Mean      Gmd 
    1309        0        2    0.708      500    0.382   0.4725 

----------------------------------------------------------------------------------------------
sex 
       n  missing distinct 
    1309        0        2 
                        
Value      female   male
Frequency     466    843
Proportion  0.356  0.644
----------------------------------------------------------------------------------------------
age : Age [Year] 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1046      263       98    0.999    29.88    16.06        5       14       21       28 
     .75      .90      .95 
      39       50       57 

lowest :  0.1667  0.3333  0.4167  0.6667  0.7500, highest: 70.5000 71.0000 74.0000 76.0000 80.0000
----------------------------------------------------------------------------------------------
sibsp : Number of Siblings/Spouses Aboard 
       n  missing distinct     Info     Mean      Gmd 
    1309        0        7     0.67   0.4989    0.777 

lowest : 0 1 2 3 4, highest: 2 3 4 5 8
                                                    
Value          0     1     2     3     4     5     8
Frequency    891   319    42    20    22     6     9
Proportion 0.681 0.244 0.032 0.015 0.017 0.005 0.007
----------------------------------------------------------------------------------------------
parch : Number of Parents/Children Aboard 
       n  missing distinct     Info     Mean      Gmd 
    1309        0        8    0.549    0.385   0.6375 

lowest : 0 1 2 3 4, highest: 3 4 5 6 9
                                                          
Value          0     1     2     3     4     5     6     9
Frequency   1002   170   113     8     6     6     2     2
Proportion 0.765 0.130 0.086 0.006 0.005 0.005 0.002 0.002
----------------------------------------------------------------------------------------------
fare : Passenger Fare [British Pound (\243)] 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1308        1      281        1     33.3    38.61    7.225    7.567    7.896   14.454 
     .75      .90      .95 
  31.275   78.051  133.650 

lowest :   0.0000   3.1708   4.0125   5.0000   6.2375, highest: 227.5250 247.5208 262.3750 263.0000 512.3292
----------------------------------------------------------------------------------------------
~~~

**A First Look at the Data**

let's plot the survival marginal distributions.

~~~
##The survival marginal distributions
ggplot(titanic,aes(x=as.factor(survived),fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  theme(legend.position = "none")+
  scale_x_discrete(labels=c("No","Yes"))+
  xlab("survived")+
  ggtitle("The survival marginal distributions")
~~~

![The survival marginal distributions](https://user-images.githubusercontent.com/41892582/224572566-5c5c6263-2d32-40d2-b15b-ec37949cf099.jpg)

As we see, just about 38% of the passengers made it out alive.

~~~
##The survival marginal distributions according to sex
ggplot(titanic,aes(x=sex,fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  guides(fill=guide_legend(title="survived"))+
  ggtitle("The survival marginal distributions according to sex")+
~~~

![The survival marginal distributions according to sex](https://user-images.githubusercontent.com/41892582/224573535-d088fed6-ba10-4bd1-acfa-b5b4cb049b96.jpg)

The survival marginal distributions according to class

~~~
##The survival marginal distributions according to class
ggplot(titanic,aes(x=pclass,fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  guides(fill=guide_legend(title="survived"))+
  ggtitle("The survival marginal distributions according to class")
~~~

![The survival marginal distributions according to sex](https://user-images.githubusercontent.com/41892582/224573911-bb9926da-0757-4147-bf66-50d65a623274.jpg)


The survival marginal distributions according to class
~~~
##The survival marginal distributions according to class
ggplot(titanic,aes(x=pclass,fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  guides(fill=guide_legend(title="survived"))+
  ggtitle("The survival marginal distributions according to class")
~~~

![##The survival marginal distributions according to class](https://user-images.githubusercontent.com/41892582/224573973-89eb803e-75c0-4bef-a1c1-854193a79909.jpg)

Clearly, sex and passenger class have a significant effect on the possibility of survival.

**using variables to investigate trends**

age and the probability of survival

~~~
##age and the probability of survival
ggplot(titanic,aes(x=age,y=survived))+
  geom_smooth(method="loess")+
  ylim (0,1)
~~~

![age and propapility of survival](https://user-images.githubusercontent.com/41892582/224591954-50713206-c5c8-4531-b478-74813bb8ad7f.jpg)

We will employ restricted cubic splines on age with 3, 4, or 5 knots since it is unclear how age affects survival probability and it is obvious that there is not a linear relationship.

Does sex make the effect of age more obvious?  

~~~
##Does sex make the effect of age more obvious?  
ggplot(titanic,aes(x=age,y=survived,col=sex))+
  geom_smooth(method="loess")+
  ylim (0,1)
~~~

![sex and age](https://user-images.githubusercontent.com/41892582/224593059-a4f385d6-904c-4d67-a9bd-184dfd5bcdfc.jpg)

The impact of age is more pronounced when each sex is taken into account separately. where the older female is more likely to survive than the younger, and the opposite is true for a male.

Let's see the impact of passenger class along with age.
~~~
##Let's see the impact of passenger class along with age.
ggplot(titanic,aes(x=age,y=survived,col=pclass))+
  geom_smooth(method="loess")+
  ylim (0,1)
~~~  

![pclass and age](https://user-images.githubusercontent.com/41892582/224594957-5b630d89-6410-4d1f-a4d4-2c70e9c956fd.jpg)

Considering that the lines aren't parallel, we'll include a passenger class/age interaction term.

Split the passenger class and age groups based on gender.
~~~
##Split the passenger class and age groups based on gender.
ggplot(titanic,aes(x=age,y=survived,col=pclass))+
  facet_grid(~sex)+
  geom_smooth(method="loess")+
  ylim (0,1)
~~~

![sex age pclass](https://user-images.githubusercontent.com/41892582/224597407-45e109ca-de0b-4bd3-b473-4b1d03afce91.jpg)

the impact of fare 

~~~
##the impact of fare 
ggplot(titanic,aes(x=fare,y=survived))+
  geom_smooth(method="loess")+
  ylim (0,1)
~~~

![fare](https://user-images.githubusercontent.com/41892582/224651869-84f066a7-fd2c-4aae-be64-a5998e902aa7.jpg)

very positive impact on the probability of survival; investing for outliers

~~~
##investing for outliers
ggplot(titanic,aes(fare,fill=fare))+
  geom_histogram()
~~~

![fare dis ](https://user-images.githubusercontent.com/41892582/224655181-81ce80a9-29b9-4583-9c01-4c5127b3fe1c.jpg)

~~~
ggplot(titanic,aes(fare))+
  geom_histogram()+
  facet_grid(~pclass)
~~~

![fare dis 2](https://user-images.githubusercontent.com/41892582/224655260-3782fc41-9832-4e19-a953-c3a5d58bdaeb.jpg)

There are some outliers among passengers in first class.

# Data cleaning and imputation

first we need to convert the survived variable to factor
~~~
##convert the survived variable to factor
titanic$survived<-as.factor(titanic$survived)
~~~

**Checking for NAs**

~~~
colSums(is.na(titanic))
~~~

~~~
  pclass survived      sex      age    sibsp    parch     fare 
       0        0        0      263        0        0        1 
~~~

There is one NA in fare variable and 263 in age.

pattern of NAs
~~~
md.pattern(titanic)
~~~

![na](https://user-images.githubusercontent.com/41892582/224658766-1e945981-8b28-4065-9c17-2512ec565993.jpg)

Let's look at the observation of the one missing value on fare.

~~~
titanic[which(is.na(titanic$fare)),]
~~~

~~~
     pclass survived  sex  age sibsp parch fare
1226    3rd        0 male 60.5     0     0   NA
~~~

Analysis of Variance between the fare and class of Passengers
~~~
summary(aov(fare~pclass,data=titanic))
~~~

~~~
              Df  Sum Sq Mean Sq F value Pr(>F)    
pclass         2 1272986  636493   372.7 <2e-16 ***
Residuals   1305 2228415    1708                   
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
1 observation deleted due to missingness
~~~

It is statistically significant that the means of the fares differ amongst the classes.

Tukey test to see where the difference is
~~~
TukeyHSD(aov(fare~pclass,data=titanic))
~~~

~~~
  Tukey multiple comparisons of means
    95% family-wise confidence level

Fit: aov(formula = fare ~ pclass, data = titanic)

$pclass
              diff       lwr        upr     p adj
2nd-1st -66.329795 -74.26990 -58.389693 0.0000000
3rd-1st -74.206103 -80.71643 -67.695772 0.0000000
3rd-2nd  -7.876308 -14.74783  -1.004781 0.0198199
~~~

The difference between each class is statistically significant, so the third class fare's median will be used to fill in the fare's missing value.

~~~
titanic$fare[which(is.na(titanic$fare))]<-median(titanic$fare[titanic$pclass=="3rd"],na.rm = TRUE)
~~~

**Age imputation**

Let's start by attempting to determine why there are so many missing values for the age field.

test the theory of the missing age owing to not surviving.

~~~
summary(is.na(age)~survived,data=titanic)
~~~

~~~
is.na(age)      N= 1309  

+--------+-+----+----------+
|        | |   N|is.na(age)|
+--------+-+----+----------+
|survived|0| 809| 0.2348578|
|        |1| 500| 0.1460000|
+--------+-+----+----------+
| Overall| |1309| 0.2009167|
+--------+-+----+----------+
~~~

There is no evidence that the age value was lost as a result of not surviving.

Check it against all the factors

~~~
age.na<-glm(is.na(age)~sex+pclass+survived+parch+sibsp,data=titanic)
summary(age.na)
~~~

~~~
Call:
glm(formula = is.na(age) ~ sex + pclass + survived + parch + 
    sibsp, data = titanic)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-0.3222  -0.2845  -0.1228  -0.0410   1.0389  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  0.157060   0.036134   4.347 1.49e-05 ***
sexmale     -0.002480   0.026931  -0.092  0.92665    
pclass2nd   -0.069131   0.031976  -2.162  0.03080 *  
pclass3rd    0.161677   0.027340   5.914 4.27e-09 ***
survived1   -0.034264   0.027185  -1.260  0.20775    
parch       -0.039656   0.013551  -2.927  0.00349 ** 
sibsp        0.001743   0.011134   0.157  0.87561    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.1493451)

    Null deviance: 210.16  on 1308  degrees of freedom
Residual deviance: 194.45  on 1302  degrees of freedom
AIC: 1234.7

Number of Fisher Scoring iterations: 2
~~~

The only predictor that has a statistically significant impact on missing values in age is "parch ", so we will use the random forest method in the mice function to impute the missing values.

We'll now divide the data into training and testing data and impute the missing age values to each of them separately.

~~~
##make it reproducible
set.seed(113)
#use 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(titanic), replace=TRUE, prob=c(0.8,0.2))
train.titanic  <- titanic[sample, ]
test.titanic   <- titanic[!sample, ]
~~~

~~~
##imputing age into the training set
set.seed(113)
imputed_data<-mice(test.titanic,m=5,maxit = 50,method = 'rf')
~~~

The one whose mean and standard deviation are closest to the original data will be selected from the five we generate.


~~~
mean(titanic$age,na.rm = TRUE)     ##the original data's mean value
[1] 29.88113
lapply(imputed_data$imp$age,mean)  ##each imputed set's mean
$`1`
[1] 30.40678

$`2`
[1] 27.5

$`3`
[1] 28.52542

$`4`
[1] 28.72881

$`5`
[1] 26.01412

sd(titanic$age,na.rm = TRUE)     ##the original data's standard deviation value
[1] 14.4135
lapply(imputed_data$imp$age,sd)  ##each imputed set's standard deviation 
$`1`
[1] 14.64184

$`2`
[1] 12.71267

$`3`
[1] 13.8727

$`4`
[1] 11.49975

$`5`
[1] 11.72532
~~~

The first set will be the one we choose.

~~~
titanic.train.complete<-complete(imputed_data,1)
~~~

~~~
##assign the survived variable from the test.titanic to to "actual"
actual<-test.titanic$survived
##drop it from the dataset
test.titanic<-select(test.titanic,-survived)
~~~

~~~
##imputing age into the testing data
set.seed(113)
imputed_data.test<-mice(test.titanic,m=5,maxit = 50,method = 'rf')

mean(titanic$age,na.rm = TRUE)    ##the original data's mean value
[1] 29.88113
lapply(imputed_data.test$imp$age,mean) ##each imputed set's mean
$`1`
[1] 25.61017

$`2`
[1] 28.33475

$`3`
[1] 28.42938

$`4`
[1] 28.79237

$`5`
[1] 28.34322

sd(titanic$age,na.rm = TRUE)      ##the original data's standard deviation value 
[1] 14.4135
lapply(imputed_data.test$imp$age,sd)  ####each imputed set's standard deviation 
$`1`
[1] 11.51126

$`2`
[1] 13.07831

$`3`
[1] 13.13758

$`4`
[1] 13.33263

$`5`
[1] 13.33414
~~~

The fourth set will be the one we choose.

~~~
titanic.test.complete<-complete(imputed_data.test,4)
~~~

~~~
##Change the sibsp and sibsp variables to factor
titanic.train.complete$sibsp<-as.factor(titanic.train.complete$sibsp)
titanic.train.complete$parch<-as.factor(titanic.train.complete$parch)
titanic.test.complete$sibsp<-as.factor(titanic.test.complete$sibsp)
titanic.test.complete$parch<-as.factor(titanic.test.complete$parch)
~~~

# Fitting and testing the model

Let's look at how each variable impacts the probability of surviving once more before we create our model.

~~~
plot(summary(survived~sex+pclass+age+parch+sibsp,data=titanic3))
~~~

![sur](https://user-images.githubusercontent.com/41892582/224773825-cddbbc17-a0c7-4a55-8fa5-c43eda6f26d6.jpg)


~~~
dd <- datadist (titanic.train.complete)
options(datadist = 'dd')
##our first model
fit1<-lrm(survived~age+sex+pclass+sibsp+parch+fare,data = titanic.train.complete)
fit1
~~~

~~~
Logistic Regression Model

lrm(formula = survived ~ age + sex + pclass + sibsp + parch + 
    fare, data = titanic.train.complete)

                      Model Likelihood       Discrimination    Rank Discrim.    
                            Ratio Test              Indexes          Indexes    
Obs         1023    LR chi2     429.68       R2       0.465    C       0.843    
 0           624    d.f.            18     R2(18,1023)0.331    Dxy     0.686    
 1           399    Pr(> chi2) <0.0001    R2(18,730.1)0.431    gamma   0.687    
max |deriv| 0.07                             Brier    0.147    tau-a   0.327    

           Coef    S.E.    Wald Z Pr(>|Z|)
Intercept   3.0416  0.3994   7.61 <0.0001 
age        -0.0254  0.0069  -3.69 0.0002  
sex=male   -2.5289  0.1777 -14.23 <0.0001 
pclass=2nd -1.1240  0.2714  -4.14 <0.0001 
pclass=3rd -1.9147  0.2678  -7.15 <0.0001 
sibsp=1    -0.1618  0.2042  -0.79 0.4282  
sibsp=2    -0.3075  0.4664  -0.66 0.5096  
sibsp=3    -1.9282  0.6324  -3.05 0.0023  
sibsp=4    -1.8224  0.7512  -2.43 0.0153  
sibsp=5    -9.4166 53.5468  -0.18 0.8604  
sibsp=8    -8.9065 55.8148  -0.16 0.8732  
parch=1     0.5163  0.2696   1.91 0.0555  
parch=2     0.3341  0.3343   1.00 0.3175  
parch=3    -0.0044  0.9354   0.00 0.9963  
parch=4    -8.3219 65.3777  -0.13 0.8987  
parch=5    -1.2721  1.1556  -1.10 0.2710  
parch=6    -8.3988 65.1696  -0.13 0.8975  
parch=9    -8.6391 67.3570  -0.13 0.8979  
fare        0.0019  0.0022   0.89 0.3732 
~~~

The following models will start to include restricted cubic splines on age as well as interaction terms with age and sex.

~~~
fit2<-lrm(survived~pclass+sex*rcs(age,3)+sibsp+parch+fare,data=titanic.train.complete)
fit2
~~~

~~~
Logistic Regression Model

lrm(formula = survived ~ pclass + sibsp + sex * rcs(age, 3) + 
    parch + fare, data = titanic.train.complete)

                      Model Likelihood       Discrimination    Rank Discrim.    
                            Ratio Test              Indexes          Indexes    
Obs         1023    LR chi2     454.44       R2       0.486    C       0.850    
 0           624    d.f.            21     R2(21,1023)0.345    Dxy     0.700    
 1           399    Pr(> chi2) <0.0001    R2(21,730.1)0.448    gamma   0.701    
max |deriv| 0.07                             Brier    0.141    tau-a   0.334    

                Coef    S.E.    Wald Z Pr(>|Z|)
Intercept        2.3887  0.5330  4.48  <0.0001 
pclass=2nd      -1.2219  0.2868 -4.26  <0.0001 
pclass=3rd      -1.9886  0.2798 -7.11  <0.0001 
sibsp=1         -0.1968  0.2082 -0.94  0.3447  
sibsp=2         -0.2332  0.4597 -0.51  0.6119  
sibsp=3         -1.7161  0.6139 -2.80  0.0052  
sibsp=4         -1.8432  0.7021 -2.63  0.0087  
sibsp=5         -9.4349 57.2107 -0.16  0.8690  
sibsp=8         -9.4003 56.7384 -0.17  0.8684  
sex=male        -0.3041  0.5451 -0.56  0.5769  
age              0.0041  0.0201  0.20  0.8398  
age'             0.0016  0.0340  0.05  0.9623  
parch=1          0.3442  0.2835  1.21  0.2247  
parch=2          0.2199  0.3404  0.65  0.5184  
parch=3         -0.3863  0.9536 -0.41  0.6854  
parch=4         -8.7418 61.9833 -0.14  0.8878  
parch=5         -1.6564  1.1700 -1.42  0.1568  
parch=6         -8.7588 61.9897 -0.14  0.8876  
parch=9         -9.0449 67.5271 -0.13  0.8934  
fare             0.0015  0.0022  0.68  0.4934  
sex=male * age  -0.0954  0.0254 -3.75  0.0002  
sex=male * age'  0.0722  0.0409  1.77  0.0775  


~~~

Trying to increase the number of knots on restricted cubic age, eliminating the parch variable due to its insignificance, and defining interaction terms between pclass and sex.

~~~
fit3<-lrm(survived~sex*rcs(age,5)+sibsp+pclass*sex+fare,data=titanic.train.complete)
fit3

~~~

~~~
Logistic Regression Model

lrm(formula = survived ~ sex * rcs(age, 5) + sibsp + pclass * 
    sex + fare, data = titanic.train.complete)

                      Model Likelihood       Discrimination    Rank Discrim.    
                            Ratio Test              Indexes          Indexes    
Obs         1023    LR chi2     504.92       R2       0.528    C       0.868    
 0           624    d.f.            20     R2(20,1023)0.378    Dxy     0.736    
 1           399    Pr(> chi2) <0.0001    R2(20,730.1)0.485    gamma   0.737    
max |deriv|  0.1                             Brier    0.135    tau-a   0.351    

                      Coef    S.E.    Wald Z Pr(>|Z|)
Intercept              4.8729  0.9310  5.23  <0.0001 
sex=male              -1.4096  1.0473 -1.35  0.1783  
age                   -0.0409  0.0455 -0.90  0.3691  
age'                   0.1177  0.2909  0.40  0.6859  
age''                 -0.4607  1.9946 -0.23  0.8173  
age'''                 0.2249  3.0893  0.07  0.9420  
sibsp=1               -0.2850  0.2131 -1.34  0.1810  
sibsp=2               -0.1314  0.4992 -0.26  0.7924  
sibsp=3               -1.9996  0.6822 -2.93  0.0034  
sibsp=4               -2.1627  0.7274 -2.97  0.0029  
sibsp=5               -9.0033 32.1878 -0.28  0.7797  
sibsp=8               -8.1088 34.2258 -0.24  0.8127  
pclass=2nd            -1.7139  0.7652 -2.24  0.0251  
pclass=3rd            -4.0339  0.7232 -5.58  <0.0001 
fare                   0.0011  0.0022  0.49  0.6252  
sex=male * age        -0.2272  0.0611 -3.72  0.0002  
sex=male * age'        1.0723  0.3768  2.85  0.0044  
sex=male * age''      -6.2594  2.5076 -2.50  0.0126  
sex=male * age'''      8.6736  3.8019  2.28  0.0225  
sex=male * pclass=2nd -0.0391  0.8237 -0.05  0.9621  
sex=male * pclass=3rd  2.5798  0.7466  3.46  0.0005  
~~~

trying to eliminate the fare variable

~~~
fit4<-lrm(survived~sex*rcs(age,5)+sibsp+pclass*sex,data=titanic.train.complete)
fit4
~~~~

~~~
Logistic Regression Model

lrm(formula = survived ~ sex * rcs(age, 5) + sibsp + pclass * 
    sex, data = titanic.train.complete)

                      Model Likelihood       Discrimination    Rank Discrim.    
                            Ratio Test              Indexes          Indexes    
Obs         1023    LR chi2     504.68       R2       0.528    C       0.868    
 0           624    d.f.            19     R2(19,1023)0.378    Dxy     0.737    
 1           399    Pr(> chi2) <0.0001    R2(19,730.1)0.486    gamma   0.738    
max |deriv| 0.02                             Brier    0.135    tau-a   0.351    

                      Coef    S.E.    Wald Z Pr(>|Z|)
Intercept              4.9757  0.9082  5.48  <0.0001 
sex=male              -1.4395  1.0452 -1.38  0.1684  
age                   -0.0405  0.0455 -0.89  0.3734  
age'                   0.1156  0.2908  0.40  0.6910  
age''                 -0.4411  1.9941 -0.22  0.8249  
age'''                 0.1859  3.0886  0.06  0.9520  
sibsp=1               -0.2676  0.2100 -1.27  0.2025  
sibsp=2               -0.1012  0.4948 -0.20  0.8379  
sibsp=3               -1.9606  0.6737 -2.91  0.0036  
sibsp=4               -2.1372  0.7250 -2.95  0.0032  
sibsp=5               -8.9599 32.1937 -0.28  0.7808  
sibsp=8               -8.0443 34.2281 -0.24  0.8142  
pclass=2nd            -1.8066  0.7424 -2.43  0.0150  
pclass=3rd            -4.1360  0.6938 -5.96  <0.0001 
sex=male * age        -0.2272  0.0610 -3.72  0.0002  
sex=male * age'        1.0718  0.3767  2.85  0.0044  
sex=male * age''      -6.2724  2.5071 -2.50  0.0124  
sex=male * age'''      8.7128  3.8011  2.29  0.0219  
sex=male * pclass=2nd -0.0029  0.8207  0.00  0.9972  
sex=male * pclass=3rd  2.6159  0.7434  3.52  0.0004  
~~~

In his high ROC area, where c=0.868, we will use model 4 (fit4).

~~~
anova(fit4)

                Wald Statistics          Response: survived 

 Factor                                      Chi-Square d.f. P     
 sex  (Factor+Higher Order Factors)          192.56      7   <.0001
  All Interactions                            53.24      6   <.0001
 age  (Factor+Higher Order Factors)           54.13      8   <.0001
  All Interactions                            17.29      4   0.0017
  Nonlinear (Factor+Higher Order Factors)     33.21      6   <.0001
 sibsp                                        15.68      6   0.0156
 pclass  (Factor+Higher Order Factors)        94.46      4   <.0001
  All Interactions                            31.19      2   <.0001
 sex * age  (Factor+Higher Order Factors)     17.29      4   0.0017
  Nonlinear                                   13.73      3   0.0033
  Nonlinear Interaction : f(A,B) vs. AB       13.73      3   0.0033
 sex * pclass  (Factor+Higher Order Factors)  31.19      2   <.0001
 TOTAL NONLINEAR                              33.21      6   <.0001
 TOTAL INTERACTION                            53.24      6   <.0001
 TOTAL NONLINEAR + INTERACTION                69.83      9   <.0001
 TOTAL                                       235.51     19   <.0001
~~~

let's take a look how the model work at sibsp is zero

~~~
ggplot(Predict(fit4, age , sex , pclass, sibsp=0 , fun=plogis))
~~~

![fit4](https://user-images.githubusercontent.com/41892582/225168149-015079c7-bef4-4bb0-8b03-6cfb62318b6d.jpg)

sibsp are one

~~~
ggplot(Predict(fit4, age , sex , pclass, sibsp=0 , fun=plogis))
~~~

![si](https://user-images.githubusercontent.com/41892582/225168362-983ff25c-481b-494d-8bbb-965d18a3cedc.jpg)

It's time to predict the survived of the remaining 20%.

~~~
test<-predict(fit4,titanic.test.complete)   ##apply the model to the test data
test<-ifelse(test>0.5,1,0)                 ##convert the probability value to survive or not.
confusionMatrix(table(test4,actual))        ##compare the model's output to the actual data
~~~

~~~
Confusion Matrix and Statistics

     actual
test4   0   1
    0 175  49
    1  10  52
                                          
               Accuracy : 0.7937          
                 95% CI : (0.7421, 0.8391)
    No Information Rate : 0.6469          
    P-Value [Acc > NIR] : 4.384e-08       
                                          
                  Kappa : 0.5051          
                                          
 Mcnemar's Test P-Value : 7.530e-07       
                                          
            Sensitivity : 0.9459          
            Specificity : 0.5149          
         Pos Pred Value : 0.7813          
         Neg Pred Value : 0.8387          
             Prevalence : 0.6469          
         Detection Rate : 0.6119          
   Detection Prevalence : 0.7832          
      Balanced Accuracy : 0.7304          
                                          
       'Positive' Class : 0   
~~~

Sensitivity appears to be good (95%) but specificity isn't (only 51%), and overall accuracy appears to be decent (79%).
