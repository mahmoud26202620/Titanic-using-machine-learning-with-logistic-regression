library(tidyverse)
library(Hmisc)
library(rms)
library(mice)

##Getting the data from Hmisc and rms libraries
getHdata(titanic3)
titanic<-titanic3    ##assign it to a variable called "titanic"

##the structure of the data
glimpse(titanic)


##Eliminate the variables that we wouldn't be using in our analysis.
titanic<-titanic%>%
  select(-name,-ticket,-cabin,-embarked,-boat,-body,-home.dest)

##summarize the data
summary(titanic)

##basic description of the data
describe(titanic)

##A First Look at the Data

##The survival marginal distributions
ggplot(titanic,aes(x=as.factor(survived),fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  theme(legend.position = "none")+
  scale_x_discrete(labels=c("No","Yes"))+
  xlab("survived")+
  ggtitle("The survival marginal distributions")
##Just about 38% of the passengers made it out alive

##The survival marginal distributions according to sex
ggplot(titanic,aes(x=sex,fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  guides(fill=guide_legend(title="survived"))+
  ggtitle("The survival marginal distributions according to sex")

##The survival marginal distributions according to class
ggplot(titanic,aes(x=pclass,fill=as.factor(survived)))+
  geom_bar(width = 0.6)+
  guides(fill=guide_legend(title="survived"))+
  ggtitle("The survival marginal distributions according to class")

ggplot(titanic,aes(x=survived,y=age))+
  geom_boxplot()+
  facet_grid(~sex)

##using variables to explore trends

##age and the probability of survival
ggplot(titanic,aes(x=age,y=survived))+
  geom_smooth(method="loess")+
  ylim (0,1)

##Does sex make the effect of age more obvious?  
ggplot(titanic,aes(x=age,y=survived,col=sex))+
  geom_smooth(method="loess")+
  ylim (0,1)

##Let's see the impact of passenger class along with age.
ggplot(titanic,aes(x=age,y=survived,col=pclass))+
  geom_smooth(method="loess")+
  ylim (0,1)

##Split the passenger class and age groups based on gender.
ggplot(titanic,aes(x=age,y=survived,col=pclass))+
  facet_grid(~sex)+
  geom_smooth(method="loess")+
  ylim (0,1)

##the impact of fare 
ggplot(titanic,aes(x=fare,y=survived))+
  geom_smooth(method="loess")+
  ylim (0,1)

##investing for outliers
ggplot(titanic,aes(fare,fill=fare))+
  geom_histogram()

ggplot(titanic,aes(fare))+
  geom_histogram()+
  facet_grid(~pclass)


##data cleaning and imputation

##convert the survived variable to factor
titanic$survived<-as.factor(titanic$survived)

##Checking for NAs
colSums(is.na(titanic))
##pattern of NAs
md.pattern(titanic)

##Let's look at the observation of the one missing value on fare.
titanic[which(is.na(titanic$fare)),]

##Analysis of Variance between the fare and class of Passengers
summary(aov(fare~pclass,data=titanic))

##Tukey test to see where the difference is
TukeyHSD(aov(fare~pclass,data=titanic))

##The difference between each class is statistically significant, so the third class fare's median will be used to fill in the fare's missing value.
titanic$fare[which(is.na(titanic$fare))]<-median(titanic$fare[titanic$pclass=="3rd"],na.rm = TRUE)

##We'll now divide the data into training and testing data and impute the missing age values to each of them separately.


##Age imputation
plot(summary(is.na(age)~sex+pclass+survived+parch+sibsp,data=titanic))
age.na<-glm(is.na(age)~sex+pclass+survived+parch+sibsp,data=titanic)
summary(age.na)

##make it reproducible
set.seed(113)
#use 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(titanic), replace=TRUE, prob=c(0.8,0.2))
train.titanic  <- titanic[sample, ]
test.titanic   <- titanic[!sample, ]

##imputing age into the training set
set.seed(113)
imputed_data<-mice(train.titanic,m=5,maxit = 50,method = 'rf')

mean(titanic$age,na.rm = TRUE)    ##the original data's mean value
lapply(imputed_data$imp$age,mean) ##each imputed set's mean
sd(titanic$age,na.rm = TRUE)      ##the original data's standard deviation value 
lapply(imputed_data$imp$age,sd)   ##each imputed set's standard deviation 

titanic.train.complete<-complete(imputed_data,1)

##assign the survived variable from the test.titanic to to "actual"
actual<-test.titanic$survived
##drop it from the dataset
test.titanic<-select(test.titanic,-survived)

##imputing age into the testing data
set.seed(113)
imputed_data.test<-mice(test.titanic,m=5,maxit = 50,method = 'rf')

mean(titanic$age,na.rm = TRUE)    ##the original data's mean value
lapply(imputed_data.test$imp$age,mean) ##each imputed set's mean
sd(titanic$age,na.rm = TRUE)      ##the original data's standard deviation value 
lapply(imputed_data.test$imp$age,sd)  ####each imputed set's standard deviation 

titanic.test.complete<-complete(imputed_data.test,4)

##Change the sibsp and sibsp variables to factor
titanic.train.complete$sibsp<-as.factor(titanic.train.complete$sibsp)
titanic.train.complete$parch<-as.factor(titanic.train.complete$parch)
titanic.test.complete$sibsp<-as.factor(titanic.test.complete$sibsp)
titanic.test.complete$parch<-as.factor(titanic.test.complete$parch)

##Fitting and testing the model

##Let's look at how each variable impacts the probability of surviving once more before we create our model.
plot(summary(survived~sex+pclass+age+parch+sibsp,data=titanic3))


dd <- datadist (titanic.train.complete)
options(datadist = 'dd')

##our first model
fit1<-lrm(survived~age+sex+pclass+sibsp+parch+fare,data = titanic.train.complete)
print(fit1)

##The second model
fit2<-lrm(survived~pclass+sibsp+sex*rcs(age,3)+parch+fare,data=titanic.train.complete)
print(fit2)

##the third model
fit3<-lrm(survived~sex*rcs(age,5)+sibsp+pclass*sex+fare,data=titanic.train.complete)
print(fit3)

##the fourth
fit4<-lrm(survived~sex*rcs(age,5)+sibsp+pclass*sex,data=titanic.train.complete)
print(fit4)

anova(fit4)

##let's take a look how the model work at sibsp is zero
ggplot(Predict(fit4, age , sex , pclass, sibsp=0 , fun=plogis))

##sibsp are two
ggplot(Predict(fit4, age , sex , pclass, sibsp=0 , fun=plogis))

##It's time to predict the survived of the remaining 20%.

test<-predict(fit4,titanic.test.complete)   ##apply the model to the test data
test<-ifelse(test4>0.5,1,0)                 ##convert the probability value to survive or not.
confusionMatrix(table(test4,actual))        ##compare the model's output to the actual data

