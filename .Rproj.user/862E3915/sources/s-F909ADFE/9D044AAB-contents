"
Contributor: https://www.kaggle.com/meepbobeep/intro-to-regression-and-classification-in-r
"

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(corrgram) # Correlograms http://www.datavis.ca/papers/corrgram.pdf
library(lattice) #required for nearest neighbors
library(FNN) # nearest neighbors techniques
library(pROC) # to make ROC curve

wine_data = read.table("winequality-red.csv",sep=",",header=T) # read in the data in a dataframe
head(wine_data) #show the first few rows of data


"
What is in the data?
In this data set, we have a variety of numerical categories, 
and the result at the end is a quality score.

I used the head() function to show the first few rows of the table.

We can see each observation is a row, and there are 12 numerical data elements:

fixed.acidity
volatile.acidity
citric.acid
residual.sugar
chlorides
free.sulfur.dioxide
total.sulfur.dioxide
density
pH
sulphates
alcohol
quality
We can see the different scales of the data, 
and the first 11 are obviously objective variables involving chemical measurement.

The 12th dimension, quality, is the variable we're trying to predict. Let's get some summary statistics on this.
"



summary(wine_data$quality)
table(wine_data$quality)

"summary() which gives summary statistics of the data.
I see the data run from 3 to 8, 
and it's very interesting that all the quartiles are round numbers...
and that the median and the 3rd quartile (aka the 50th and 75th percentiles) are the same.

So then I used table which gives me a better idea of what the data look like: 
it's whole numbers from 3 to 8, with a lot of the data heaped about 5 and 6

So there are some ways we can try to predict these scores:

1. regression techniques which will output numbers
2. classification techniques, where each quality score is considered a class or category

I will show a few approaches to each problem, as well as considerations when we're evaluating how these perform.

###########################################
Regression Approaches to Quality Score
In the following, I will try:

1. linear regression
2. generalized linear regression
3. k-nearest neighbors approach
##########################################


######################

Linear regression
Let's just do a linear regression for quality, using all the other variables as input.

########################"

linear_quality = lm(quality ~ fixed.acidity +
                      volatile.acidity+
                      citric.acid+
                      residual.sugar+
                      chlorides+
                      free.sulfur.dioxide+
                      total.sulfur.dioxide+
                      density+pH+sulphates+
                      alcohol, 
                    data=wine_data)
summary(linear_quality)


"
NOTE:

Some of these items... it's hard to tell how well they work together. 
Just because something has a very small significance value, may not mean it really is significant.
Perhaps it would be helpful to visualize the data so we can better decide the best input variables to use.

Visualizing variable relationships
Below, I use a correlogram, which reflects pairwise correlations.

In the lower left half of the display, hcolor saturation indicates strength of correlation and the colors indicate direction of correlation (red=negative, blue=positive...for the colorblind, we also get white lines indicating direction).

In the upper right half of the display, we see an ellipse and a red line. The red line is showing a smoothed line indicating relationship between the two variables, and the ellipse provide the confidence intervals.

A quick start on correlograms can be seen at this link: https://www.statmethods.net/advgraphs/correlograms.html
"

install.packages("corrgram")
library(corrgram)
corrgram(wine_data, lower.panel=panel.shade, upper.panel=panel.ellipse)

"
Interpreting the correlogram

We're trying to predict quality, so we care about the final row and column 
in order to choose our variables. 
We may wish to look elsewhere in the matrix, too, 
to make sure that we're not picking independent variables that are quite correlated: 
for example, the free.sulphur.dioxide and total.sulfur.dioxide have extremely strong correlation, 
so we would not want to use both variables in a simplified model.

Our first try was using all the variables, so let's use fewer for our next two tries.
"

"
Try 2: Single-variable linear regression
Why don't we start with just the variable with the strongest correlation: alcohol.
"

linear_quality_1 = lm(quality ~ alcohol, data = wine_data)
summary(linear_quality_1)

"

Unsurprisingly, our residuals are worse than using all the variables. 
The more variables, the better the fit 
- but it doesn't mean it is necessarily a better model

Try 3: Four-variable linear regression¶
I will build one more linear model, using the variables: alcohol, volatile.acidity, citric.acid, and sulphates.
"

linear_quality_4 = lm(quality ~ alcohol + volatile.acidity + citric.acid + sulphates, data = wine_data)
summary(linear_quality_4)


"
Visualizing the fits¶
One of the things we're interested in is how well these models do against each other.

For each of these fits, we can get the residuals, 
and we can plat these against some dimension. 
I'm simply going to plot against alcohol, 
because that's the main variable I used to fit for the 1-dimension regression.

"

linear_quality.res = resid(linear_quality) # gets residuals
linear_quality_1.res = resid(linear_quality_1)
linear_quality_4.res = resid(linear_quality_4)

plot(wine_data$alcohol, linear_quality.res) # plot residuals against alcohol variable
points(wine_data$alcohol, linear_quality_1.res, col="red") # add the residuals for 1-dimension
points(wine_data$alcohol, linear_quality_4.res, col="blue") # add residuals for 4 dimension

"
Generalized linear model
################################################

Let's try the 1-dimension alcohol variable again for regression, 
and try different families for the regression.
"

glm_quality_1 = glm(quality~alcohol, data=wine_data, family=gaussian(link="identity"))
summary(glm_quality_1)

"
Okay, I was being tricky with that first one 
-- that was exactly the same as doing regular linear regression.

Let's try a few others.

"

glm_quality_2 = glm(quality~alcohol, data=wine_data, family=gaussian(link="log"))
summary(glm_quality_2)


glm_quality_3 = glm(quality~alcohol+sulphates,data=wine_data,family=poisson(link="identity"))
summary(glm_quality_3)

glm_quality_1.res = resid(glm_quality_1) # gets residuals
glm_quality_2.res = resid(glm_quality_2)
glm_quality_3.res = resid(glm_quality_3)

plot(wine_data$alcohol, glm_quality_1.res) # plot residuals against alcohol variable
points(wine_data$alcohol, glm_quality_2.res, col="red") # add the residuals for 1-dimension
points(wine_data$alcohol, glm_quality_3.res, col="blue") # add res

"
Check out those blue dots! Those residuals look far better than our first two attempts.

Let's actually see what the data look like versus the fitted models.
"

plot(wine_data$alcohol,wine_data$quality)
points(wine_data$alcohol,predict(glm_quality_3,wine_data),col="blue")
points(wine_data$alcohol,predict(glm_quality_1,wine_data),col="red")

"Finally, let's try k-nearest neighbors regressions."
install.packages("FNN")
library(FNN) # nearest neighbors techniques

knn10 = knn.reg(train=wine_data[,1:11], test=wine_data[,1:11], y=wine_data$quality, k =10) 
knn20 = knn.reg(train=wine_data[,1:11],test=wine_data[,1:11], y = wine_data$quality, k=20)
plot(wine_data$alcohol,wine_data$quality)
points(wine_data$alcohol,knn10$pred,col="red")
points(wine_data$alcohol,knn20$pred,col="blue")

"
Check out those blue dots! Those residuals look far better than our first two attempts.

Let's actually see what the data look like versus the fitted models.
"

######################################### Classification

"
Classification Approaches to Quality Score

I wasn't very happy with the results we're getting from the regression approaches 
- the quality scores are whole numbers, and regression will give us continuous results.

To simplify the matter, I'm going to make for a classification of wine - poor, okay, and good. 
Poor will be those with scores of either 3 or 4; 
okay will be with scores of 5 or 6; 
good will be with scores 7 or 8.

I'm going to make three extra columns that will be either 0 or 1 
depending on whether it falls into that classification.
"

unique(wine_data$quality )

wine_data$poor <- wine_data$quality <= 4
wine_data$okay <- wine_data$quality == 5 | wine_data$quality == 6
wine_data$good <- wine_data$quality >= 7
head(wine_data)
summary(wine_data)


"
We will try 3 different approaches to classification:

Logistic function fit
K-nearest neighbors
Decision trees
For logistic fit, we can check out the ROC curve.

Logistic function fit
The idea here is to fit a logistic function, and for this, 
I will use two models - one using just the alcohol level; one using 4 variables.

For right now, let's just say I'm really interested in finding good wine more than anything else, 
so all my classification attempts will be for the ones falling in the good category. 
I will likely come back at a later time and extend the classification to the other items.

I put up the AUC (area under the curve) for each graph.
"

log1_good = glm(good~alcohol, data=wine_data, family=binomial(link="logit"))
log2_good = glm(good~alcohol + volatile.acidity + citric.acid + sulphates,data=wine_data,family=binomial(link="logit"))

summary(log1_good)
summary(log2_good)

Slog1_good <- pnorm(predict(log1_good))
Slog2_good <- pnorm(predict(log2_good))

install.packages("pROC")
library(pROC) # to make ROC curve
roc1 <- plot.roc(wine_data$good,Slog1_good,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc1.se <- ci.se(roc1,specificities=seq(0,100,5))
plot(roc1.se,type="shape", col="grey")

roc2 <-plot.roc(wine_data$good,Slog2_good,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc2.se <- ci.se(roc2,specificities=seq(0,100,5))
plot(roc2.se,type="shape", col="blue")


"

K nearest neighbors classification
####################################

Let's try k-nearest neighbors now.

Again, I'll be focusing solely on the good category.

For each of the k-nearest neighbors fits, I take a look at what's called the confusion matrix. 
Unfortunately, I have a lot of false negatives, meaning I will miss a lot of good wines.
"

class_knn10 = knn(train=wine_data[,1:11], test=wine_data[,1:11], cl=wine_data$good, k =10) 
class_knn20 = knn(train=wine_data[,1:11],test=wine_data[,1:11], cl = wine_data$good, k=20)
table(wine_data$good,class_knn10)
table(wine_data$good,class_knn20)

"
Decision trees¶
####################################

Our final classification approach will be decision trees.

Again, I will focus only on finding the good wines.
"
install.packages("rpart")
library(rpart) #for trees
tree1 <- rpart(good ~ alcohol + sulphates, data = wine_data, method="class")
summary(tree1)

install.packages("rpart.plot")
library(rpart.plot) # plotting trees
library(caret)
rpart.plot(tree1)
pred1 <- predict(tree1,newdata=wine_data,type="class")


tree2 <- rpart(good ~ alcohol + volatile.acidity + citric.acid + sulphates, data = wine_data, method="class")
summary(tree2)
rpart.plot(tree2)
pred2 <- predict(tree2,newdata=wine_data,type="class")


"
Let's compare the confusion matrices of the two trees.
So we ended up with fewer false negatives with the second, but more false positives. Not too shabby.

"

table(wine_data$good,pred1)
table(wine_data$good,pred2)