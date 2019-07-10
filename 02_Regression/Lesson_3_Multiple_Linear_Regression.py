'''
Introduction to Multiple Linear Regression

Linear regression is useful when we want to predict the values of a variable from its relationship with other variables. There are two different types of linear regression models (simple linear regression and multiple linear regression).

In predicting the price of a home, one factor to consider is the size of the home. The relationship between those two variables, price and size, is important, but there are other variables that factor in to pricing a home: location, air quality, demographics, parking, and more. When making predictions for price, our dependent variable, we'll want to use multiple independent variables. To do this, we'll use Multiple Linear Regression.

Multiple Linear Regression uses two or more independent variables to predict the values of the dependent variable. It is based on the following equation that we'll explore later on:

y=b+m1x1+m2x2+...+mnxny = b + m_{1}x_{1} + m_{2}x_{2} + ... + m_{n}x_{n}y=b+m1​x1​+m2​x2​+...+mn​xn​

StreetEasy Dataset

You'll learn multiple linear regression by performing it on this dataset. It contains information about apartments in New York.
Instructions
1.

Before we start digging into the StreetEasy data, add this line at the end of script.py:

plt.show()

And then press run to see the graph!

In this example, we used size (ft²) and building age (years) as independent variables to predict the rent ($).

When we have two independent variables, we can create a linear regression plane. We can now guess what the rent is by plugging in the independent variables and finding where they lie on the plane.

'''
import codecademylib3_seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['size_sqft','building_age_yrs']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

ols = LinearRegression()

ols.fit(x_train, y_train)

# Plot the figure

fig = plt.figure(1, figsize=(6, 4))
plt.clf()

elev = 43.5
azim = -110

ax = Axes3D(fig, elev=elev, azim=azim)

ax.scatter(x_train[['size_sqft']], x_train[['building_age_yrs']], y_train, c='k', marker='+')

ax.plot_surface(np.array([[0, 0], [4500, 4500]]), np.array([[0, 140], [0, 140]]), ols.predict(np.array([[0, 0, 4500, 4500], [0, 140, 0, 140]]).T).reshape((2, 2)), alpha=.7)

ax.set_xlabel('Size (ft$^2$)')
ax.set_ylabel('Building Age (Years)')
ax.set_zlabel('Rent ($)')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# Add the code below:
plt.show()


'''
StreetEasy Dataset

[StreetEasy Logo]

StreetEasy is New York City's leading real estate marketplace — from studios to high-rises, Brooklyn Heights to Harlem.

In this lesson, you will be working with a dataset that contains a sample of 5,000 rentals listings in Manhattan, Brooklyn, and Queens, active on StreetEasy in June 2016.

It has the following columns:

    rental_id: rental ID
    rent: price of rent in dollars
    bedrooms: number of bedrooms
    bathrooms: number of bathrooms
    size_sqft: size in square feet
    min_to_subway: distance from subway station in minutes
    floor: floor number
    building_age_yrs: building's age in years
    no_fee: does it have a broker fee? (0 for fee, 1 for no fee)
    has_roofdeck: does it have a roof deck? (0 for no, 1 for yes)
    has_washer_dryer: does it have washer/dryer in unit? (0/1)
    has_doorman: does it have a doorman? (0/1)
    has_elevator: does it have an elevator? (0/1)
    has_dishwasher: does it have a dishwasher (0/1)
    has_patio: does it have a patio? (0/1)
    has_gym: does the building have a gym? (0/1)
    neighborhood: (ex: Greenpoint)
    borough: (ex: Brooklyn)

More information about this dataset can be found in the StreetEasy Dataset article.

Let's start by doing exploratory data analysis to understand the dataset better. We have broken the dataset for you into:

    manhattan.csv
    brooklyn.csv
    queens.csv

1.

First, pick a borough out of the three (Manhattan, Brooklyn, and Queens) that you are most interested in!

We are going to import the dataset and store it in a variable called streeteasy.

To import, we will need to run this snippet:

pd.read_csv("path")

Replace path with one of the three URL's above.
2.

Create a DataFrame using streeteasy.
3.

Let's take a look at the first few rows using df.head():

    How far is the apartment in the third row from a subway station?
    Which neighborhood is it in?

'''

import codecademylib3_seaborn
import pandas as pd

streeteasy = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/brooklyn.csv")

df = pd.DataFrame(streeteasy)

print(df.head())

'''
Training Set vs. Test Set

As with most machine learning algorithms, we have to split our dataset into:

    Training set: the data used to fit the model
    Test set: the data partitioned away at the very start of the experiment (to provide an unbiased evaluation of the model)

[Training Set vs. Testing Set]

In general, putting 80% of your data in the training set and 20% of your data in the test set is a good place to start.

Suppose you have some values in x and some values in y:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

Here are the parameters:

    train_size: the proportion of the dataset to include in the train split (between 0.0 and 1.0)
    test_size: the proportion of the dataset to include in the test split (between 0.0 and 1.0)
    random_state: the seed used by the random number generator [optional]

To learn more, here is a Training Set vs Validation Set vs Test Set article.
1.

Import train_test_split from sklearn.model_selection.
2.

Create a DataFrame x that selects the following columns from the main df DataFrame:

    'bedrooms'
    'bathrooms'
    'size_sqft'
    'min_to_subway'
    'floor'
    'building_age_yrs'
    'no_fee'
    'has_roofdeck'
    'has_washer_dryer'
    'has_doorman'
    'has_elevator'
    'has_dishwasher'
    'has_patio'
    'has_gym'

Create a DataFrame y that selects the rent column from the main df DataFrame.

These are the columns we want to use for our regression model.
3.

Use scikit-learn's train_test_split() method to split x into 80% training set and 20% testing set and generate:

    x_train
    x_test
    y_train
    y_test

Set the random_state to 6.
4.

Let's take a look at the shapes of x_train, x_test, y_train, and y_test to see we got the proportion we wanted.

We have 14 features that we're looking for for each apartment, and 1 label we're looking for for each apartment.
'''
import codecademylib3_seaborn
import pandas as pd

# import train_test_split
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck','has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym']]
y = df['rent']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=6)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

'''
Multiple Linear Regression: Scikit-Learn

Now we have the training set and the test set, let's use scikit-learn to build the linear regression model!

The steps for multiple linear regression in scikit-learn are identical to the steps for simple linear regression. Just like simple linear regression, we need to import LinearRegression from the linear_model module:

from sklearn.linear_model import LinearRegression

Then, create a LinearRegression model, and then fit it to your x_train and y_train data:

mlr = LinearRegression()

mlr.fit(x_train, y_train) 
# finds the coefficients and the intercept value

We can also use the .predict() function to pass in x-values. It returns the y-values that this plane would predict:

y_predicted = mlr.predict(x_text)
# takes values calculated by `.fit()` and the `x` values, plugs them into the multiple linear regression equation, and calculates the predicted y values.

We will start by using two of these columns to teach you how to predict the values of the dependent variable, prices.
1.

Import LinearRegression from scikit-learn's linear_model module.
2.

Create a Linear Regression model and call it mlr.

Fit the model using x_train and y_train.
3.

Use the model to predict y-values from x_test. Store the predictions in a variable called y_predict.

Now we have:

    x_test
    x_train
    y_test
    y_train
    and y_predict!

4.

To see this model in action, let's test it on Sonny's apartment in Greenpoint, Brooklyn!

Or if you reside in New York, plug in your own apartment's values and see if you are over or underpaying!
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# Add the code here:
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()

mlr.fit(x_train, y_train)
# finds the coefficients and the intercept value

y_predict = mlr.predict(x_test)
# takes values calculated by `.fit()` and the `x` values, plugs them into the multiple linear regression equation, and calculates the predicted y values.

'''
Visualizing Results with Matplotlib

You've performed Multiple Linear Regression, and you also have the predictions in y_predict. However, we don't have insight into the data, yet. In this exercise, you'll create a 2D scatterplot to see how the independent variables impact prices.

How do you create 2D graphs?

Graphs can be created using Matplotlib's pyplot module. Here is the code with inline comments explaining how to plot using Matplotlib's .scatter():

# Create a scatter plot
plt.scatter(x, y, alpha=0.4)

# Create x-axis label and y-axis label
plt.xlabel("the x-axis label")
plt.ylabel("the y-axis label")

# Create a title
plt.title("title!")

# Show the plot
plt.show()

We want to create a scatterplot like this:

[Visualization]
Instructions
1.

Create a 2D scatter plot using y_test and y_predict.

The x-axis should represent actual rent prices and the y-axis should represent predicted rent prices.
2.

Add appropriate x-axis labels and y-axis labels, as well as a title.
3.

Show the plot using plt.show().
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model=lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)
plt.scatter(y_test,y_predict)
plt.xlabel('Actual Rent [$]')
plt.ylabel('Predicted Rent [$]')
plt.title('Actual vs Predicted Rent')
plt.show()


'''
Multiple Linear Regression Equation

Now that we have implemented Multiple Linear Regression, we will learn how to tune and evaluate the model. Before we do that, however, it's essential to learn the equation behind it.

Equation 6.1 The equation for multiple linear regression that uses two independent variables is this:

y=b+m1x1+m2x2


Equation 6.2 The equation for multiple linear regression that uses three independent variables is this:

y=b+m1x1+m2x2+m3x3

Equation 6.3 As a result, since multiple linear regression can use any number of independent variables, its general equation becomes:

y=b+m1x1+m2x2+...+mnxn

Here, m1, m2, m3, ... mn refer to the coefficients, and b refers to the intercept that you want to find. You can plug these values back into the equation to compute the predicted y values.

Remember, with sklearn's LinearRegression() method, we can get these values with ease.

The .fit() method gives the model two variables that are useful to us:

    .coef_, which contains the coefficients
    .intercept_, which contains the intercept

After performing multiple linear regression, you can print the coefficients using .coef_.

Coefficients are most helpful in determining which independent variable carries more weight. For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, with the former impacting prices negatively and latter positively.
Instructions
1.

Print out the coefficients using .coef_.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:
print(model.coef_)
print(model.intercept_)


'''
Correlations

In our Manhattan model, we used 14 variables, so there are 14 coefficients:

[ -302.73009383  1199.3859951  4.79976742  -24.28993151  24.19824177  -7.58272473  -140.90664773  48.85017415  191.4257324  -151.11453388  89.408889  -57.89714551  -19.31948556  -38.92369828 ]]

    bedrooms - number of bedrooms
    bathrooms - number of bathrooms
    size_sqft - size in square feet
    min_to_subway - distance from subway station in minutes
    floor - floor number
    building_age_yrs - building's age in years
    no_fee - has no broker fee (0 for fee, 1 for no fee)
    has_roofdeck - has roof deck (0 for no, 1 for yes)
    has_washer_dryer - has in-unit washer/dryer (0/1)
    has_doorman - has doorman (0/1)
    has_elevator - has elevator (0/1)
    has_dishwasher - has dishwasher (0/1)
    has_patio - has patio (0/1)
    has_gym - has gym (0/1)

To see if there are any features that don't affect price linearly, let's graph the different features against rent.

Interpreting graphs

In regression, the independent variables will either have a positive linear relationship to the dependent variable, a negative linear relationship, or no relationship. A negative linear relationship means that as X values increase, Y values will decrease. Similarly, a positive linear relationship means that as X values increase, Y values will also increase.

Graphically, when you see a downward trend, it means a negative linear relationship exists. When you find an upward trend, it indicates a positive linear relationship. Here are two graphs indicating positive and negative linear relationships:

Positive and Negative Linear Relationships
Instructions
1.

Create a scatterplot of size_sqft and rent:

plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)

Is there a strong correlation?
2.

Create a scatterplot of min_to_subway and rent:

plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)

Is there a strong correlation?
3.

Do the same for a few others and write down the ones that don't have strong correlations.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

# Input code here:
#plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
#plt.show() # positive correlation

#plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)
#plt.show() # no correlation?

plt.scatter(df[['floor']], df[['rent']], alpha=0.4)
plt.show()

'''
Evaluating the Model's Accuracy

When trying to evaluate the accuracy of our multiple linear regression model, one technique we can use is Residual Analysis.

The difference between the actual value y, and the predicted value ŷ is the residual e. The equation is:

e=y−ŷ

In the StreetEasy dataset, y is the actual rent and the ŷ is the predicted rent. The real y values should be pretty close to these predicted y values.

sklearn's linear_model.LinearRegression comes with a .score() method that returns the coefficient of determination R² of the prediction.

The coefficient R² is defined as:

1 − (u/v)

where u is the residual sum of squares:

((y - y_predict) ** 2).sum()

and v is the total sum of squares (TSS):

((y - y.mean()) ** 2).sum()

The TSS tells you how much variation there is in the y variable.

R² is the percentage variation in y explained by all the x variables together.

For example, say we are trying to predict rent based on the size_sqft and the bedrooms in the apartment and the R² for our model is 0.72 — that means that all the x variables (square feet and number of bedrooms) together explain 72% variation in y (rent).

Now let's say we add another x variable, building's age, to our model. By adding this third relevant x variable, the R² is expected to go up. Let say the new R² is 0.95. This means that square feet, number of bedrooms and age of the building together explain 95% of the variation in the rent.

The best possible R² is 1.00 (and it can be negative because the model can be arbitrarily worse). Usually, a R² of 0.70 is considered good.
Instructions
1.

Use the .score() method from LinearRegression to find the mean squared error regression loss for the training set.

Write that number down.
2.

Use the .score() method from LinearRegression to find the mean squared error regression loss for the testing set.

Write that number down.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:
# Use the .score() method from LinearRegression to find the mean squared error regression loss for the training set.
print("Train score:")
print(model.score(x_train, y_train))  # 0.772546055982

print("Test score:")
print(model.score(x_test, y_test))  # 0.805037197536

'''
Rebuild the Model

Now let's rebuild the model using the new features as well as evaluate the new model to see if we improved!

For Manhattan, the scores returned:

Train score: 0.772546055982
Test score:  0.805037197536

For Brooklyn, the scores returned:

Train score: 0.613221453798
Test score:  0.584349923873

For Queens, the scores returned:

Train score: 0.665836031009
Test score:  0.665170319781

For whichever borough you used, let's see if we can improve these scores!
Instructions
1.

Print the coefficients again to see which ones are strongest.
2.

Currently the x should look something like:

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

Remove some of the features that don't have strong correlations and see if your scores improved!

Post your best model in the Slack channel!
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

# x = df[['bedrooms', 'bathrooms', 'size_sqft', 'building_age_yrs']]  # my best model 0.808 test score

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()

zoe_apartment = [[1, 1, 620, 16, 1, 98, 0, 0, 1, 0, 0, 0, 1, 0]]
predict = model.predict(zoe_apartment)
print("Predicted rent: $%.2f" % predict)

'''
Review

Great work! Let’s review the concepts before you move on:

    Multiple Linear Regression uses two or more variables to make predictions about another variable:

y=b+m1x1+m2x2+...+mnxn

    Multiple linear regression uses a set of independent variables and a dependent variable. It uses these variables to learn how to find optimal parameters. It takes a labeled dataset and learns from it. Once we confirm that it's learned correctly, we can then use it to make predictions by plugging in new x values.
    We can use scikit-learn's LinearRegression() to perform multiple linear regression.
    Residual Analysis is used to evaluate the regression model's accuracy. In other words, it's used to see if the model has learned the coefficients correctly.
    Scikit-learn's linear_model.LinearRegression comes with a .score() method that returns the coefficient of determination R² of the prediction. The best score is 1.0.

'''

