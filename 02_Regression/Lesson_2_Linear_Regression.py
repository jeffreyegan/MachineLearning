'''
Linear Regression
About 2 hours

Given a set of data points, find a line that fits the data best! This is the simplest form of regression, and allows us to make predictions of future points.

'''

'''
Introduction to Linear Regression

The purpose of machine learning is often to create a model that explains some real-world data, so that we can predict what may happen next, with different inputs.

The simplest model that we can fit to data is a line. When we are trying to find a line that fits a set of data best, we are performing Linear Regression.

We often want to find lines to fit data, so that we can predict unknowns. For example:

    The market price of a house vs. the square footage of a house. Can we predict how much a house will sell for, given its size?
    The tax rate of a country vs. its GDP. Can we predict taxation based on a country's GDP?
    The amount of chips left in the bag vs. number of chips taken. Can we predict how much longer this bag of chips will last, given how much people at this party have been eating?

Imagine that we had this set of weights plotted against heights of a large set of professional baseball players:

heights vs weights

To create a linear model to explain this data, we might draw this line:

heights vs weights

Now, if we wanted to estimate the weight of a player with a height of 73 inches, we could estimate that it is around 143 pounds.

A line is a rough approximation, but it allows us the ability to explain and predict variables that have a linear relationship with each other. In the rest of the lesson, we will learn how to perform Linear Regression.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

plt.plot(months, revenue, "o")

plt.title("Sandra's Lemonade Stand")

plt.xlabel("Months")
plt.ylabel("Revenue ($)")

plt.show()

# What do you think the revenue in month 13 would be?
month_13 = 200

'''
Points and Lines

In the last exercise, you were probably able to make a rough estimate about the next data point for Sandra's lemonade stand without thinking too hard about it. For our program to make the same level of guess, we have to determine what a line would look like through those data points.

A line is determined by its slope and its intercept. In other words, for each point y on a line we can say:

y=mx+by = m x + by=mx+b

where m is the slope, and b is the intercept. y is a given point on the y-axis, and it corresponds to a given x on the x-axis.

The slope is a measure of how steep the line is, while the intercept is a measure of where the line hits the y-axis.

When we perform Linear Regression, the goal is to get the "best" m and b for our data. We will determine what "best" means in the next exercises.
Instructions
1.

We have provided a slope, m, and an intercept, b, that seems to describe the revenue data you have been given.

Create a new list, y, that has every element in months, multiplied by m and added to b.

A list comprehension is probably the easiest way to do this!
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#slope:
m = 8
#intercept:
b = 40

plt.plot(months, revenue, "o")


y = [month*m + b for month in months]

plt.plot(months, y, "-")

plt.show()

'''
Loss

When we think about how we can assign a slope and intercept to fit a set of points, we have to define what the best fit is.

For each data point, we calculate loss, a number that measures how bad the model's (in this case, the line's) prediction was. You may have seen this being referred to as error.

We can think about loss as the squared distance from the point to the line. We do the squared distance (instead of just the distance) so that points above and below the line both contribute to total loss in the same way:

In this example:

    For point A, the squared distance is 9 (3²)
    For point B, the squared distance is 1 (1²)

So the total loss, with this model, is 10. If we found a line that had less loss than 10, that line would be a better model for this data.
Instructions
1.

We have three points, (1, 5), (2, 1), and (3, 3). We are trying to find a line that produces lowest loss.

We have provided you the list of x-values, x, and y-values, y, for these points.

Find the y-values that the line with weights m1 and b1 would predict for the x-values given. Store these in a list called y_predicted1.
'''
x = [1, 2, 3]
y = [5, 1, 3]

#y = x
m1 = 1
b1 = 0

#y = 0.5x + 1
m2 = 0.5
b2 = 1

y_predicted1 = [xval * m1 + b1 for xval in x]
y_predicted2 = [xval * m2 + b2 for xval in x]

total_loss1 = 0
total_loss2 = 0

for idx in range(len(y)):
  total_loss1 += (y[idx]-y_predicted1[idx])**2
  total_loss2 += (y[idx]-y_predicted2[idx])**2

print(total_loss1)
print(total_loss2)
better_fit = 2

'''
Minimizing Loss

The goal of a linear regression model is to find the slope and intercept pair that minimizes loss on average across all of the data.

The interactive visualization in the browser lets you try to find the line of best fit for a random set of data points:

    The slider on the left controls the m (slope)
    The slider on the right controls the b (intercept)

You can see the total loss on the right side of the visualization. To get the line of best fit, we want this loss to be as small as possible.

To check if you got the best line, check the "Plot Best-Fit" box.

Randomize a new set of points and try to fit a new line by entering the number of points you want (try 8!) in the textbox and pressing Randomize Points.
Instructions

Play with the interactive visualization in the browser.

Try to notice what method you, as a human and not a computer (hopefully), use to minimize loss:

    Do you first get the slope to where it produces lowest loss, and then move the intercept to where it produces lowest loss?
    Do you create a rough idea in your mind where the line should be first, and then enter the parameters to match that image?


'''


'''
Gradient Descent for Intercept

As we try to minimize loss, we take each parameter we are changing, and move it as long as we are decreasing loss. It's like we are moving down a hill, and stop once we reach the bottom:

The process by which we do this is called gradient descent. We move in the direction that decreases our loss the most. Gradient refers to the slope of the curve at any point.

For example, let's say we are trying to find the intercept for a line. We currently have a guess of 10 for the intercept. At the point of 10 on the curve, the slope is downward. Therefore, if we increase the intercept, we should be lowering the loss. So we follow the gradient downwards.

We derive these gradients using calculus. It is not crucial to understand how we arrive at the gradient equation. To find the gradient of loss as intercept changes, the formula comes out to be:

2/N* ∑ −(yi−(mxi+b)) for i=1,N


    N is the number of points we have in our dataset
    m is the current gradient guess
    b is the current intercept guess

Basically:

    we find the sum of y_value - (m*x_value + b) for all the y_values and x_values we have
    and then we multiply the sum by a factor of -2/N. N is the number of points we have.

Instructions
1.

Define a function called get_gradient_at_b() that takes in a set of x values, x, a set of y values, y, a slope m, and an intercept value b.

For now, have it return b, unchanged.
2.

In the get_gradient_at_b() function, we want to go through all of the x values and all of the y values and compute (y - (m*x+b)) for each of them.

Create a variable called diff that has the sum of all of these values.

Instead of returning b from the get_gradient_at_b() function, return diff.
3.

Still in the get_gradient_at_b() function, define a variable called b_gradient and set it equal to the -2/N multiplied by diff.

Note: N is the number of points, i.e. the length of the x list or the y list.

Instead of returning diff, return b_gradient.


'''


def get_gradient_at_b(x, y, m, b):
    diff = 0
    N = len(x)
    for idx in range(N):
        diff += (y[idx] - (m * x[idx] + b))
    b_gradient = -2 / N * diff
    return b_gradient

'''
Gradient Descent for Slope

We have a function to find the gradient of b at every point. To find the m gradient, or the way the loss changes as the slope of our line changes, we can use this formula:

2/N * ∑ −xi(yi−(mxi+b)) for i=1,N


Once more:

    N is the number of points you have in your dataset
    m is the current gradient guess
    b is the current intercept guess

To find the m gradient:

    we find the sum of x_value * (y_value - (m*x_value + b)) for all the y_values and x_values we have
    and then we multiply the sum by a factor of -2/N. N is the number of points we have.

Once we have a way to calculate both the m gradient and the b gradient, we'll be able to follow both of those gradients downwards to the point of lowest loss for both the m value and the b value. Then, we'll have the best m and the best b to fit our data!
1.

Define a function called get_gradient_at_m() that takes in a set of x values, x, a set of y values, y, a slope m, and an intercept value b.

For now, have it return m.
2.

In this function, we want to go through all of the x values and all of the y values and compute x*(y - (m*x+b)) for each of them.

Create a variable called diff that has the sum of all of these values, and return it from the function.
3.

Define a variable called m_gradient and set it equal to the -2/N multiplied by diff.

Instead of returning diff, return m_gradient.

'''


def get_gradient_at_b(x, y, m, b):
    diff = 0
    N = len(x)
    for i in range(N):
        y_val = y[i]
        x_val = x[i]
        diff += (y_val - ((m * x_val) + b))
    b_gradient = -2 / N * diff
    return b_gradient


def get_gradient_at_m(x, y, m, b):
    diff = 0
    N = len(x)
    for i in range(N):
        y_val = y[i]
        x_val = x[i]
        diff += x_val * (y_val - ((m * x_val) + b))
    m_gradient = -2 / N * diff
    return m_gradient

'''
Put it Together

Now that we know how to calculate the gradient, we want to take a "step" in that direction. However, it's important to think about whether that step is too big or too small. We don't want to overshoot the minimum error!

We can scale the size of the step by multiplying the gradient by a learning rate.

To find a new b value, we would say:

new_b = current_b - (learning_rate * b_gradient)

where current_b is our guess for what the b value is, b_gradient is the gradient of the loss curve at our current guess, and learning_rate is proportional to the size of the step we want to take.

In a few exercises, we'll talk about the implications of a large or small learning rate, but for now, let's use a fairly small value.
1.

Define a function called step_gradient() that takes in x, y, b_current, and m_current.

This function will find the gradients at b and m_current, and then return new b and m values that have been moved in that direction.

For now, just return the pair (b_current, m_current).
2.

Inside step_gradient(), find the gradient at b and the gradient at m using the functions defined before (get_gradient_at_b and get_gradient_at_m).

Store these gradients in variables called b_gradient and m_gradient, and return these from the function instead of b_current and m_current.

Return them as a list.
3.

Let's try to move the parameter values in the direction of the gradient at a rate of 0.01.

Create variables called b and m:

    b should be b_current - (0.01 * b_gradient)
    m should be m_current - (0.01 * m_gradient)

Return the pair b and m from the function.
4.

We have provided Sandra's lemonade data once more. We have a guess for what we think the b and m might be.

Call your function to perform one step of gradient descent. Store the results in the variables b and m.

Great! We have a way to step to new b and m values! Next, we will call this function a bunch, in order to move those values towards lower and lower loss.
'''
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff
  return m_gradient

# Define your step_gradient function here
def step_gradient(x,y,b_current,m_current):
  b_gradient = get_gradient_at_b(x, y, b_current, m_current)
  m_gradient = get_gradient_at_m(x, y, b_current, m_current)
  b = b_current - (0.01 * b_gradient)
  m = m_current - (0.01 * m_gradient)
  return[b, m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

# current intercept guess:
b = 0
# current slope guess:
m = 0

# Call your function here to update b and m
b, m = step_gradient(months,revenue,b,m)
print(b, m)


'''
Convergence

How do we know when we should stop changing the parameters m and b? How will we know when our program has learned enough?

To answer this, we have to define convergence. Convergence is when the loss stops changing (or changes very slowly) when parameters are changed.

Hopefully, the algorithm will converge at the best values for the parameters m and b.
Instructions
1.

Run the file. Look at the graph. This graph shows how much the parameter b changed with each iteration of a gradient descent runner.
2.

How many iterations did it take for this program to converge?

Enter your answer in a variable called num_iterations.
3.

At what b value did this program converge?

Enter your answer in a variable called convergence_b.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01

iterations = range(1400)

plt.plot(iterations, bs)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

num_iterations = 800
convergence_b = 47


'''
Learning Rate

We want our program to be able to iteratively learn what the best m and b values are. So for each m and b pair that we guess, we want to move them in the direction of the gradients we've calculated. But how far do we move in that direction?

We have to choose a learning rate, which will determine how far down the loss curve we go.

A small learning rate will take a long time to converge — you might run out of time or cycles before getting an answer. A large learning rate might skip over the best value. It might never converge! Oh no!

Finding the absolute best learning rate is not necessary for training a model. You just have to find a learning rate large enough that gradient descent converges with the efficiency you need, and not so large that convergence never happens.
Instructions
1.

We have imported two new lists representing how the b value changed with different learning rates:

    bs_000000001: 1400 iterations of gradient descent on b with a learning rate of 0.000000001
    bs_01: 100 iterations of gradient descent on b with a learning rate of 0.01

Change the plot to plot bs_000000001 instead of bs.

Does the gradient descent algorithm still converge to the same b value? Does it converge at all? Look at the values on the y-axis!
2.

Change the plot to plot bs_01 instead of bs_000000001. Unfortunately, our computers blew up after 100 iterations of this, so you'll also have to change the number of iterations to 100 instead of 1400:

iterations = range(100)

Does the gradient descent algorithm still converge to the same b value? Does it converge at all?
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01

iterations = range(100)

plt.plot(iterations, bs_01)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

num_iterations = 800
convergence_b = 47

'''
Put it Together II

At each step, we know how to calculate the gradient and move in that direction with a step size proportional to our learning rate. Now, we want to make these steps until we reach convergence.
1.

We have all of the functions we have defined throughout the lesson.

Now, let's create a function called gradient_descent() that takes in x, y, learning_rate, and a num_iterations.

For now, return [-1,-1].
2.

In the function gradient_descent(), create variables b and m and set them both to zero for our initial guess.

Return b and m from the function.
3.

Update your step_gradient() function to take in the parameter learning_rate (as the last parameter) and replace the 0.01s in the calculations of b_gradient and m_gradient with learning_rate.
4.

Let's go back and finish the gradient_descent() function.

Create a loop that runs num_iterations times. At each step, it should:

    Call step_gradient() with b, m, x, y, and learning_rate
    Update the values of b and m with the values step_gradient() returns.

5.

Outside of the function, uncomment the line that calls gradient_descent on months and revenue, with a learning rate of 0.01 and 1000 iterations.

It stores the results in variables called b and m.
6.

Uncomment the lines that will plot the result to the browser.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt


def get_gradient_at_b(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += (y_val - ((m * x_val) + b))
    b_gradient = -(2 / N) * diff
    return b_gradient


def get_gradient_at_m(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += x_val * (y_val - ((m * x_val) + b))
    m_gradient = -(2 / N) * diff
    return m_gradient


# Your step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


# Your gradient_descent function here:
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]


months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

# Uncomment the line below to run your gradient_descent function
b, m = gradient_descent(months, revenue, 0.01, 1000)

# Uncomment the lines below to see the line you've settled upon!
y = [m * x + b for x in months]

plt.plot(months, revenue, "o")
plt.plot(months, y)

plt.show()

'''
Use Your Functions on Real Data

We have constructed a way to find the "best" b and m values using gradient descent! Let's try this on the set of baseball players' heights and weights that we saw at the beginning of the lesson.
1.

Run the code in script.py.

This is a scatterplot of weight vs height.
2.

We have imported your gradient_descent() function. Call it with parameters:

    X
    y
    num_iterations of 1000
    learning_rate of 0.0001

Store the result in variables called b and m.
3.

Create a list called y_predictions. Set it to be every element of X multiplied by m and added to b.

The easiest way to do this would be a list comprehension:

new_y = [element*slope + intercept for element in y]

4.

Plot X vs y_predictions on the same plot as the scatterplot.

Does the line look right?
'''
import codecademylib3_seaborn
from gradient_descent_funcs import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

X = df["height"]
y = df["weight"]

plt.plot(X, y, 'o')
#plot your line here:
b, m = gradient_descent(X,y,0.0001,1000)
y_predictions = [Xval*m+b for Xval in X]

plt.plot(X,y_predictions)
plt.show()

'''
Scikit-Learn

Congratulations! You've now built a linear regression algorithm from scratch.

Luckily, we don't have to do this every time we want to use linear regression. We can use Python's scikit-learn library. Scikit-learn, or sklearn, is used specifically for Machine Learning. Inside the linear_model module, there is a LinearRegression() function we can use:

from sklearn.linear_model import LinearRegression

You can first create a LinearRegression model, and then fit it to your x and y data:

line_fitter = LinearRegression()
line_fitter.fit(X, y)

The .fit() method gives the model two variables that are useful to us:

    the line_fitter.coef_, which contains the slope
    the line_fitter.intercept_, which contains the intercept

We can also use the .predict() function to pass in x-values and receive the y-values that this line would predict:

y_predicted = line_fitter.predict(X)

Note: the num_iterations and the learning_rate that you learned about in your own implementation have default values within scikit-learn, so you don't need to worry about setting them specifically!
1.

We have imported a dataset of soup sales data vs temperature.

Run the code to see the scatterplot. Can you envision the line that would fit this data?
2.

Create an sklearn linear regression model and call it line_fitter.
3.

Fit the line_fitter object to temperature and sales.
4.

Create a list called sales_predict that is the predicted sales values that line_fitter would generate from the temperature list.
5.

Plot sales_predict against temperature as a line, on the same plot as the scatterplot.
'''
import codecademylib3_seaborn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]


line_fitter = LinearRegression()
line_fitter.fit(temperature,sales)

sales_predict = line_fitter.predict(temperature)

plt.plot(temperature, sales, 'o')
plt.plot(temperature, sales_predict, '-')
plt.show()

'''
Review

We have seen how to implement a linear regression algorithm in Python, and how to use the linear regression model from scikit-learn. We learned:

    We can measure how well a line fits by measuring loss.
    The goal of linear regression is to minimize loss.
    To find the line of best fit, we try to find the b value (intercept) and the m value (slope) that minimize loss.
    Convergence refers to when the parameters stop changing with each iteration.
    Learning rate refers to how much the parameters are changed on each iteration.
    We can use Scikit-learn's LinearRegression() model to perform linear regression on a set of points.

These are important tools to have in your toolkit as you continue your exploration of data science.

Find another dataset, maybe in scikit-learn's example datasets. Or on Kaggle, a great resource for tons of interesting data.
https://www.kaggle.com/datasets

Try to perform linear regression on your own! If you find any cool linear correlations, make sure to share them!

As a starter, we've loaded in the Boston housing dataset. We made the X values the nitrogen oxides concentration (parts per 10 million), and the y values the housing prices. See if you can perform regression on these houses!
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Boston housing dataset
boston = load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)

# Set the x-values to the nitrogen oxide concentration:
X = df[['NOX']]
# Y-values are the prices:
y = boston.target

# Can we do linear regression on this?
line_fitter = LinearRegression()
line_fitter.fit(X,y)

y_predict = line_fitter.predict(X)



plt.scatter(X, y, alpha=0.4)
plt.plot(X, y_predict, '-')
# Plot line here:

plt.title("Boston Housing Dataset")
plt.xlabel("Nitric Oxides Concentration")
plt.ylabel("House Price ($)")
plt.show()