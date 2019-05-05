'''
Random Forest

We’ve seen that decision trees can be powerful supervised machine learning models. However, they’re not without their weaknesses — decision trees are often prone to overfitting.

We’ve discussed some strategies to minimize this problem, like pruning, but sometimes that isn’t enough. We need to find another way to generalize our trees. This is where the concept of a random forest comes in handy.

A random forest is an ensemble machine learning technique — a random forest contains many decision trees that all work together to classify new points. When a random forest is asked to classify a new point, the random forest gives that point to each of the decision trees. Each of those trees reports their classification and the random forest returns the most popular classification. It’s like every tree gets a vote, and the most popular classification wins.

Some of the trees in the random forest may be overfit, but by making the prediction based on a large number of trees, overfitting will have less of an impact.

In this lesson, we’ll learn how the trees in a random forest get created.

'''

'''
Bagging

You might be wondering how the trees in the random forest get created. After all, right now, our algorithm for creating a decision tree is deterministic — given a training set, the same tree will be made every time.

Random forests create different trees using a process known as bagging. Every time a decision tree is made, it is created using a different subset of the points in the training set. For example, if our training set had 1000 rows in it, we could make a decision tree by picking 100 of those rows at random to build the tree. This way, every tree is different, but all trees will still be created from a portion of the training data.

One thing to note is that when we’re randomly selecting these 100 rows, we’re doing so with replacement. Picture putting all 100 rows in a bag and reaching in and grabbing one row at random. After writing down what row we picked, we put that row back in our bag.

This means that when we’re picking our 100 random rows, we could pick the same row more than once. In fact, it’s very unlikely, but all 100 randomly picked rows could all be the same row!

Because we’re picking these rows with replacement, there’s no need to shrink our bagged training set from 1000 rows to 100. We can pick 1000 rows at random, and because we can get the same row more than once, we’ll still end up with a unique data set.

Let’s implement bagging! We’ll be using the data set of cars that we used in our decision tree lesson.
Instructions
1.

Start by creating a tree using all of the data we’ve given you. Create a variable named tree and set it equal to the build_tree() function using car_data and car_labels as parameters.

Then call print_tree() using tree as a parameter. Scroll up to the top to see the root of the tree. Which feature is used to split the data at the root?
2.

For now, comment out printing the tree.

Let’s now implement bagging. The original dataset has 1000 items in it. We want to randomly select a subset of those with replacement.

Create a list named indices that contains 1000 random numbers between 0 and 1000. We’ll use this list to remember the 1000 cars and the 1000 labels that we’re going to build a tree with.

You can use either a for loop or list comprehension to make this list. To get a random number between 0 and 1000, use random.randint(0, 999).
3.

Create two new lists named data_subset and labels_subset. These two lists should contain the cars and labels found at each index in indices.

Once again, you can use either a for loop or list comprehension to make these lists.
4.

Create a tree named subset_tree using the build_tree() function with data_subset and labels_subset as parameters.

Print subset_tree using the print_tree() function.

Which feature is used to split the data at the root? Is it a different feature than the feature that split the tree that was created using all of the data?

You’ve just created a new tree from the training set! If you used 1000 different indices, you’d get another different tree. You could now create a random forest by creating multiple different trees!
'''

from tree import build_tree, print_tree, car_data, car_labels
import random
random.seed(4)

#tree = build_tree(car_data, car_labels)
#print_tree(tree)

indices = [random.randint(0, 999) for i in range(1000)]

data_subset = [car_data[index] for index in indices]
labels_subset = [car_labels[index] for index in indices]

subset_tree = build_tree(data_subset, labels_subset)
print_tree(subset_tree)

'''
Bagging Features

We’re now making trees based on different random subsets of our initial dataset. But we can continue to add variety to the ways our trees are created by changing the features that we use.

Recall that for our car data set, the original features were the following:

    The price of the car
    The cost of maintenance
    The number of doors
    The number of people the car can hold
    The size of the trunk
    The safety rating

Right now when we create a decision tree, we look at every one of those features and choose to split the data based on the feature that produces the most information gain. We could change how the tree is created by only allowing a subset of those features to be considered at each split.

For example, when finding which feature to split the data on the first time, we might randomly choose to only consider the price of the car, the number of doors, and the safety rating.

After splitting the data on the best feature from that subset, we’ll likely want to split again. For this next split, we’ll randomly select three features again to consider. This time those features might be the cost of maintenance, the number of doors, and the size of the trunk. We’ll continue this process until the tree is complete.

One question to consider is how to choose the number of features to randomly select. Why did we choose 3 in this example? A good rule of thumb is to randomly select the square root of the total number of features. Our car dataset doesn’t have a lot of features, so in this example, it’s difficult to follow this rule. But if we had a dataset with 25 features, we’d want to randomly select 5 features to consider at every split point.
Instructions
1.

We’ve given you access to the code that finds the best feature to split on. Right now, it considers all possible features. We’re going to want to change that!

For now, let’s see what the best feature to split the dataset is. At the bottom of your code, call find_best_split() using data_subset and labels_subset as parameters and print the results.

This function returns the information gain and the index of the best feature. What was the index?

That index corresponds to the features of our car. For example, if the best feature index to split on was 0, that means we’re splitting on the price of the car.
2.

We now want to modify our find_best_split() function to only consider a subset of the features. We want to pick 3 features without replacement.

The random.choice() function found in Python’s numpy module can help us do this. random.choice() returns a list of values between 0 and the first parameter. The size of the list is determined by the second parameter. And we can choose without replacement by setting replace = False.

For example, the following code would choose ten unique numbers between 0 and 100 (exclusive) and put them in a list.

lst = np.random.choice(100, 10, replace = False)

Inside find_best_split(), create a list named features that contains 3 numbers between 0 and len(dataset[0]).

Instead of looping through feature in range(len(dataset[0])), loop through feature in features.

Now that we’ve implemented feature bagging, what is the best index to use as the split index?
'''

from tree import car_data, car_labels, split, information_gain
import random
import numpy as np
np.random.seed(1)
random.seed(4)

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    #Create features here
    features = np.random.choice(len(dataset[0]), 3, replace=False)
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_gain, best_feature
  
indices = [random.randint(0, 999) for i in range(1000)]

data_subset = [car_data[index] for index in indices]
labels_subset = [car_labels[index] for index in indices]
print(find_best_split(data_subset, labels_subset))

'''
Classify

Now that we can make different decision trees, it’s time to plant a whole forest! Let’s say we make different 8 trees using bagging and feature bagging. We can now take a new unlabeled point, give that point to each tree in the forest, and count the number of times different labels are predicted.

The trees give us their votes and the label that is predicted most often will be our final classification! For example, if we gave our random forest of 8 trees a new data point, we might get the following results:

["vgood", "vgood", "good", "vgood", "acc", "vgood", "good", "vgood"]

Since the most commonly predicted classification was "vgood", this would be the random forest’s final classification.

Let’s write some code that can classify an unlabeled point!
Instructions
1.

At the top of your code, we’ve included a new unlabeled car named unlabeled_point that we want to classify. We’ve also created a tree named subset_tree that was created using bagging and feature bagging.

Let’s see how that tree classifies this point. Print the results of classify() using unlabeled_point and subset_tree as parameters.
2.

That’s the prediction using one tree. Let’s make 20 trees and record the prediction of each one!

Take all of your code between creating indices and the print statement you just wrote and put it in a for loop that happens 20 times.

Above your for loop, create a variable named predictions and set it equal to an empty list. Inside your for loop, instead of printing the prediction, use .append() to add it to predictions.

Finally after your for loop, print predictions.
3.

We now have a list of 20 predictions — let’s find the most common one! You can find the most common element in a list by using this line of code:

max(predictions, key=predictions.count)

Outside of your for loop, store the most common element in a variable named final_prediction and print that variable.
'''
from tree import build_tree, print_tree, car_data, car_labels, classify
from collections import Counter
import random
random.seed(4)

# The features are the price of the car, the cost of maintenance, the number of doors, the number of people the car can hold, the size of the trunk, and the safety rating
unlabeled_point = ['high', 'vhigh', '3', 'more', 'med', 'med']

predictions = []
for i in range(20):
  indices = [random.randint(0, 999) for i in range(1000)]
  data_subset = [car_data[index] for index in indices]
  labels_subset = [car_labels[index] for index in indices]
  subset_tree = build_tree(data_subset, labels_subset)
  predictions.append(classify(unlabeled_point, subset_tree))
final_prediction = max(predictions, key=predictions.count)
print(final_prediction)

'''
Test Set

We’re now able to create a random forest, but how accurate is it compared to a single decision tree? To answer this question we’ve split our data into a training set and test set. By building our models using the training set and testing on every data point in the test set, we can calculate the accuracy of both a single decision tree and a random forest.

We’ve given you code that calculates the accuracy of a single tree. This tree was made without using any of the bagging techniques we just learned. We created the tree by using every row from the training set once and considered every feature when splitting the data rather than a random subset.

Let’s also calculate the accuracy of a random forest and see how it compares!
Instructions
1.

Begin by taking a look at the code we’ve given you. We’ve created a single tree using the training data, looped through every point in the test set, counted the number of points the tree classified correctly and reported the percentage of correctly classified points — this percentage is known as the accuracy of the model.

Run the code to see the accuracy of the single decision tree.
2.

Right below where tree is created, create a random forest named forest using our make_random_forest() function.

This function takes three parameters — the number of trees in the forest, the training data, and the training labels. It returns a list of trees.

Create a random forest with 40 trees using training_data and training_labels.

You should also create a variable named forest_correct and start it at 0. This is the variable that will keep track of how many points in the test set the random forest correctly classifies.
3.

For every data point in the test set, we want every tree to classify the data point, find the most common classification, and compare that prediction to the true label of the data point. This is very similar to what you did in the previous exercise.

To begin, at the end of the for loop outside the if statement, create an empty list named predictions. Next, loop through every forest_tree in forest. Call classify() using testing_data[i] and forest_tree as parameters and append the result to predictions.
4.

After we loop through every tree in the forest, we now want to find the most common prediction and compare it to the true label. The true label can be found using testing_labels[i]. If they’re equal, we’ve correctly classified a point and should add 1 to forest_correct.

An easy way of finding the most common prediction is by using this line of code:

forest_prediction = max(predictions,key=predictions.count)

5.

Finally, after looping through all of the points in the test set, we want to print out the accuracy of our random forest. Divide forest_correct by the number of items in the test set and print the result.

How did the random forest do compared to the single decision tree?
'''

from tree import training_data, training_labels, testing_data, testing_labels, make_random_forest, make_single_tree, classify
import numpy as np
import random
np.random.seed(1)
random.seed(1)
from collections import Counter

tree = make_single_tree(training_data, training_labels)
single_tree_correct = 0

forest = make_random_forest(40, training_data, training_labels)
forest_correct = 0

for i in range(len(testing_data)):
  prediction = classify(testing_data[i], tree)
  if prediction == testing_labels[i]:
    single_tree_correct += 1
  predictions = []
  for forest_tree in forest:
    predictions.append(classify(testing_data[i], forest_tree))
  forest_prediction = max(predictions,key=predictions.count)
  if forest_prediction == testing_labels[i]:
    forest_correct += 1
    
print(single_tree_correct/len(testing_data))
print(forest_correct/len(testing_data))


'''
Random Forest in Scikit-learn

You now have the ability to make a random forest using your own decision trees. However, scikit-learn has a RandomForestClassifier class that will do all of this work for you! RandomForestClassifier is in the sklearn.ensemble module.

RandomForestClassifier works almost identically to DecisionTreeClassifier — the .fit(), .predict(), and .score() methods work in the exact same way.

When creating a RandomForestClassifier, you can choose how many trees to include in the random forest by using the n_estimators parameter like this:

classifier = RandomForestClassifier(n_estimators = 100)

We now have a very powerful machine learning model that is fairly resistant to overfitting!
Instructions
1.

Create a RandomForestClassifier named classifier. When you create it, pass two parameters to the constructor:

    n_estimators should be 2000. Our forest will be pretty big!
    random_state should be 0. There’s an element of randomness when creating random forests thanks to bagging. Setting the random_state to 0 will help us test your code.

2.

Train the forest using the training data by calling the .fit() method. .fit() takes two parameters — training_points and training_labels.
3.

Test the random forest on the testing set and print the results. How accurate was the model?
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from cars import training_points, training_labels, testing_points, testing_labels
import warnings
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=2000, random_state=0)
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))

'''
Review

Nice work! Here are some of the major takeaways about random forests:

    A random forest is an ensemble machine learning model. It makes a classification by aggregating the classifications of many decision trees.
    Random forests are used to avoid overfitting. By aggregating the classification of multiple trees, having overfitted trees in a random forest is less impactful.
    Every decision tree in a random forest is created by using a different subset of data points from the training set. Those data points are chosen at random with replacement, which means a single data point can be chosen more than once. This process is known as bagging.
    When creating a tree in a random forest, a randomly selected subset of features are considered as candidates for the best splitting feature. If your dataset has n features, it is common practice to randomly select the square root of n features.

'''

