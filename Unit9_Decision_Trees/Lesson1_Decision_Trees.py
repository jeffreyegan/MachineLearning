'''Gini Impurity

Consider the two trees below. Which tree would be more useful as a model that tries to predict whether someone would get an A in a class?
A tree where the leaf nodes have different types of classification A tree where the leaf nodes have only one type of classification

Let’s say you use the top tree. You’ll end up at a leaf node where the label is up for debate. The training data has labels from both classes! If you use the bottom tree, you’ll end up at a leaf where there’s only one type of label. There’s no debate at all! We’d be much more confident about our classification if we used the bottom tree.

This idea can be quantified by calculating the Gini impurity of a set of data points. To find the Gini impurity, start at 1 and subtract the squared percentage of each label in the set. For example, if a data set had three items of class A and one item of class B, the Gini impurity of the set would be

1−(34)2−(14)2=0.3751 - \bigg(\frac{3}{4}\bigg)^2 - \bigg(\frac{1}{4}\bigg)^2 = 0.3751−(4

3​)2−(4

1​)2=0.375

If a data set has only one class, you’d end up with a Gini impurity of 0. The lower the impurity, the better the decision tree!
Instructions
1.

Let’s find the Gini impurity of the set of labels we’ve given you.

Let’s start by creating a variable named impurity and set it to 1.
2.

We now want to count up how many times every unique label is in the dataset. Python’s Counter object can do this quickly.

For example, given the following code:

lst = ["A", "A", "B"]
counts = Counter(lst)

would result in counts storing this object:

Counter({"A": 2, "B": 1})

Create a counter object of labels‘ items named label_counts.

Print out label_counts to see if it matches what you expect.
3.

Let’s find the probability of each label given the dataset. Loop through each label in label_counts.

Inside the for loop, create a variable named probability_of_label. Set it equal to the label count divided by the total number of labels in the dataset.

For every label, the count associated with that label can be found at label_count[label].

We can find the total number of labels in the dataset with len(labels).
4.

We now want to take probability_of_label, square it, and subtract it from impurity.

Inside the for loop, subtract probability_of_label squared from impurity.

In Python, you can square x by using x ** 2.
5.

Outside of the for loop, print impurity.

Test out some of the other labels that we’ve given you by uncommenting them. Which one do you expect to have the lowest impurity?

In the next exercise, we’ll put all of your code into a function. If you want a challenge, try creating the function yourself! Ours is named gini(), takes labels as a parameter, and returns impurity.
'''

from collections import Counter

labels = ["unacc", "unacc", "acc", "acc", "good", "good"]
#labels = ["unacc","unacc","unacc", "good", "vgood", "vgood"]
#labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc"]

impurity = 1
label_counts = Counter(labels)
print(label_counts)

for l in label_counts:
  probability_of_label = 1.0*label_counts[l] / len(labels)
  impurity = impurity - probability_of_label ** 2
print(impurity)

'''Information Gain

We know that we want to end up with leaves with a low Gini Impurity, but we still need to figure out which features to split on in order to achieve this. For example, is it better if we split our dataset of students based on how much sleep they got or how much time they spent studying?

To answer this question, we can calculate the information gain of splitting the data on a certain feature. Information gain measures difference in the impurity of the data before and after the split. For example, let’s say you had a dataset with an impurity of 0.5. After splitting the data based on a feature, you end up with three groups with impurities 0, 0.375, and 0. The information gain of splitting the data in that way is 0.5 - 0 - 0.375 - 0 = 0.125.

Not bad! By splitting the data in that way, we’ve gained some information about how the data is structured — the datasets after the split are purer than they were before the split. The higher the information gain the better — if information gain is 0, then splitting the data on that feature was useless! Unfortunately, right now it’s possible for information gain to be negative. In the next exercise, we’ll calculate weighted information gain to fix that problem.
Instructions
1.

We’ve given you a set of labels named unsplit_labels and two different ways of splitting those labels into smaller subsets. Let’s calculate the information gain of splitting the labels in this way.

At the bottom of your code, begin by creating a variable named info_gain. info_gain should start at the Gini impurity of the unsplit_labels.
2.

We now want to subtract the impurity of each subset in split_labels_1 from info_gain.

Loop through every subset in split_labels_1. We want to change the value of info_gain.

For every subset, calculate the Gini impurity and subtract it from info_gain.
3.

Outside of your loop, print info_gain.

We’ve given you a second way to split the data. Instead of looping through the subsets in split_labels_1, loop through the subsets in split_labels_2.

Which split resulted in more information gain?

Once again, in the next exercise, we’ll put the code you wrote into a function named information_gain that takes unsplit_labels and split_labels as parameters.
'''
from collections import Counter

unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood", "vgood", "vgood"]

split_labels_1 = [
  ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"], 
  [ "good", "good"], 
  ["vgood", "vgood"]
]

split_labels_2 = [
  ["unacc", "unacc", "unacc", "unacc","unacc", "unacc", "good", "good", "good", "good"], 
  ["vgood", "vgood", "vgood"]
]

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

info_gain = gini(unsplit_labels)
for subset in split_labels_1:
  info_gain -= gini(subset)
print(info_gain)  # 0.14522609394404262

info_gain = gini(unsplit_labels)
for subset in split_labels_2:
  info_gain -= gini(subset)
print(info_gain)  # 0.15905325443786977

'''
Weighted Information Gain

We’re not quite done calculating the information gain of a set of objects. The sizes of the subset that get created after the split are important too! For example, the image below shows two sets with the same impurity. Which set would you rather have in your decision tree?

Both of these sets are perfectly pure, but the purity of the second set is much more meaningful. Because there are so many items in the second set, we can be confident that whatever we did to produce this set wasn’t an accident.

It might be helpful to think about the inverse as well. Consider these two sets with the same impurity:

Both of these sets are completely impure. However, that impurity is much more meaningful in the set with more instances. We know that we are going to have to do a lot more work in order to completely separate the two classes. Meanwhile, the impurity of the set with two items isn’t as important. We know that we’ll only need to split the set one more time in order to make two pure sets.

Let’s modify the formula for information gain to reflect the fact that the size of the set is relevant. Instead of simply subtracting the impurity of each set, we’ll subtract the weighted impurity of each of the split sets. If the data before the split contained 20 items and one of the resulting splits contained 2 items, then the weighted impurity of that subset would be 2/20 * impurity. We’re lowering the importance of the impurity of sets with few elements.

Now that we can calculate the information gain using weighted impurity, let’s do that for every possible feature. If we do this, we can find the best feature to split the data on.
Instructions
1.

Let’s update the information_gain function to make it calculate weighted information gain.

When subtracting the impurity of a subset from info_gain, first multiply the impurity by the correct percentage.

The percentage should be the number of labels in the subset, len(subset), divided by the number of labels before the split, len(starting_labels).
2.

We’ve given you a split() function along with ten cars and the car_labels associated with those cars.

After your information_gain() function, call split() using cars, car_labels and 3 as a parameter. This will split the data based on the third index (That feature was the number of people the car could hold).

split() returns two lists. Create two variables named split_data and split_labels and set them equal to the result of the split function.

We’ll explore what these variables contain in a second!
3.

Take a look at what these variables are. Begin by printing split_data. It’s kind of hard to tell what’s going on there! There are so many lists of lists!

Try printing the length of split_data. What do you think this is telling you?

Also try printing split_data[0]. What do you notice about the items at index 3 of all these lists? (Remember, when we called split, we used 3 as the split index).

Try printing split_data[1]. What do you notice about the items at index 3 of these lists?
4.

We now know that split_data contains the cars split into different subsets. split_labels contains the labels of those cars split into different subsets.

Use those split labels to find the information gain of splitting on index 3! Remember, the information_gain() function takes a list of the labels before the split (car_labels), and a list of the subsets of labels after the split (split_labels).

Call this function and print the result! How did we do when we split the function on index 3?
5.

We found the information gain when splitting on feature 3. Let’s do the same for every possible feature.

Loop through all of the features of our data to find the best one to split on! Each car has six features, so we want to loop through the indices 0 through 5.

Inside your for loop, call split() using the unsplit data, the unsplit labels, and the index that you’re looping through.

Call information_gain() using the resulting split labels and print the results. Which feature produces the most information gain?
'''
from collections import Counter

cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    # Multiply gini(subset) by the correct percentage below
    info_gain -= 1.0*len(subset)/len(starting_labels) * gini(subset)
  return info_gain

for f in list(range(0,6)):
  split_data, split_labels = split(cars, car_labels, f)
  print(information_gain(car_labels, split_labels))

#0.2733333333333334
#0.04000000000000001
#0.10666666666666663
#0.30666666666666675
#0.15000000000000002
#0.29000000000000004

'''Recursive Tree Building

Now that we can find the best feature to split the dataset, we can repeat this process again and again to create the full tree. This is a recursive algorithm! We start with every data point from the training set, find the best feature to split the data, split the data based on that feature, and then recursively repeat the process again on each subset that was created from the split.

We’ll stop the recursion when we can no longer find a feature that results in any information gain. In other words, we want to create a leaf of the tree when we can’t find a way to split the data that makes purer subsets.

The leaf should keep track of the classes of the data points from the training set that ended up in the leaf. In our implementation, we’ll use a Counter object to keep track of the counts of labels.

We’ll use these counts to make predictions about new data that we give the tree.
Instructions
1.

We’ve given you the function find_best_split() that takes a set of data points and a set of labels.

The function returns the index of the feature that causes the best split and the information gain caused by that split.

For now, at the bottom of your code, call this function using car_data and car_labels as parameters and store the values in variables named best_feature and best_gain.

Print those two variables. What was the best feature to split on and what was the information gain?
2.

Let’s create a function called build_tree() that takes data and labels as parameters.

Move your call of find_best_split() inside this function, but change the parameters from car_data and car_labels to data and labels.

If best_gain is 0, return a Counter object of labels. We’ve reached the base case — there’s no way to gain any more information so we want to create a leaf.
3.

After the if statement, we want to start working on the recursive case.

In the recursive case, we want to split the data into subsets using the best feature, and then recursively call the build_tree() function on those subsets to create subtrees. Finally, we want to return a list of all those subtrees.

Let’s begin by splitting the data. You can do this by using the split() function which takes three parameters — the data and labels that you want to split and the index of the feature you want to split on.

Store the result of the split() function in two variables named data_subsets and label_subsets.

For now, return data_subsets at the bottom of your function.
4.

Before that final return statement, create an empty list named branches. This list will store all of the subtrees we’re about to make from our recursive calls.

We now want to loop through all of the subsets of data and labels. Set up your for loop like this

for i in range(len(data_subsets)):

Inside the for loop, call build_tree using data_subsets[i] and label_subsets[i] as parameters and append the result to branches.

Finally outside the for loop, return branches instead of data_subsets.
5.

Let’s test our function! At the bottom of your code outside of your function definition, call build_tree() using car_data and car_labels as parameters and store the result in a variable named tree.

We’ve written a function called print_tree() that will help you visualize the tree. Call print_tree() using tree as a parameter.
'''
from tree import *

car_data = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain == 0:
    return Counter(labels)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branch = build_tree(data_subsets[i], label_subsets[i])
    branches.append(branch)
  return branches
  
tree = build_tree(car_data, car_labels)
print_tree(tree)  



'''
Decision Trees
Classifying New Data

We can finally use our tree as a classifier! Given a new data point, we start at the top of the tree and follow the path of the tree until we hit a leaf. Once we get to a leaf, we’ll use the classes of the points from the training set to make a classification.

We’ve slightly changed the way our build_tree() function works. Instead of returning a list of branches or a Counter object, the build_tree() function now returns a Leaf object or an Internal_Node object. We’ll explain how to use these objects in the instructions!

Let’s write a function that will use our tree to classify new points!
Instructions
1.

We’ve created a tree named tree using a lot of car data. Use the print_tree() function with tree as a parameter to see it.

Notice that the tree now knows which feature was used to split the data. This new information is contained in the Leaf and Internal_Node classes. This will come in handy when we write our classify function!

Comment out printing the tree once you get a sense of how large it is!
2.

Let’s start writing the classify() function. classify() should take a datapoint and a tree as a parameter.

The first thing classify should do is check to see if we’re at a leaf.

Check to see if tree is a Leaf by using the isinstance() function.

For example, isinstance(a, list) will be True if a is a list. You should check if tree is a Leaf.

If we’ve found a Leaf, that means we want to return the label with the highest count. The label counts are stored in tree.labels.

You could find the label with the largest count by using a for loop, or by using this rather complicated line of code:

return max(tree.labels.items(), key=operator.itemgetter(1))[0]

3.

If we’re not at a leaf, we want to find the branch that corresponds to our data point. For example, if we’re splitting on index 0 and our data point is ['med', 'low', '4', '2', 'big', 'low'], we want to find the branch that contains all of the points with med at index 0.

To start, let’s find datapoint‘s value of the feature we’re looking for. If datapoint were the example above, and the feature we’re interested is 0, this would be med.

Outside the if statement, create a variable named value and set it equal to datapoint[tree.feature]. tree.feature contains the index of the feature that we’re splitting on, so datapoint[tree.feature] is the value at that index.

To help us check your code, return value.
4.

Start by deleting return value.

Let’s now loop through all of the branches in the tree to find the one that has all the data points with value at the correct index.

Your loop should look like this:

for branch in tree.branches:

Next, inside the loop, check to see if branch.value is equal to value. If it is, we’ve found the branch that we’re looking for! We want to now recursively call classify() on that branch:

return classify(datapoint, branch)

We know that one of these branches will be the one we’re looking for, so we know that this return statement will happen once.
5.

Finally, outside of your function, call classify() using test_point and tree as parameters. Print the results. You should see a classification for this new point.

'''



from tree import *
import operator

test_point = ['vhigh', 'low', '3', '4', 'med', 'med']

def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]

  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)

print(classify(test_point, tree))

'''
Decision Trees in scikit-learn

Nice work! You’ve written a decision tree from scratch that is able to classify new points. Let’s take a look at how the Python library scikit-learn implements decision trees.

The sklearn.tree module contains the DecisionTreeClassifier class. To create a DecisionTreeClassifier object, call the constructor:

classifier = DecisionTreeClassifier()

Next, we want to create the tree based on our training data. To do this, we’ll use the .fit() method.

.fit() takes a list of data points followed by a list of the labels associated with that data. Note that when we built our tree from scratch, our data points contained strings like "vhigh" or "5more". When creating the tree using scikit-learn, it’s a good idea to map those strings to numbers. For example, for the first feature representing the price of the car, "low" would map to 1, "med" would map to 2, and so on.

classifier.fit(training_data, training_labels)

Finally, once we’ve made our tree, we can use it to classify new data points. The .predict() method takes an array of data points and will return an array of classifications for those data points.

predictions = classifier.predict(test_data)

If you’ve split your data into a test set, you can find the accuracy of the model by calling the .score() method using the test data and the test labels as parameters.

print(classifier.score(test_data, test_labels))

.score() returns the percentage of data points from the test set that it classified correctly.
Instructions
1.

We’ve imported a the full car dataset and split it into a training and test set. We’ve also mapped the features that were strings like "vgood" to numbers.

Print training_points[0] and training_labels[0] to see the first car in the training set.
2.

Create a DecisionTreeClassifier and name it classifier.
3.

Build the tree using the training data by calling the .fit() method. .fit() takes two parameters — the training data and the training labels.
4.

Test the decision tree on the testing set and print the results. How accurate was the model?

'''
from cars import training_points, training_labels, testing_points, testing_labels
from sklearn.tree import DecisionTreeClassifier

print(training_points[0])
print(training_labels[0])
classifier = DecisionTreeClassifier()
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))

'''
Decision Tree Limitations

Now that we have an understanding of how decision trees are created and used, let’s talk about some of their limitations.

One problem with the way we’re currently making our decision trees is that our trees aren’t always globablly optimal. This means that there might be a better tree out there somewhere that produces better results. But wait, why did we go through all that work of finding information gain if it’s not producing the best possible tree?

Our current strategy of creating trees is greedy. We assume that the best way to create a tree is to find the feature that will result in the largest information gain right now and split on that feature. We never consider the ramifications of that split further down the tree. It’s possible that if we split on a suboptimal feature right now, we would find even better splits later on. Unfortunately, finding a globally optimal tree is an extremely difficult task, and finding a tree using our greedy approach is a reasonable substitute.

Another problem with our trees is that they potentially overfit the data. This means that the structure of the tree is too dependent on the training data and doesn’t accurately represent the way the data in the real world looks like. In general, larger trees tend to overfit the data more. As the tree gets bigger, it becomes more tuned to the training data and it loses a more generalized understanding of the real world data.

One way to solve this problem is to prune the tree. The goal of pruning is to shrink the size of the tree. There are a few different pruning strategies, and we won’t go into the details of them here. scikit-learn currently doesn’t prune the tree by default, however we can dig into the code a bit to prune it ourselves.

'''
from cars import training_points, training_labels, testing_points, testing_labels
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state = 0, max_depth=11)
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))
print(classifier.tree_.max_depth)

'''
Review

Great work! In this lesson, you learned how to create decision trees and use them to make classifications. Here are some of the major takeaways:

    Good decision trees have pure leaves. A leaf is pure if all of the data points in that class have the same label.
    Decision trees are created using a greedy algorithm that prioritizes finding the feature that results in the largest information gain when splitting the data using that feature.
    Creating an optimal decision tree is difficult. The greedy algorithm doesn’t always find the globally optimal tree.
    Decision trees often suffer from overfitting. Making the tree small by pruning helps to generalize the tree so it is more accurate on data in the real world.

'''
