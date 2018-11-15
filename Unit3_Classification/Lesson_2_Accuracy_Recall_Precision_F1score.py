'''
Accuracy

After creating a machine learning algorithm capable of making classifications, the next step in the process is to calculate its predictive power. In order to calculate these statistics, we’ll need to split our data into a training set and validation set.

Let’s say you’re using a machine learning algorithm to try to predict whether or not you will get above a B on a test. The features of your data could be something like:

    The number of hours you studied this week.
    The number of hours you watched Netflix this week.
    The time you went to bed the night before the test.
    Your average in the class before taking the test.

The simplest way of reporting the effectiveness of an algorithm is by calculating its accuracy. Accuracy is calculated by finding the total number of correctly classified points and dividing by the total number of points.

In other words, accuracy can be defined as:

(True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)

Let’s define those terms in the context of our grade example :

    True Positive: The algorithm predicted you would get above a B, and you did.
    True Negative: The algorithm predicted you would get below a B, and you did.
    False Positive: The algorithm predicted you would get above a B, and you didn’t.
    False Negative: The algorithm predicted you would get below a B, and you didn’t.

Let’s calculate the accuracy of a classification algorithm!
1.

Create four variables that start at 0. They should be called true_positives, true_negatives, false_positives, and false_negatives
2.

We've given you two lists. labels represents the true labels of your dataset. Each 1 represents a test that you got above a B on, and each 0 represents a test that was below a B. guesses represents the classifications a machine learning algorithm might return. For every test, the classifier guessed whether your grade was above or below a B.

Loop through guesses and add 1 to true_positives every time your algorithm found a true positive. A true positive is when the real label and the classifier's guess are both 1.
3.

Inside the for loop, count the number of true negatives, false positives, and false negatives.
4.

Outside of the for loop, calculate the accuracy and store it in a variable named accuracy. Print accuracy.

'''

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for idx in range(len(labels)):
  if labels[idx] == 1 and guesses[idx] == 1:
    true_positives += 1
  if labels[idx] == 0 and guesses[idx] == 0:
    true_negatives += 1
  if labels[idx] == 0 and guesses[idx] == 1:
    false_positives += 1
  if labels[idx] == 1 and guesses[idx] == 0:
    false_negatives += 1

accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
print(accuracy)  # 0.3

'''
Recall

Accuracy can be an extremely misleading statistic depending on your data. Consider the example of an algorithm that is trying to predict whether or not there will be over 3 feet of snow on the ground tomorrow. We can write a pretty accurate classifier right now: always predict False. This classifier will be incredibly accurate — there are hardly ever many days with that much snow. But this classifier never finds the information we’re actually interested in.

In this situation, the statistic that would be helpful is recall. Recall measures the percentage of relevant items that your classifier found. In this example, recall is the number of snow days the algorithm correctly predicted divided by the total number of snow days. Another way of saying this is:

True Positives / (True Positives + False Negatives)

Our algorithm that always predicts False might have a very high accuracy, but it never will find any True Positives, so its recall is 0. This makes sense; recall should be very low for such an absurd classifier.
Instructions
1.

After printing the accuracy, calculate and print the recall. Store recall in a variable named recall.
'''

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
    # True Positives
    if labels[i] == 1 and guesses[i] == 1:
        true_positives += 1
    # True Negatives
    if labels[i] == 0 and guesses[i] == 0:
        true_negatives += 1
    # False Positives
    if labels[i] == 0 and guesses[i] == 1:
        false_positives += 1
    # False Negatives
    if labels[i] == 1 and guesses[i] == 0:
        false_negatives += 1

accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)  # 0.3

recall = (true_positives) / (true_positives + false_negatives)
print(recall)  # 0.42857142857142855

'''
Precision

Unfortunately, recall isn’t a perfect statistic either. For example, we could create a snow day classifier that always returns True. This would have low accuracy, but its recall would be 1 because it would be able to accurately find every snow day. But this classifier is just as nonsensical as the one before! The statistic that will help demonstrate that this algorithm is flawed is precision.

In the snow day example, precision is the number of snow days the algorithm correctly predicted divided by the number of times it predicted there would be a snow day. The formula for precision is below:

True Positives / (True Positives + False Positives)

The algorithm that predicts every day is a snow day has recall of 1, but it will have very low precision. It correctly predicts every snow day, but there are tons of false positives as well.

Precision and recall are statistics that are on opposite ends of a scale. If one goes down, the other will go up.
Instructions
1.

After printing the recall, calculate and print the precision. Store the precision in a variable named precision
'''
labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
    # True Positives
    if labels[i] == 1 and guesses[i] == 1:
        true_positives += 1
    # True Negatives
    if labels[i] == 0 and guesses[i] == 0:
        true_negatives += 1
    # False Positives
    if labels[i] == 0 and guesses[i] == 1:
        false_positives += 1
    # False Negatives
    if labels[i] == 1 and guesses[i] == 0:
        false_negatives += 1

accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)  # 0.3

recall = true_positives / (true_positives + false_negatives)
print(recall)  # 0.42857142857142855

precision = true_positives / (true_positives + false_positives)
print(precision)  # 0.5


'''
F1 Score

It is useful to consider the precision and recall of an algorithm, however, we still don’t have one number that can sufficiently describe how effective our algorithm is. This is the job of the F1 score — F1 score is the harmonic mean of precision and recall. The harmonic mean of a group of numbers is a way to average them together. The formula for F1 score is below:

F1 = 2 \* (precision \* recall) / (precision + recall)

The F1 score combines both precision and recall into a single statistic. We use the harmonic mean rather than the traditional arithmetic mean because we want the F1 score to have a low value when either precision or recall is 0.

For example, consider a classifier where recall = 1 and precision = 0.01. We know that there is most likely a problem with this classifier since the precision is so low, and so we want the F1 score to reflect that.

If we took the arithmetic mean, we’d get:

(1 + 0.01) / 2 = 0.505

That looks way too high! But if we calculate the harmonic mean, we get:

2 * (1 * 0.01) / (1 + 0.01) = 0.019

That’s much better! The F1 score is now accurately describing the effectiveness of this classifier.
Instructions
1.

After printing the precision, calculate and print the F1 score. Store the F1 score in a variable named f_1
'''

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
    # True Positives
    if labels[i] == 1 and guesses[i] == 1:
        true_positives += 1
    # True Negatives
    if labels[i] == 0 and guesses[i] == 0:
        true_negatives += 1
    # False Positives
    if labels[i] == 0 and guesses[i] == 1:
        false_positives += 1
    # False Negatives
    if labels[i] == 1 and guesses[i] == 0:
        false_negatives += 1

accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)  # 0.3

recall = true_positives / (true_positives + false_negatives)
print(recall)  # 0.42857142857142855

precision = true_positives / (true_positives + false_positives)
print(precision)  # 0.5

f_1 = 2 * (precision * recall) / (precision + recall)
print(f_1)  # 0.4615384615384615

'''
Review

You’ve now learned many different ways to analyze the predictive power of your algorithm. Some of the key insights for this course include:

    Classifying a single point can result in a true positive (truth = 1, guess = 1), a true negative (truth = 0, guess = 0), a false positive (truth = 0, guess = 1), or a false negative (truth = 1, guess = 0).
    Accuracy measures how many classifications your algorithm got correct out of every classification it made.
    Recall measures the percentage of the relevant items your classifier was able to successfully find.
    Precision measures the percentage of items your classifier found that were actually relevant.
    Precision and recall are tied to each other. As one goes up, the other will go down.
    F1 score is a combination of precision and recall.
    F1 score will be low if either precision or recall is low.

The decision to use precision, recall, or F1 score ultimately comes down to the context of your classification. Maybe you don't care if your classifier has a lot of false positives. If that's the case, precision doesn't matter as much.

As long as you have an understanding of what question you're trying to answer, you should be able to determine which statistic is most relevant to you.

The Python library scikit-learn has some functions that will calculate these statistics for you!
Instructions
1.

Python's scikit-learn library has functions that will find accuracy, recall, precision, and F1 score for you. They all take two parameters — a list of the true labels and a list of the predicted classifications.

Call accuracy_score() using the correct parameters and print the results.
2.

Call the three other functions and print the results. The name of those functions are:

    recall_score()
    precision_score()
    f1_score()

'''

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

print(accuracy_score(labels, guesses))
print(recall_score(labels, guesses))
print(precision_score(labels, guesses))
print(f1_score(labels, guesses))
'''
0.3
0.428571428571
0.5
0.461538461538
'''