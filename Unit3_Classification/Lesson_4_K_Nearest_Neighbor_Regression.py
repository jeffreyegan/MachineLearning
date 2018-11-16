'''
Regression

The K-Nearest Neighbors algorithm is a powerful supervised machine learning algorithm typically used for classification. However, it can also perform regression.

In this lesson, we will use the movie dataset that was used in the K-Nearest Neighbors classifier lesson. However, instead of classifying a new movie as either good or bad, we are now going to predict its IMDb rating as a real number.

This process is almost identical to classification, except for the final step. Once again, we are going to find the k nearest neighbors of the new movie by using the distance formula. However, instead of counting the number of good and bad neighbors, the regressor averages their IMDb ratings.

For example, if the three nearest neighbors to an unrated movie have ratings of 5.0, 9.2, and 6.8, then we could predict that this new movie will have a rating of 7.0.
1.

We've imported most of the K-Nearest Neighbor algorithm. Before we dive into finishing the regressor, let's refresh ourselves with the data.

At the bottom of your code, print movie_dataset["Life of Pi"]. You should see a list of three values. These values are the normalized values for the movie's budget, runtime, and release year.
2.

Print the rating for "Life of Pi". This can be found in movie_ratings.
3.

We've included the majority of the K-Nearest Neighbor algorithm in the predict() function. Right now, the variable neighbors stores a list of [distance, title] pairs.

Loop through every neighbor and find its rating in movie_ratings. Add those ratings together and return that sum divided by the total number of neighbors.
4.

Call predict with the following parameters:

    [0.016, 0.300, 1.022]
    movie_dataset
    movie_ratings
    5

Print the result.

Note that the list [0.016, 0.300, 1.022] is the normalized budget, runtime, and year of the movie Incredibles 2! The normalized year is larger than 1 because our training set only had movies that were released between 1927 and 2016 — Incredibles 2 was released in 2018.

'''

from movies import movie_dataset, movie_ratings


def distance(movie1, movie2):
    squared_difference = 0
    for i in range(len(movie1)):
        squared_difference += (movie1[i] - movie2[i]) ** 2
    final_distance = squared_difference ** 0.5
    return final_distance


def predict(unknown, dataset, movie_ratings, k):
    distances = []
    # Looping through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    distances.sort()
    # Taking only the k closest points
    neighbors = distances[0:k]
    total = 0
    for neighbor in neighbors:
        title = neighbor[1]
        total += movie_ratings[title]
    return total / len(neighbors)


print(movie_dataset["Life of Pi"])  # [0.00982356711895032, 0.30716723549488056, 0.9550561797752809]
print(movie_ratings["Life of Pi"])  # 8.0

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))  # 6.86



'''
Weighted Regression

We're off to a good start, but we can be even more clever in the way that we compute the average. We can compute a weighted average based on how close each neighbor is.

Let's say we're trying to predict the rating of movie X and we've found its three nearest neighbors. Consider the following table:
Movie 	Rating 	Distance to movie X
A 	5.0 	3.2
B 	6.8 	11.5
C 	9.0 	1.1

If we find the mean, the predicted rating for X would be 6.93. However, movie X is most similar to movie C, so movie C's rating should be more important when computing the average. Using a weighted average, we can find movie X's rating:

The numerator is the sum of every rating divided by their respective distances. The denominator is the sum of one over every distance. Even though the ratings are the same as before, the weighted average has now gone up to 7.9.
1.

Let's redo our predict() function so it computes the weighted average.

Before you begin looping through the neighbors, create a variable named numerator and set it to 0. Loop through every neighbor and add the neighbor's rating (found in movie_ratings) divided by the neighbor's distance to numerator.

For now, return numerator.
2.

Let's now calculate the denominator of the weighted average. Before your loop, create a variable named denominator and set it equal to 0.

Inside your for loop, add 1 divided by the neighbor's distance to denominator.

Outside the loop, return numerator/denominator.
3.

Once again call your predict function using Incredibles 2's features. Those features were [0.016, 0.300, 1.022]. Set k = 5. Print the results.

How did using a weighted average change the predicted rating? Remember, before calculating the weighted average the prediction was 6.86.
'''

from movies import movie_dataset, movie_ratings


def distance(movie1, movie2):
    squared_difference = 0
    for i in range(len(movie1)):
        squared_difference += (movie1[i] - movie2[i]) ** 2
    final_distance = squared_difference ** 0.5
    return final_distance


def predict(unknown, dataset, movie_ratings, k):
    distances = []
    # Looping through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    distances.sort()
    # Taking only the k closest points
    neighbors = distances[0:k]
    numerator = 0
    denominator = 0
    for neighbor in neighbors:
        title = neighbor[1]
        numerator += movie_ratings[title] / neighbor[0]
        denominator += 1 / neighbor[0]
    return numerator / denominator


print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))  # 6.84913967844


'''
Scikit-learn

Now that you've written your own K-Nearest Neighbor regression model, let's take a look at scikit-learn's implementation. The KNeighborsRegressor class is very similar to KNeighborsClassifier.

We first need to create the regressor. We can use the parameter n_neighbors to define our value for k.

We can also choose whether or not to use a weighted average using the parameter weights. If weights equals "uniform", all neighbors will be considered equally in the average. If weights equals "distance", then a weighted average is used.

classifier = KNeighborsRegressor(n_neighbors = 3, weights = "distance")

Next, we need to fit the model to our training data using the .fit() method. .fit() takes two parameters. The first is a list of points, and the second is a list of values associated with those points.

training_points = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]

training_labels = [5.0, 6.8, 9.0]
classifier.fit(training_points, training_labels)

Finally, we can make predictions on new data points using the .predict() method. .predict() takes a list of points and returns a list of predictions for those points.

unknown_points = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

guesses = classifier.predict(unknown_points)

1.

Create a KNeighborsRegressor named regressor where n_neighbors = 5 and weights = "distance".
2.

We've also imported some movie data. Train your classifier using movie_dataset as the training points and movie_ratings as the training values.
3.

Let's predict some movie ratings. Predict the ratings for the following movies:

    [0.016, 0.300, 1.022],
    [0.0004092981, 0.283, 1.0112],
    [0.00687649, 0.235, 1.0112].

These three lists are the features for Incredibles 2, The Big Sick, and The Greatest Showman. Those three numbers associated with a movie are the normalized budget, runtime, and year of release.

Print the predictions!
'''

from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = 'distance')

regressor.fit(movie_dataset, movie_ratings)

unk_data = [[0.016, 0.300, 1.022], [0.0004092981, 0.283, 1.0112], [0.00687649, 0.235, 1.0112]]

guesses = regressor.predict(unk_data)
print(guesses)  # [ 6.84913968  5.47572913  6.91067999]

'''
Review

Great work! Here are some of the major takeaways from this lesson:

    The K-Nearest Neighbor algorithm can be used for regression. Rather than returning a classification, it returns a number.
    By using a weighted average, data points that are extremely similar to the input point will have more of a say in the final result.
    scikit-learn has an implementation of a K-Nearest Neighbor regressor named KNeighborsRegressor.

In the browser, you'll find an example of a K-Nearest Neighbor regressor in action. Instead of the training data coming from IMDb ratings, you can create the training data yourself! Rate the movies that you have seen. Once you've rated more than k movies, a K-Nearest Neighbor regressor will train on those ratings. It will then make predictions for every movie that you haven't seen.

As you add more and more ratings, the predictor should become more accurate. After all, the regressor needs information from the user in order to make personalized recommendations. As a result, the system is somewhat useless to brand new users — it takes some time for the system to "warm up" and get enough data about a user. This conundrum is an example of the cold start problem.
'''

