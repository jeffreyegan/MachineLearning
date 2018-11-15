'''
The Naive Bayes Classifier

A Naive Bayes classifier is a supervised machine learning algorithm that leverages Bayes' Theorem to make predictions and classifications. Recall Bayes' Theorem:

P(A | B) = ( P(B ∣ A) ⋅ P(A) ) / P(B)​

This equation is finding the probability of A given B. This can be turned into a classifier if we replace B with a data point and A with a class. For example, let's say we're trying to classify an email as either spam or not spam. We could calculate P(spam | email) and P(not spam | email). Whichever probability is higher will be the classifier's prediction. Naive Bayes classifiers are often used for text classification.

So why is this a supervised machine learning algorithm? In order to compute the probabilities used in Bayes' theorem, we need previous data points. For example, in the spam example, we'll need to compute P(spam). This can be found by looking at a tagged dataset of emails and finding the ratio of spam to non-spam emails.
'''

'''
Investigate the Data

In this lesson, we are going to create a Naive Bayes classifier that can predict whether a review for a product is positive or negative. This type of classifier could be extremely helpful for a company that is curious about the public reaction to a new product. Rather than reading thousands of reviews or tweets about the product, you could feed those documents into the Naive Bayes classifier and instantly find out how many are positive and how many are negative.

The dataset we will be using for this lesson contains Amazon product reviews for baby products. The original dataset contained many different features including the reviewer's name, the date the review was made, and the overall score. We’ve removed many of those features; the only features that we’re interested in are the text of the review and whether the review was "positive" or "negative". We labeled all reviews with a score less than 4 as a negative review.

Note that in the next two lessons, we've only imported a small percentage of the data to help the code run faster. We'll import the full dataset later when we put everything together!
1.

Let’s look at the data given to us. Print pos_list[0]. Would you classify this review as positive or negative?
2.

Take a look at the first review in neg_list as well. Does that one look negative?
3.

We’ve also created a Counter object for all of the positive reviews and one for all of the negative reviews. These counters are like Python dictionaries — you could find the number of times the word “baby” was used in the positive reviews by printing pos_counter['baby'].

Print the number of times the word "no" was used in the positive and negative reviews. In which set was it used more often?
'''
from reviews import neg_list, pos_list, neg_counter, pos_counter

print(pos_list[0])
print(neg_list[0])
print(pos_counter['no'])

'''
Bayes Theorem I

For the rest of this lesson, we’re going to write a classifier that can predict whether the review "This crib was amazing" is a positive or negative review. We want to compute both P(positive | review) and P(negative | review) and find which probability is larger. To do this, we’ll be using Bayes' Theorem. Let’s look at Bayes' Theorem for P(positive | review).

P(positive ∣ review)=P(review | positive)⋅P(positive)P(review)P(\text{positive}\ | \ \text{review}) = \frac{P(\text{review\ |\ positive}) \cdot P(\text{positive})}{P(\text{review})}P(positive ∣ review)=P(review)

P(review | positive)⋅P(positive)​

The first part of Bayes' Theorem that we are going to tackle is P(positive). This is the probability that any review is positive. To find this, we need to look at all of our reviews in our dataset - both positive and negative - and find the percentage of reviews that are positive.

We've bolded the part of Bayes' Theorem we're working on.

P(positive ∣ review)=P(review | positive)⋅P(positive)P(review)P(\text{positive}\ | \ \text{review}) = \frac{P(\text{review\ |\ positive}) \cdot \textbf{P(positive})}{P(\text{review})}P(positive ∣ review)=P(review)

P(review | positive)⋅P(positive)​

Instructions
1.

Find the total number of positive reviews by finding the length of pos_list. Do the same for neg_list.

Add those two numbers together and save the sum in a variable called total_reviews.
2.

Create variables named percent_pos and percent_neg. percent_pos should be the number of positive reviews divided by total_reviews. Do the same for percent_neg.
3.

Print percent_pos and percent_neg. They should add up to 1!
'''
from reviews import neg_list, pos_list, neg_counter, pos_counter

total_reviews = len(pos_list) + len(neg_list)
percent_pos = len(pos_list) / total_reviews
percent_neg = len(neg_list) / total_reviews
print(total_reviews)
print(percent_pos)
print(percent_neg)



'''
Bayes Theorem II

Let's continue to try to classify the review "This crib was amazing".

The second part of Bayes' Theorem is a bit more extensive. We now want to compute P(review | positive).

P(positive ∣ review) = ( P(review | positive)⋅P(positive) / P(review) ) 

In other words, if we assume that the review is positive, what is the probability that the words "This", "crib", "was", and "amazing" are the only words in the review?

To find this, we have to assume that each word is conditionally independent. This means that one word appearing doesn't affect the probability of another word from showing up. This is a pretty big assumption!

We now have this equation. You can scroll to the right to see the full equation.

P(“This crib was amazing" ∣ positive) = P(“This" ∣ positive)⋅P(“crib" ∣ positive)⋅P(“was" ∣ positive)⋅P(“amazing" ∣ positive)

Let’s break this down even further by looking at one of these terms. P("crib"|positive) is the probability that the word "crib" appears in a positive review. To find this, we need to count up the total number of times "crib" appeared in our dataset of positive reviews. If we take that number and divide it by the total number of words in our positive review dataset, we will end up with the probability of "crib" appearing in a positive review.

P(“crib" ∣ positive) = (# of “crib" in positive) / (# of words in positive)


If we do this for every word in our review and multiply the results together, we have P(review | positive).
1.

Let’s first find the total number of words in all positive reviews and store that number in a variable named total_pos.

To do this, we can use the built-in Python sum() function. sum() takes a list as a parameter. The list that you want to sum is the values of the dictionary pos_counter, which you can get by using pos_counter.values().

Do the same for total_neg.
2.

Create two variables named pos_probability and neg_probability. Each of these variables should start at 1. These are the variables we are going to use to keep track of the probabilities.
3.

Create a list of the words in review and store it in a variable named review_words. You can do this by using Python's .split() function.

For example if the string test contained "Hello there", then test.split() would return ["Hello", "there"].
4.

Loop through every word in review_words. Find the number of times word appears in pos_counter and neg_counter. Store those values in variables named word_in_pos and word_in_neg.

In the next steps, we'll use this variable inside the for loop to do a series of multiplications.
5.

Inside the for loop, set pos_probability to be pos_probability multiplied by word_in_pos / total_pos.

This step is finding each term to be multiplied together. For example, when word is "crib", you're calculating the following:

P(“crib" ∣ positive  = (# of “crib" in positive) / (# of words in positive)


6.

Do the same multiplication for neg_probability.

Outside the for loop, print both pos_probability and neg_probability. Those values are P("This crib was amazing"|positive) and P("This crib was amazing"|negative).
'''

from reviews import neg_counter, pos_counter

review = "This crib was amazing"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = "This crib was amazing".split()

for word in review_words:
    word_in_pos = pos_counter[word]
    word_in_neg = neg_counter[word]
    pos_probability *= word_in_pos / total_pos
    neg_probability *= word_in_neg / total_neg

print(pos_probability)  # P("This crib was amazing"|positive)
print(neg_probability)  # P("This crib was amazing"|negative)


'''
Smoothing

In the last exercise, one of the probabilities that we computed was the following:

P(“crib" ∣ positive) = ( # of “crib" in positive ) / ( # of words in positive) 

But what happens if "crib" was never in any of the positive reviews in our dataset? This fraction would then be 0, and since everything is multiplied together, the entire probability P(review | positive) would become 0.

This is especially problematic if there are typos in the review we are trying to classify. If the unclassified review has a typo in it, it is very unlikely that that same exact typo will be in the dataset, and the entire probability will be 0. To solve this problem, we will use a technique called smoothing.

In this case, we smooth by adding 1 to the numerator of each probability and N to the denominator of each probability. N is the number of unique words in our review dataset.

For example, P("crib" | positive) goes from this:

P(“crib" ∣ positive) = ( # of “crib" in positive ) / ( # of words in positive )

To this:

P(“crib" ∣ positive) = ( # of “crib" in positive  + 1 ) / ( # of words in positive + N ) 


1.

Let’s demonstrate how these probabilities break if there’s a word that never appears in the given datasets.

Change review to "This cribb was amazing". Notice the second b in cribb.
2.

Inside your for loop, when you multiply pos_probability and neg_probability by a fraction, add 1 to the numerator.

Make sure to include parentheses around the numerator!
3.

In the denominator of those fractions, add the number of unique words in the appropriate dataset.

For the positive probability, this should be the length of pos_counter which can be found using len().

Again, make sure to put parentheses around your denominator so the division happens after the addition!

Did smoothing fix the problem?
'''

from reviews import neg_counter, pos_counter

review = "This cribb was amazing"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
    word_in_pos = pos_counter[word]
    word_in_neg = neg_counter[word]

    pos_probability *= (word_in_pos + 1) / (total_pos + len(pos_counter))
    neg_probability *= (word_in_neg + 1) / (total_neg + len(neg_counter))

print(pos_probability)
print(neg_probability)

'''
Classify

If we look back to Bayes' Theorem, we’ve now completed both parts of the numerator. We now need to multiply them together.

P(positive ∣ review) = ( P(review | positive)⋅P(positive) ) / P(review)


Let’s now consider the denominator P(review). In our small example, this is the probability that "This", "crib", "was", and "amazing" are the only words in the review. Notice that this is extremely similar to P(review | positive). The only difference is that we don’t assume that the review is positive.

However, before we start to compute the denominator, let’s think about what our ultimate question is. We want to predict whether the review "This crib was amazing" is a positive or negative review. In other words, we’re asking whether P(positive | review) is greater than P(negative | review). If we expand those two probabilities, we end up with the following equations.

P(positive ∣ review) = ( P(review | positive)⋅P(positive) ) / P(review)


P(negative ∣ review) = ( P(review | negative)⋅P(negative) ) / P(review)


Notice that P(review) is in the denominator of each. That value will be the same in both cases! Since we’re only interested in comparing these two probabilities, there’s no reason why we need to divide them by the same value. We can completely ignore the denominator!

Let’s see if our review was more likely to be positive or negative!
1.

After the for loop, multiply pos_probability by percent_pos and neg_probability by percent_neg. Store the two values in final_pos and final_neg and print both.
2.

Compare final_pos to final_neg:

    If final_pos was greater than final_neg, print "The review is positive".
    Otherwise print "The review is negative".

Did our Naive Bayes Classifier get it right for the review "This crib was amazing"?
3.

Replace the review "This crib was amazing" with one that you think should be classified as negative. Run your program again.

Did your classifier correctly classify the new review?
'''
from reviews import neg_counter, pos_counter

review = "I would never buy a crib like this again"

percent_pos = 0.5
percent_neg = 0.5

total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

review_words = review.split()

for word in review_words:
    word_in_pos = pos_counter[word]
    word_in_neg = neg_counter[word]

    pos_probability *= (word_in_pos + 1) / (total_pos + len(pos_counter))
    neg_probability *= (word_in_neg + 1) / (total_neg + len(neg_counter))

final_pos = pos_probability * percent_pos
final_neg = neg_probability * percent_neg
print(final_pos)
print(final_neg)

if final_pos > final_neg:
    print("The review is positive")
else:
    print("The review is negative")


'''
Formatting the Data for scikit-learn

Congratulations! You've made your own Naive Bayes text classifier. If you have a dataset of text that has been tagged with different classes, you can give your classifier a brand new document and it will predict what class it belongs to.

We're now going to look at how Python's scikit-learn library can do all of that work for us!

In order to use scikit-learn's Naive Bayes classifier, we need to first transform our data into a format that scikit-learn can use. To do so, we're going to use scikit-learn's CountVectorizer object.

To begin, we need to create a CountVectorizer and teach it the vocabulary of the training set. This is done by calling the .fit() method.

For example, in the code below, we've created a CountVectorizer that has been trained on the vocabulary "Training", "review", "one", and "Second".

vectorizer = CountVectorizer()

vectorizer.fit(["Training review one", "Second review"])

After fitting the vectorizer, we can now call its .transform() method. The .transform() method takes a list of strings and will transform those strings into counts of the trained words. Take a look at the code below.

counts = vectorizer.transform(["one review two review"])

counts now stores the array [2, 1, 0, 0]. The word "review" appeared twice, the word "one" appeared once, and neither "Training" nor "Second" appeared at all.

But how did we know that the 2 corresponded to review? You can print vectorizer.vocabulary_ to see the index that each word corresponds to. It might look something like this:

{'one': 1, 'Training': 2, 'review': 0, 'Second': 3}

Finally, notice that even though the word "two" was in our new review, there wasn't an index for it in the vocabulary. This is because "two" wasn't in any of the strings used in the .fit() method.

We can now usecounts as input to our Naive Bayes Classifier.

Note that in the code in the editor, we've imported only a small percentage of our review dataset to make load times faster. We'll import the full dataset later when we put all of the pieces together!
1.

Create a CountVectorizer and name it counter.
2.

Call counter's .fit() method. .fit() takes a list of strings and it will learn the vocabulary of those strings. We want our counter to learn the vocabulary from both neg_list and pos_list.

Call .fit() using neg_list + pos_list as a parameter.
3.

Print counter.vocabulary_. This is the vocabulary that your counter just learned. The numbers associated with each word are the indices of each word when you transform a review.
4.

Let's transform our brand new review. Create a variable named review_counts and set it equal to counter's .transform() function. Remember, .transform() takes a list of strings to transform. So call .transform() using [review] as a parameter.

Print review_counts.toarray(). If you don't include the toarray(), review_counts won't print in a readable format.

It looks like this is an array of all 0s, but the indices that correspond to the words "this", "crib", "was", and "amazing" should all be 1.
5.

We'll use review_counts as the test point for our Naive Bayes Classifier, but we also need to transform our training set.

Our training set is neg_list + pos_list. Call .transform() using that as a parameter. Store the results in a variable named training_counts. We'll use these variables in the next exercise.
'''
from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer

review = "This crib was amazing"

counter = CountVectorizer()

counter.fit(neg_list + pos_list)

print(counter.vocabulary_)

review_counts = counter.transform([review])
print(review_counts.toarray())

training_counts = counter.transform(neg_list + pos_list)

'''
Using scikit-learn

Now that we've formatted our data correctly, we can use it using scikit-learn's MultinomialNB classifier.

This classifier can be trained using the .fit() method. .fit() takes two parameters: The array of data points (which we just made) and an array of labels corresponding to each data point.

Finally, once the model has been trained, we can use the .predict() method to predict the labels of new points. .predict() takes a list of points that you want to classify and it returns the predicted labels of those points.

Finally, .predict_proba() will return the probability of each label given a point. Instead of just returning whether the review was good or bad, it will return the likelihood of a good or bad review.

Note that in the code editor, we've imported some of the variables you created last time. Specifically, we've imported the counter object, training_counts and then make review_counts. This means the program won't have to re-create those variables and should help the runtime of your program.
1.

Begin by making a MultinomialNB object called classifier.
2.

We now want to fit the classifier. We have the transformed points (found in training_counts), but we don't have the labels associated with those points.

We made the training points by combining neg_list and pos_list. So the first half of the labels should be 0 (for negative) and the second half should be 1 (for positive).

Create a list named training_labels that has 1000 0s followed by 1000 1s.

Note that there are 1000 negative and 1000 positive reviews. Normally you could find this out by asking for the length of your dataset — in this example, we haven't included the dataset because it takes so long to load!
3.

Call classifier's .fit() function. Fit takes two parameters: the training set and the training labels.
4.

Call classifier's .predict() method and print the results. This method takes a list of the points that you want to test.

Was your review classified as a positive or negative review?
5.

After printing predict, print a call to the predict_proba method. The parameter to predict_proba should be the same as predict.

The first number printed is the probability that the review was a 0 (bad) and the second number is the probability the review was a 1 (good).
6.

Change the text review to see the probabilities change.

Can you create a review that the algorithm is really confident about being positive?

The review "This crib was great amazing and wonderful" had the following probabilities:

[[ 0.04977729 0.95022271]]

Can you create a review that is even more positive?

Another interesting challenge is to create a clearly negative review that our classifier thinks is positive.
'''
from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "This crib was amazing"
review = "I will never buy a crib like this again"
review_counts = counter.transform([review])

classifier = MultinomialNB()

training_labels = [0] * 1000 + [1] * 1000

classifier.fit(training_counts, training_labels)

print(classifier.predict(review_counts))   # Your test points are found in the variable review_counts. From the way we constructed our labels, 0 is bad and 1 is good.

print(classifier.predict_proba(review_counts))

'''
Review

In this lesson, you’ve learned how to leverage Bayes' Theorem to create a supervised machine learning algorithm. Here are some of the major takeaways from the lesson:

    A tagged dataset is necessary to calculate the probabilities used in Bayes' Theorem.
    In this example, the features of our dataset are the words used in a product review. In order to apply Bayes' Theorem, we assume that these features are independent.
    Using Bayes' Theorem, we can find P(class|data point) for every possible class. In this example, there were two classes — positive and negative. The class with the highest probability will be the algorithm’s prediction.

Even though our algorithm is running smoothly, there’s always more that we can add to try to improve performance. The following techniques are focused on ways in which we process data before feeding it into the Naive Bayes classifier:

    Remove punctuation from the training set. Right now in our dataset, there are 702 instances of "great!" and 2322 instances of "great.". We should probably combine those into 3024 instances of "great".
    Lowercase every word in the training set. We do this for the same reason why we remove punctuation. We want "Great" and "great" to be the same.
    Use a bigram or trigram model. Right now, the features of a review are individual words. For example, the features of the point "This crib is great" are "This", "crib", "is", and "great". If we used a bigram model, the features would be "This crib", "crib is", and "is great". Using a bigram model makes the assumption of independence more reasonable.

These three improvements would all be considered part of the field Natural Language Processing.

You can find the baby product review dataset, along with many others, on Dr. Julian McAuley's website: http://jmcauley.ucsd.edu/data/amazon
Instructions
1.

In the code editor, we've included three Naive Bayes classifiers that have been trained on different datasets. The training sets used are the baby product reviews, reviews for Amazon Instant Videos, and reviews about video games.

Try changing review again and see how the different classifiers react!

'''
from reviews import baby_counter, baby_training, instant_video_counter, instant_video_training, video_game_counter, video_game_training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "this game was violent"

baby_review_counts = baby_counter.transform([review])
instant_video_review_counts = instant_video_counter.transform([review])
video_game_review_counts = video_game_counter.transform([review])

baby_classifier = MultinomialNB()
instant_video_classifier = MultinomialNB()
video_game_classifier = MultinomialNB()

baby_labels = [0] * 1000 + [1] * 1000
instant_video_labels = [0] * 1000 + [1] * 1000
video_game_labels = [0] * 1000 + [1] * 1000


baby_classifier.fit(baby_training, baby_labels)
instant_video_classifier.fit(instant_video_training, instant_video_labels)
video_game_classifier.fit(video_game_training, video_game_labels)

print("Baby training set: " +str(baby_classifier.predict_proba(baby_review_counts)))
print("Amazon Instant Video training set: " + str(instant_video_classifier.predict_proba(instant_video_review_counts)))
print("Video Games training set: " + str(video_game_classifier.predict_proba(video_game_review_counts)))

