'''Lesson
Distance Formula
About 40 minutes

Before we get into Regression algorithms, it's important to understand the way data is represented. One of the foundational ways we compare data points to each other is through distance.

In this lesson, you will explore three different ways to define the distance between two points:

    Euclidean Distance
    Manhattan Distance
    Hamming Distance

Then we will look at how to use them using Python's SciPy library.

'''


'''
Representing Points

In this lesson, you will learn three different ways to define the distance between two points:

    Euclidean Distance
    Manhattan Distance
    Hamming Distance

Before diving into the distance formulas, it is first important to consider how to represent points in your code.

In this exercise, we will use a list, where each item in the list represents a dimension of the point. For example, the point (5, 8) could be represented in Python like this:

pt1 = [5, 8]

Points aren't limited to just two dimensions. For example, a five-dimensional point could be represented as [4, 8, 15, 16, 23].

Ultimately, we want to find the distance between two points. We'll be writing functions that look like this:

distance([1, 2, 3], [5, 8, 9])

Note that we can only find the difference between two points if they have the same number of dimensions!
'''
two_d = [10, 2]
five_d = [30, -1, 50, 0, 2]

four_d = [1, 8, 8, 12]

'''
Euclidean Distance

Euclidean Distance is the most commonly used distance formula. To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. If we add up all of these squared differences and take the square root, we've computed the Euclidean distance.

Let's take a look at the equation that represents what we just learned:

(a1−b1)2+(a2−b2)2+…+(an−bn)2\sqrt{(a_1-b_1)^2+(a_2-b_2)^2+\ldots+(a_n - b_n)^2}(a1​−b1​)2+(a2​−b2​)2+…+(an​−bn​)2

​

The image below shows a visual of Euclidean distance being calculated:

The Euclidean distance between two points.

d=(a1−b1)2+(a2−b2)2d = \sqrt{(a_1-b_1)^2+(a_2-b_2)^2}d=(a1​−b1​)2+(a2​−b2​)2

​'''
def euclidean_distance(pt1, pt2):
  distance = 0
  for idx in range(len(pt1)):
    distance += (pt1[idx] - pt2[idx])**2
  distance = (distance)**0.5
  return distance

print(euclidean_distance([1,2],[4,0]))
print(euclidean_distance([5,4,3],[1,7,9]))
print(euclidean_distance([],[]))


'''
Manhattan Distance

Manhattan Distance is extremely similar to Euclidean distance. Rather than summing the squared difference between each dimension, we instead sum the absolute value of the difference between each dimension. It's called Manhattan distance because it's similar to how you might navigate when walking city blocks. If you've ever wondered "how many blocks will it take me to get from point A to point B", you've computed the Manhattan distance.

The equation is shown below:

∣a1−b1∣+∣a2−b2∣+…+∣an−bn∣\mid a_1 - b_1 \mid + \mid a_2 - b_2 \mid + \ldots + \mid a_n - b_n \mid∣a1​−b1​∣+∣a2​−b2​∣+…+∣an​−bn​∣

Note that Manhattan distance will always be greater than or equal to Euclidean distance. Take a look at the image below visualizing Manhattan Distance:

The Manhattan distance between two points.

d=∣a1−b1∣+∣a2−b2∣d = \mid a_1 - b_1 \mid + \mid a_2 - b_2 \midd=∣a1​−b1​∣+∣a2​−b2​∣

'''
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

print(euclidean_distance([1, 2], [4, 0]))
print(euclidean_distance([5, 4, 3], [1, 7, 9]))

def manhattan_distance(pt1, pt2):
  distance = 0
  for idx in range(len(pt1)):
    distance += abs(pt1[idx]-pt2[idx])
  return distance


print(manhattan_distance([1, 2], [4, 0]))
print(manhattan_distance([5, 4, 3], [1, 7, 9]))

'''
Hamming Distance

Hamming Distance is another slightly different variation on the distance formula. Instead of finding the difference of each dimension, Hamming distance only cares about whether the dimensions are exactly equal. When finding the Hamming distance between two points, add one for every dimension that has different values.

Hamming distance is used in spell checking algorithms. For example, the Hamming distance between the word "there" and the typo "thete" is one. Each letter is a dimension, and each dimension has the same value except for one.

'''
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1, pt2):
  distance = 0
  for idx in range(len(pt1)):
    if not pt1[idx] == pt2[idx]:
      distance += 1
  return distance

print(hamming_distance([1,2],[1,100]))
print(hamming_distance([5,4,9],[1,7,9]))

'''
SciPy Distances

Now that you've written these three distance formulas yourself, let's look at how to use them using Python's SciPy library:

    Euclidean Distance .euclidean()
    Manhattan Distance .cityblock()
    Hamming Distance .hamming()

There are a few noteworthy details to talk about:

First, the scipy implementation of Manhattan distance is called cityblock(). Remember, computing Manhattan distance is like asking how many blocks away you are from a point.

Second, the scipy implementation of Hamming distance will always return a number between 0 an 1. Rather than summing the number of differences in dimensions, this implementation sums those differences and then divides by the total number of dimensions. For example, in your implementation, the Hamming distance between [1, 2, 3] and [7, 2, -10] would be 2. In scipy's version, it would be 2/3.

'''

from scipy.spatial import distance

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance

print(euclidean_distance([1, 2], [4, 0]))
print(manhattan_distance([1, 2], [4, 0]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))

print(distance.euclidean([1, 2], [4, 0]))
print(distance.cityblock([1, 2], [4, 0]))
print(distance.hamming([5,4,9], [1,7,9]))


