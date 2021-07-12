---
layout: post
title: Mid GSoC Report
tags: [gsoc, mlpack]
---

The mid of GSoC 2021 is reached with the first evaluations starting from today. This is a report of the progress achieved until now.
Before starting let me apologize as I promised to write a blog every week but it did not go as I planned. I will try to cover all the progress
till now in this report within as little space as possible, so read till the end.

First, I started to work on [#2961](https://github.com/mlpack/mlpack/pull/2961), this PR was meant as a reference for me, to try different things and come up with the best
approach. A major "hurdle" that I was facing was that I had to include different `_main.cpp` files inside the same file:
Let me back up a little and explain with an example.
So, let us take the `linear_regression` program. This program is divided into 2 different programs i.e. `linear_regression_fit` and
`linear_regression_predict`. This is done so that these individual programs can be wrapped by methods (`fit` and `predict` respectively) of a
class named `LinearRegression`. Since each of these programs has its own set of parameters hence, we need to include both of these programs
together to access both their parameters together for wrapper class construction.
Of course, this is easier said than done. Based on the current structure of these bindings and the `IO` class, this was not possible to do (at least
not in a sensible way), so Ryan and I decided that we need to find a more long-term solution for this. As a result, Ryan started to work on 
[#2995](https://github.com/mlpack/mlpack/pull/2995), which makes it easier for me to achieve what I want. [#2995](https://github.com/mlpack/mlpack/pull/2995) was made to solve the thread-safety issue [here](https://github.com/mlpack/mlpack/issues/2832), it turned out as a bonus for me :).
Then I helped with refactoring the python bindings in [#2995](https://github.com/mlpack/mlpack/pulls/2995). After this was achieved, I jumped back to my GSoC project.

I started my work on [this](https://github.com/NippunSharma/mlpack/tree/revamp_bindings_2) branch, which is built upon Ryan's PR. I took help from earlier DRAFT PR, which had a basic blueprint of everything, I
just had to refine it. After this bit was completed and I was able to generate a wrapper for the `linear_regression` program, the next steps seemed to be
easy to follow. I just had to go through each binding and refactor it as I did for `linear_regression`. This is how the API looked:

```python
from mlpack import LinearRegression

X_train = [1,2,3,4]
y_train = [1,2,3,4]
X_test = [5,6,7]

model = LinearRegression()
model.fit(training=X_train, training_responses=y_train)
preds = model.predict(test=X_test)
```
I aimed to make the API as close to the scikit learn API as possible as it is the most used library in the machine learning world (at least for python).
Then it hit me and I discussed an idea with my mentors in our weekly meeting. I appreciate their open-mindedness that they allowed me to pursue
this idea to check whether it is worth it or not. The idea was to make the mlpack methods scikit compatible (in python).
For people who are not aware of this, scikit compatibility would allow mlpack users to use mlpack models with various scikit-learn utilities, such as:
1. `GridSearchCV` and `BayesSearchCV` are provided by scikit-learn and scikit-optimize respectively for hyperparameter tuning.
2. `Pipeline`, this is a very useful structure provided by scikit-learn, where a user can perform preprocessing and model training in just a few lines of code.
3. `cross_val_score`, to test the model's performance using cross-validation.


There might me some more utilities, but these are the ones that are used the most.

After spending some time, thinking about this idea and how to integrate it with mlpack codebase, I came to the conclusion that not all mlpack
models can be scikit compatible. Only the mlpack's classifiers and regressors can be made scikit compatible, but this was still really good news for
me.


A question that pops to mind here is that making mlpack's API scikit compatible makes scikit-learn a dependancy for mlpack? And the answer here is... NO.
The reason is that, if a user has scikit-learn already installed in their environment, then mlpack can leverage scikit's utilities but if there is no scikit-learn
installed in the environment, then also mlpack models are fully usable (just not usable with scikit utilities). Scikit-learn or not, mlpack's API remains the same.
This also means that if a user copies a code from an environment that has scikit learn and runs it in an environment that does not have scikit learn, then it will
run just fine (obviously, I am talking about the mlpack part of the code and not scikit which will not run).

This is how mlpack's models can be used in harmony with scikit utilities to achieve their full potential:


(here I have used mlpack's Adaboost along with scikit-optimize's Bayesian Search for hyperparameter tuning)

```python
from skopt import BayesSearchCV
from mlpack import Adaboost
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# tuning the model.
opt = BayesSearchCV(
    Adaboost(),
    {
        'iterations': (100, 500),
        'weak_learner': ['perceptron', 'decision_stump']
    },
    n_iter=10,
    cv=3,
    scoring="f1_weighted",
    random_state=1,
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
# output: 
# val. score: 0.9554649920291841
# test score: 0.9558136593062766

opt.best_params_
# output:
# OrderedDict([('iterations', 226), ('weak_learner', 'perceptron')])
```
Similar to this other methods can also be used with scikit learn.
This feature integrates mlpack into the workflow of the user.

The next part of the project focuses on changing the binding documentation corresponding to the changes made in the API.
First, I will try to create a mock-up of how the documentation should look (I have started doing it) and then make changes correspondingly.

Thanks for reading this far, I know I lied that I would wrap this up in little space but you should have already known from the title.

