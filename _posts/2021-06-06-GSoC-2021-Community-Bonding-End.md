---
layout: post
title: End of Community Bonding (GSoC 2021)
tags: [gsoc, mlpack]
---

Today, marks the end of the Community Bonding Period for GSoC 2021.
During the this period I completed one of my previous pending PR [#2868](https://github.com/mlpack/mlpack/pull/2868) that is now merged.
I worked on this PR for quite some time and it is a great feeling to see it get merged!
The PR adds a nice functionality proposed by Ryan in [1709](https://github.com/mlpack/mlpack/issues/1709). It is basically aimed at adding model
inspection feature in python bindings.
Now, the user can actually see the parameters of the trained model and they can alter them also.

Here is a very basic example:


Demo Script:
```python
from mlpack import linear_regression
X = [1,2,3,4,5] # training data
y = [1,2,3,4,5] # target variable
X_test = [6,7] # testing data

out = linear_regression(training=X, training_responses=y)
model = out["output_model"]

preds_before = linear_regression(input_model=model, test=X_test)["output_predictions"]
print(preds_before)

params = model.get_cpp_params()
# doubling the parameters, so, the predictions should also double.
params["LinearRegression"]["parameters"] = params["LinearRegression"]["parameters"]  * 2
model.set_cpp_params(params)

preds_after = linear_regression(input_model=model, test=X_test)["output_predictions"]
print(preds_after)
```

Output: 
```bash
[5. 6.] # before predictions.
[10. 12.] # predictions doubled.
```

I have also opened [2961](https://github.com/mlpack/mlpack/pull/2961) as a draft PR to discuss various ideas about the GSoC project.
There was also a short 20 minute meet with Ryan, where we discussed about the overall approach
and how the final API might look like.
Currently, I have added a basic "skeleton" code to the PR, it is still in raw form. But, I am able to make the complete process
work to output a wrapper `.py` file. Next, I will work on the `PrintWrapperPY()` function, that actually prints the wrapper class.

Now that the coding period has started I will try to post a blog every week or maybe every second week.

Stay tuned for future updates!!
