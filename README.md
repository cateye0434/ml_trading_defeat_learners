# Linear Regression and Decision Tree Learners

## Introduction

Create a Linear Regression Learner and a Decision Tree Learner that can take in training data, learn from it and then predict values when given test inputs. The second part of the project is to generate data that works better for one learner than the other. 


This is Part 1 in a four-part series of **Machine Learning Algorithms for Trading**:

* [**Part 1**](https://github.com/ntrang086/ml_trading_defeat_learners) (this repository): Implement the Linear Regression Learner and Decision Tree Learner and generate data that works better for one learner than the other.
* [**Part 2**](https://github.com/ntrang086/ml_trading_assess_learners): Implement Random Tree Learner, Bag learner and Insane Learner. Evaluate all the learners implemented in Parts 1 and 2.
* [**Part 3**](https://github.com/ntrang086/q_learning_robot): Implement the Q-Learning and Dyna-Q solutions to the reinforcement learning problem.
* [**Part 4**](https://github.com/ntrang086/q_learning_trading): Implement a learning trading agent using Q-learning.


## Code and project details

The code can be grouped into three categories:

1) Implement linear regression and decision tree learners. 

* **`LinRegLearner.py`** - Code for a Linear Regression Learner which accepts train data `addEvidence` and predicts Y values via `query`

* **`DTLearner.py`** - Code for a Decision Tree Learner which is based on [J.R. Quinlan's paper](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf). Other than `addEvidence` and `query`, this learner also has:
  * `__build_tree`: A private function called by `addEvidence`. It builds the decision tree recursively by choosing the best feature to split on and the splitting value. The best feature has the highest absolute correlation with dataY. If all features have the same absolute correlation, choose the first feature. The splitting value is the median of the data according to the best feature.
  * `__tree_search`(self, point, row): A private function called by query. It recursively searches the decision tree matrix and returns a predicted value for a given query.
  * `get_learner_info`: It print out a tree in the form of a pandas dataframe if verbose is set to True.

2) Generate data for each learner.

* **`gen_data.py`** - Includes two functions that return a data set. Each function uses a random number generator as part of its data generation process. Whenever the seed is the same, the same data set is returned. Different seeds should result in different data sets. 
  * `best4LinReg`: This function should return a dataset (X and Y) that will work better for linear regression than decision trees.
  * `best4DT`: This function should return a dataset (X and Y) that will work better for decision trees than linear regression.

3) Test files: `testbest4.py` and `grade_best4.py`.

## Setup

You need Python 2.7 or Python 3.x (recommended), and the following packages: pandas, numpy, and scipy.

## Run

To run any script file, use:

```bash
python <script.py>
```

Source: [Part 3](http://quantsoftware.gatech.edu/Machine_Learning_Algorithms_for_Trading) of [Machine Learning for Trading](http://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course) by Georgia Tech
