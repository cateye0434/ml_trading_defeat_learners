# Machine Learning Algorithms for Trading (Part 1)

There are two parts:

1) Implement linear regression and decision tree learners. 

* LinRegLearner.py: Add data to the learner via `addEvidence` and predict Y values via `query`

* DTLearner.py: The decision tree learner is based on [J.R. Quinlan's paper](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf). Other than `addEvidence` and `query`, this learner also has:
  * `__build_tree`: A private function called by `addEvidence`. It builds the decision tree recursively by choosing the best feature to split on and the splitting value. The best feature has the highest absolute correlation with dataY. If all features have the same absolute correlation, choose the first feature. The splitting value is the median of the data according to the best feature.
  * `__tree_search`(self, point, row): A private function called by query. It recursively searches the decision tree matrix and returns a predicted value for a given query.
  * `get_learner_info`: It print out a tree in the form of a pandas dataframe if verbose is set to True.

2) Generate data for each learner.

* gen_data.py: Includes two functions that return a data set. Each function uses a random number generator as part of its data generation process. Whenever the seed is the same, the same data set is returned. Different seeds should result in different data sets. 
  * `best4LinReg`: This function should return a dataset (X and Y) that will work better for linear regression than decision trees.
  * `best4DT`: This function should return a dataset (X and Y) that will work better for decision trees than linear regression.

* Test files are also included in the repository: testbest4.py and grade_best4.py

## Setup

You need Python 2.7+, and the following packages: pandas, numpy, and scipy.

## Run

To run any script file, use:

```bash
python <script.py>
```

Source: [Part 3](http://quantsoftware.gatech.edu/Machine_Learning_Algorithms_for_Trading) of [Machine Learning for Trading](http://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course) by Georgia Tech
