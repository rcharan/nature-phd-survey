# Leaving The Ivory Tower

Using Nature's 2019 [survey](https://www.nature.com/articles/d41586-019-03459-7)
of PhD students worldwide, I attempt to classify career intent ("likely to
remain in academia or not") based on answers to other survey questions.
([Original Data](https://figshare.com/s/74a5ea79d76ad66a8af8))

I fit several models (Logistic Regression, a simple Decision Tree,  Random Forest,
  AdaBoost, and Gradient Boost) using scikit-learn and selected Logistic Regression
  as the best model based on comparable AUC and preference for interpretability.
  I achieve a total AUC of 82% with 80% accuracy.

For the Logistic Regression, we examine the coefficients and find some surprising
associations. You can view the presentation [in the repository](https://github.com/rcharan/nature-phd-survey/blob/master/Escaping%20the%20Ivory%20Tower.pdf) or on [Google Slides](https://docs.google.com/presentation/d/1mHOpzvGjcN_gwGcptkcT-M1Vo7QSzI1jlDBYw6G05c8/edit?usp=sharing)

This is my Module 5 project for the Flatiron School Data Science Immersive program
in New York City.

## Data
6,812 PhD students worldwide answered the survey which had about 60 questions and
numerous subparts. Questions spanned a wide variety of topics; more details are in the presentation.

I end up with 111 predictors. They are binary/categorical (62 before one-hot encoding)
and ordinal (49), with no continuous predictors. The ordinal variables were treated as
continuous for the purposes of the logistic regression.

There is an overall class imbalance such that the Dummy Classifier achieves 71% accuracy by
predicting that all students will remain in academia.

## Modeling
I ran several models:
- Logistic Regression (weighted and unweighted)
- A simple Decision Tree
- A Random Forest
- AdaBoost
- Gradient Boosting

Except for the Random Forest, the models were fit by 5-fold Cross Validation
on the training data using a Grid Search for the hyper-parameters and then
evaluated on held out test data.

The Random Forest was fit using a custom Grid Search with a custom out-of-bag
(OOB) validation process. This enabled the use of scikit-learn's warm start
feature and substantially sped up fitting times. (I was unable to achieve
  speedups using the default GridSearchCV and warm start in scikit-learn
  and GridSearchCV does not allow 1-fold validation). The Random Forest was also
  evaluated on the same held-out test data.

Details of training times, scores, and optimal parameters are available in the analyze.ipynb
notebook and the presentation. There is also model inspection (e.g. feature importance) in the notebook.

Ultimately, most of the models (the simple Decision Tree aside) performed similarly, as evaluated by AUC.
Some attempts to fix class imbalance issues were unsuccessful and a deeper hyper-parameter search could
be in order to remedy this. I chose Logistic Regression as the winning model because
- It minimizes the cross-entropy loss, my secondary preferred metric
- It is highly interpretable

## Units
Please note that the cross-entropy loss and the logistic regression coefficients are reported
in units of decibels. 1 decibel is approximately 4.3 nats. Software packages including scikit-learn
by default report coefficients and cross-entropy (log) loss in units of nats.

Nats correspond to taking natural logs. Decibels correspond to taking logs in base 10 and then
multiplying by 10. Equivalently: taking logs in base 10**(1/10).

Decibels are a unit of information that is easy to interpret. An increase in 10 decibels corresponds to the odds
for an event going up by a factor of 10. (3 decibels corresponds to a factor of 2).

You can read more about this in this Medium post. [Link to come - 12/13/19]

## How to use this repository
The main analysis is performed in analysis.ipynb. Two settings to be noted:
1. go_for_broke: set this to True to utilize parallel processing in Grid Search
and Random Forests. This will use all CPUs available. (Default: True)
2. load_from_disk: set this to True to load the fitted models from models directory.
This is recommended (default True). If for some reason you want to refit the models
yourself, the fitting times with the current Grid Search parameters are reported.
Total time is a little of 3 hours on a 2.7 GHz Intel Core i5 (Dual Core). Expect this time to
double if you set go_for_broke = False. Expect it to halve if you have 4 comparable CPUs and
allow parallel processing and so on.

The data cleaning is performed on the originally provided survey data in the
data-cleaning folder (data-cleaning.ipynb). The outputs are dataset.csv and
codebook.csv.

The presentation is included as a pdf. Also, several of the tables and figures used
in their raw forms, in the tables folder and figures folder respectively. Additional,
mostly aesthetic processing of some of the tables and figures
is necessary to reproduce the presentation

The utilities.py file contains a handful of conveniences.
