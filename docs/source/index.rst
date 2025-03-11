.. conformalopt documentation master file, created by
   sphinx-quickstart on Sun Feb 23 16:26:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


conformalopt
===========================================

This site provides documentation for the :code:`conformalopt` pypi package. 

The main part of the package is the :code:`ConformalPredictor` class,
which provides predictions for online conformal prediction. It can implement all the algorithms introduced in
CITE, and we provide a basic template below. 


Basic Components
----------------

An object of the :code:`ConformalPredictor` class has two central pieces:

1. :code:`quantile_tracker`. This can be set to either :code:`'linear'` (default) or :code:`'scalar'`,
   which corresponds to linear or scalar quantile tracking. The quantile tracker on its own provides 
   coverage, and selecting :code:`'linear'` can lead to adaptivity when the scores 
   show autocorrelations. 
   
   It maintains a parameter :math:`\theta\in\mathbb{R}^d`, which is
   :math:`d=1` when :code:`'scalar'` is selected, but possibly larger when :code:`'linear'` is selected.
   Its prediction is simply the inner product with a feature vector  :math:`\phi\in\mathbb{R}^d`,
   which is typically the last :math:`(d-1)` conformal scores and a  :math:`1` appended as a bias term.
   The method :code:`ConformalPredictor.predict()` thus returns

   .. math::

      \theta^\top \phi,

   and the method :code:`ConformalPredictor.step(prediction, score)` performs a step of gradient descent 
   on :math:`\theta` for its prediction composed with the quantile loss, using a fixed or decaying
   learning rate schedule. 

2. :code:`(optional) scorecaster`. This can be set to  :code:`'ar_quantile_loss_scorecaster'`, :code:`'theta_scorecaster'`
   or :code:`None` (default). The scorecaster trains a model to predict the next score, using the past
   scores. This is often unnecessary, as the linear quantile tracker is often able to do so well
   with a fast prediction and update. 
   The prediction is added to the output of the quantile tracker. 


Template
---------

Here we provide a basic template to give an example of how to use the :code:`ConformalPredictor` class.
In the online conformal setting, we receive conformal scores in a sequential fashion. This package provides
several datasets used in CITE, which can be accessed via the :code:`data.get_scores(data_name)` method.

The first step is to fit the hyperparameters of a :code:`ConformalPredictor` object on some validation data.
Then, it can be used to make predictions, and its :code:`quantile_tracker` should be updated once the true
score is observed. 


.. code-block:: python

   import conformalopt as oc

   # Get an example dataset.
   scores = oc.data.get_scores("GOOGL_theta_absolute-residual_scores")
   split = int(0.33 * len(scores))
   val_scores, test_scores = scores[:split], scores[split:]

   # Instantiate a ConformalPredictor object for linear quantile tracking.
   cp = oc.ConformalPredictor()

   # Fit the ConformalPredictor's hyperparameters (such as learning rate).
   cp.fit(val_scores=val_scores)

   # Prediction loop.
   for t in range(len(test_scores)):
      prediction = cp.predict()  # Make a prediction for the score.
      cp.step(prediction, test_scores[t])  # Update the quantile tracker parameter.

   # Provides an evaluation in out/expt_name folder.
   cp.eval()  

The predictions of the conformal predictor can be used to generate confidence sets
for the underlying task.
This is not directly implemented here, for generality and simplicity.
The confidence set should contain all the possible labels in the underlying 
task that are scored less than the conformal predictor's prediction. 
See CITE for a full description of the online conformal problem setting.

The :code:`ConformalPredictor` class can be modified in several ways via subclassing. To modify
the feature vector beyond the last :math:`(d-1)` conformal scores and a  :math:`1` 
appended as a bias term, override
:code:`ConformalPredictor.get_covariate()`. To add a new scorecaster, override
:code:`ConformalPredictor.scorecaster_predict()`


.. toctree::
   :maxdepth: 2
   :caption: Modules:

   main
   data
