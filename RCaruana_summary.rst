++++++++
R-Caruna
++++++++

STATLOG - king et al 1995 emperical study for learning methods

Apriori Metrics
+++++++++++++++
.. math:: x->y

Support says how popular an itemset is, as measured by the proportion of transactions in which an itemset appears

.. math:: support = freq(x)

Confidence says how likely item Y is purchased when item X is purchased, 

.. math:: confidence = \frac{support(x and y)}{support(x)}

Lift says how likely item Y is purchased when item X is purchased, while controlling for how popular item Y is

.. math:: lift = \frac{support(x and y)}{support(x)*support(y)}

Isotonic regression
+++++++++++++++++++
Isotonic regression or monotonic regression is the technique of fitting a free-form line to a sequence of observations under the following constraints: the fitted free-form line has to be non-decreasing (or non-increasing) everywhere, and it has to lie as close to the observations as possible.

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Isotonic_regression.svg/600px-Isotonic_regression.svg.png

Platt Scaling
+++++++++++++
Platt Scaling or Platt calibration helps us find the probability of y=1 given x it is defined by

.. math:: p(y=1|x) = \frac{1}{1+e^{Ax+B}}

Evaluated Models
++++++++++++++++

SVM
+++
 - Linear kernel
 - Polynomial kernel with degree 2,3
 - radial with width {0.001,0.005,0.01,0.05,0.1,0.5,1,2}
 - regularization parameter was varied between :math:`10^{-7}` to :math:`10^3`

ANN
+++
 - Trained with Gradient Descent and backprop
 - Hidden Layer {1,2,4,8,32,128}
 - Momentum {0,0.2,0.5,0.9}
 - Training is halted at various steps and the best epoch is used

 