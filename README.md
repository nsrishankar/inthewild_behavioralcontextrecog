## WASH Project
1. [Vaizman talk CSE 118, 2017](https://www.youtube.com/watch?v=2cuhvEQZ_sI)
2. [Vaizman thesis research](https://github.com/nsrishankar/inthewild_behavioralcontextrecog/blob/master/thesis.pdf)

### NN Work
#### Classifiers
- Single class classifiers.
- Multi label classifiers- MLP {0-Hidden layers, 1-Hidden Layer, 2-Hidden Layers, 2-Hidden Layer with sensor dropout per minibatch} along with K-Fold cross-validation.
- Work done with LSTM/RNN, XGBoost, Decision trees.

#### Feature Engineering
- Overlapping windows for moving averages (5-minute,30-minute,1-hour,3-hour,5-hour,1-day,2-day windows).
- Overlapping windows for exponentially weighted moving averages (_Above windows, no signifiant improvements_).
- Sequentially stacked windows for moving averages.
- For features: not zero imputing values: Using timestamp data to interpolate (ARIMA, Kalman imputations).
- Missing label estimation through exclusive labels. (Accuracy decrease-unknown)
- Use of a new label ("Unclassified") if none of the 51 labels are positive(1).
- > [Working on] Changepoint analysis/ bayesian segmentation for possible label correcting.
- > [Working on] SSA Decomposition.
- > [Try] Neural decomposition.

#### Mix
- Skip-gram "word" embeddings for sequences.
- T-SNE for labels in action space from embeddings above.
- > [Working on] LSTM sequence prediction followed by sensor predictions.
- > [Working on] Autoregressive, Signal-magnitude Area, TIlt-angle, Linear-Discriminant Analysis for hierarchical network based on embedding.
- > [Working on] Convolving actual signal data.

#### Other
- > [Working on] Absolute location api data.
