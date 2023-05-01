import numpy as np
import pandas as pd

'''
Class for implementing the Prepend algorithm's Decision List.
A Decision List contains two objects: (1) predictors (2) groups. These should 
always be the same length.
predictors: a list of sklearn objects/predictors.
groups: a list of groups (Boolean functions, see below).
group: a Boolean function that outputs True if the input is in a group or False
otherwise.
'''
class DecisionList:
    def __init__(self, predictor, transformer=None):
        self.predictors = [predictor]                      # default predictor
        self.groups = [default_group]                      # default group
        if transformer:
            self.transformer = transformer                 # data transformer, implements fit_transform

    def get_group(self, idx):
        return self.groups[idx]

    def get_predictor(self, idx):
        return self.predictors[idx]
    
    def prepend(self, predictor, group):
        assert(len(self.predictors) == len(self.groups))
        self.predictors.insert(0, predictor)# append to front of decision list
        self.groups.insert(0, group)        # append to front of decision list

    def predict(self, X):
        y_pred = np.full_like(range(len(X)), np.nan, dtype=np.double)
        
        # self.groups is the Decision List's "personal" ordering
        for i, group in enumerate(self.groups):
            predictor = self.predictors[i]
            grp_indices = group_check(X, group)
            
            # Test each feature to check group membership
            not_predicted_yet = grp_indices & np.isnan(y_pred)
            if not_predicted_yet.any():
                #y_pred[not_predicted_yet] = predictor.predict(X[not_predicted_yet, :]) # changd for df indexing
                X_not_pred = X[not_predicted_yet]
                if self.transformer:
                    y_pred[not_predicted_yet] = predictor.predict(self.transformer.transform(X_not_pred))
                else:
                    y_pred[not_predicted_yet] = predictor.predict(X[not_predicted_yet,:])

        return y_pred

    def __call__(self, X):
        return self.predict(X)

'''
The group that contains every example.
input: X (np.ndarray, the examples)
'''
def default_group(x):
    return (x)

'''
Checks if a batch of examples falls in a group.
input: X (np.ndarray, the examples), group (Boolean function)
output: membership (np.ndarray of Boolean values, len(X) long)
'''
def group_check(X, group):
    membership = np.zeros_like(range(len(X)), dtype=bool)
    membership[np.where(group(X))[0]] = True
    return membership