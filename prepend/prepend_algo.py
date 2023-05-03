import numpy as np
from .dec_list import DecisionList, group_check

'''
prepend_algo: Main function that runs the entire Prepend algorithm on a dataset 
(X, y) and returns a decision list predictor.
X (np.array): Training examples.
y (np.array): Training labels.
model (sklearn predictor): Base model class for the decision list.
loss (sklearn metric): Loss function for the decision list. Should coincide with
the loss function of the hyp_class. Should output a single real number.
groups (list of Boolean functions): A list of Boolean functions, G.
    group: a Boolean function that outputs whether some example x falls in the
    group.
'''
def prepend_algo(X, y, model, loss, groups, params={}, transformer=None, verbose=False, max_iter=500):
    # sorted_X and sorted_y are |G|-long lists, each entry i a list of examples
    # in group i.
    sorted_X, sorted_y = sort_groups(X, y, groups)

    # Initialize the decision list
    h_0 = model(**params)

    if transformer:
        h_0.fit(transformer.fit_transform(X), y)             # at this point, transformer is fitted (only use .transform)
    else:
        h_0.fit(X, y)                                        # Solve ERM for h_0
    dec_list = DecisionList(h_0, transformer=transformer)    # decision list with single ERM classifier
    epsilon = 0

    # Calculate the threshold for prepending (tolerance, or eps_g)]
    #C = 0.05
    #eps_g = [C/np.sqrt(len(sorted_y[i])) for i in range(len(groups))]

    # Calculate the argmin L(h|g) for each group g.
    gp_cond_errs = []
    gp_predictors = []
    for g in range(len(groups)):
        if verbose:
            print("fitting for group {}".format(g))
            print("group {} has {} examples.".format(g, len(sorted_X[g])))
        h = model(**params)
        if transformer:
            h.fit(transformer.transform(sorted_X[g]), sorted_y[g])
        else:
            h.fit(sorted_X[g], sorted_y[g])
        gp_predictors.append(h)

        if transformer:
            gp_cond_errs.append(loss(h.predict(transformer.transform(sorted_X[g])), sorted_y[g]))
        else:
            gp_cond_errs.append(loss(h.predict(sorted_X[g]), sorted_y[g]))

    # Calculate the argmin L(f|g) for each group g.
    dec_list_errs = np.empty(len(groups))
    for g in range(len(groups)):
        dec_list_errs[g] = loss(dec_list.predict(sorted_X[g]), sorted_y[g])

    t = 0
    while True:
        # For tracking decision list improvement
        old_dec_list_errs = dec_list_errs
        
        # argmax error over groups
        diffs = [gp_cond_errs[g] + epsilon - dec_list_errs[g] for g in range(len(groups))]
        min_g = np.argmin(np.array(diffs))

        # Prepend step
        if diffs[min_g] < 0.:
            dec_list.prepend(gp_predictors[min_g], groups[min_g])
            for g in range(len(groups)):
                dec_list_errs[g] = loss(dec_list.predict(sorted_X[g]), sorted_y[g])
            diff_errs = old_dec_list_errs - dec_list_errs
            if verbose:
                print("Iteration t={}: prepended group {}".format(t, min_g))
                print("old errors={}".format(old_dec_list_errs))
                print("new errors={}".format(dec_list_errs))
                print("change in error={}".format(diff_errs))
                print(old_dec_list_errs)
        else:            
            return dec_list
        
        # Stopping conditions
        t += 1
        if t > max_iter:
            break

    return dec_list

'''
sort_groups: Helper function that sorts data into groups, to be indexed the same
as the `groups` variable.
X (np.array): Training examples.
y (np.array): Training labels.
groups (list of Boolean functions): A list of Boolean functions, G.
    group: a Boolean function that outputs whether some example x falls in the
    group.
Output (list of np.arrays): 
soted_X (list): |G|-long list, where each entry corresponds to a
group and stores an np.array of shape (n_g, d), where n_g is the number of
examples X in the group and d is the dimensionality of X.
sorted_y (list): |G|-long list, where each entry corresponds to a group and 
stores and np.array of shape (n_g, ), for the labels y corresponding to each
grouped X.
'''
def sort_groups(X, y, groups):
    sorted_X = []
    sorted_y = []
    for group in groups:
        grp_indices = group_check(X, group)
        #sorted_X.append(X[grp_indices,:])
        sorted_X.append(X[grp_indices])
        sorted_y.append(y[grp_indices])
    assert(len(X) == len(y))
    return sorted_X, sorted_y
