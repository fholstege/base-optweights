import numpy as np


def calc_BCE(y, y_pred, sample_weight=None, reduction='mean'):
    """
    Calculate the binary cross entropy loss

    Arguments:
        y: ndarray of shape (n,) representing the true labels
        y: ndarray of shape (n,) representing the predicted labels
        sample_weight: ndarray of shape (n,) representing the sample weights
        reduction: string representing the reduction type, either 'mean' or 'none'

    Returns:
        loss: float representing the binary cross entropy loss
    """
    
    # calculate the binary cross entropy loss
    loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    # if sample weight is not None, multiply the loss with the sample weight
    if sample_weight is not None:
        loss = loss * sample_weight.reshape(-1, 1)
    
    # if reduction is mean, return the mean of the loss
    if reduction == 'mean':
        return np.mean(loss)
    else:
        return loss
   
   

def calc_loss_for_model(model, loss_fn, X, y, g, weights_obj=None, type_pred='probabilities'):
    """
    Calculate the loss for the model

    Arguments:
        model: model object
        loss_fn: loss function
        X: ndarray of shape (n, d) representing the features
        y: ndarray of shape (n,) representing the true labels
        g: ndarray of shape (n,) representing the group labels
        weights_obj: weights object
        type_pred: string representing the type of prediction, either 'probabilities' or 'labels'
    
    Returns:
        loss: float representing the loss

    """

    y_pred = model.predict(X, type_pred=type_pred)

    # if no weights object, then calculate the loss without weights
    if weights_obj is None:
        loss = loss_fn(y, y_pred, sample_weight=None)

    else:
        # get the weights based on g
        w = weights_obj.assign_weights(g)

        # calculate the loss with weights
        loss =  loss_fn(y, y_pred, sample_weight=w)

    return loss



def calc_worst_group_loss(model, loss_fn, X, y, g):
    """
    Calculate the worst group loss for the model
    
    Arguments:
        model: model object
        loss_fn: loss function
        X: ndarray of shape (n, d) representing the features
        y: ndarray of shape (n,) representing the true labels
        g: ndarray of shape (n,) representing the group labels
    
    Returns:
        worst_group_loss: float representing the worst group loss
        loss_dict: dictionary with the loss per group
    
    """

    # get the unique groups
    unique_g = np.unique(g)

    # get the loss for each group
    loss_dict = {}
    for group in unique_g:

        # get the index for the group
        idx = (g == group).squeeze()
        X_g = X[idx, :]
        y_g = y[idx]

        # calculate the loss for the model, for group g
        loss = calc_loss_for_model(model, loss_fn, X_g, y_g, g[idx], weights_obj=None)
        loss_dict[group] = loss

    # get the worst group loss
    worst_group = max(loss_dict, key=loss_dict.get)
    worst_group_loss = loss_dict[worst_group]

    # return the worst group loss and loss per group
    return worst_group_loss, loss_dict


# calculate the worst-group accuracy for each model
def calc_worst_and_weighted_acc(y, y_pred, g):
    """
    Calculate the worst group accuracy and the equal-weighted accuracy

    Arguments:
        y: ndarray of shape (n,) representing the true labels
        y_pred: ndarray of shape (n,) representing the predicted labels
        g: ndarray of shape (n,) representing the group labels
    
    Returns:
        min(accuracy): float representing the worst group accuracy
        weighted_accuracy: float representing the equal-weighted accuracy


    """

    # get the combinations in g, sort from smallest to largest
    groups = list(np.unique(g))
    groups = sorted(groups)

    # correct
    correct = (y == y_pred)

    # get the accuracy per group
    accuracy = np.zeros(len(groups))
    i = 0
    for group in groups:

        # get the index of the group, and get the accuracy
        idx =  (g == group)
        correct_group = correct[idx]
        accuracy_combination = np.mean(correct_group)
        accuracy[i] = accuracy_combination
        i+=1
    
    # calculate the equal-weighted accuracy
    weighted_acc = np.mean(accuracy)

    # return the worst group accuracy, and the equal-weighted accuracy
    return min(accuracy), weighted_acc
   