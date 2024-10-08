import numpy as np


def calc_loss_for_model(model, loss_fn, X, y, g, weights_obj=None, type_pred='probabilities'):
    """
    Calculate the loss for the model
    """

    y_pred = model.predict(X, type_pred=type_pred)

    # if no weights object, then calculate the loss without weights
    if weights_obj is None:
        loss = loss_fn(y, y_pred)
    else:
        # get the weights based on g
        w = weights_obj.assign_weights(g)

        # calculate the loss with weights
        loss = loss_fn(y, y_pred, sample_weight=w)

    return loss



def calc_worst_group_loss(model, loss_fn, X, y, g):
    """
    Calculate the worst group loss for the model
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
