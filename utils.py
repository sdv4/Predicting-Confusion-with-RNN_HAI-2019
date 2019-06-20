"""
This module contains miscellaneous functions required to run the Jupyter
notebooks associated with the "Predicting Confusion from Eye-Tracking Data with
Recurrent Neural Networks" experiments. The functions contained herein are
common to all notebooks, unless otherwise defined locally.
"""

import pickle
import random
import numpy as np

import torch
from torch import nn

from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import VPacker

MANUAL_SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(MANUAL_SEED)
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
else:
    torch.manual_seed(MANUAL_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_SEQUENCE_LENGTH = 150
INPUT_SIZE = 14


def pickle_loader(input_file_name):
    """ Finishes the pre-processing of data items.

        Args:
            input_file_name (string): the name of the data item to be loaded
        Returns:
            item (numpy array): the fully processed data item

    """
    file = open(input_file_name, 'rb')
    item = pickle.load(file)
    item = item.values[-MAX_SEQUENCE_LENGTH:]
    if len(item) < MAX_SEQUENCE_LENGTH:
        num_zeros_to_pad = (MAX_SEQUENCE_LENGTH)-len(item)
        item = np.append(np.zeros((num_zeros_to_pad, INPUT_SIZE)), item, axis=0)
    file.close()
    return item


def check_metrics(model,
                  data_loader,
                  verbose=False,
                  threshold=None,
                  return_threshold=False):
    """ Computes the model accuracy, recall, specificity, ROC, FP rate, TP
        rate, thresholds for ROC curve on a given dataset.

    Args:
        model (PyTorch model): RNN whose accuracy will be tested
        data_loader (PyTorch DataLoader): the dataset over which metric are calculted
        verbose (Boolean): if True will print the accuracy as a %, size of the
            dataset, recall, and specificity
        threshold (float): if given, this threshold is used to calculate the metrics
        return_threshold (boolean): if True, the calculated optimal threshold is returned

    Returns

    (float, float, float, float, float): if return_threshold is False: accruacy, recall,
                                        specificity, AUC, and NLLLoss. All in [0.0,1.0].
    (float, float, float, float, float, float): if return_threshold is True: accruacy,
                                            recall, specificity, AUC, NLLLoss, and
                                            threshold. All in [0.0,1.0].
    """
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    y_0_scores = np.array([])

    # Make predictions
    with torch.no_grad():
        criterion = nn.NLLLoss()
        criterion = criterion.to(DEVICE)
        loss = 0
        model = model.eval()
        batches = 1
        for i, data in enumerate(data_loader, 1):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            hidden = model.init_hidden(batch_size=labels.shape[0])
            y_true = np.concatenate((labels.cpu().numpy(), y_true), axis=None)
            for j in range(MAX_SEQUENCE_LENGTH):
                outputs, hidden = model(inputs[:, j].unsqueeze(1).float(), hidden)

            loss += (criterion(outputs, labels.squeeze(0).long()).item())

            total += labels.size(0)
            y_0_scores = np.concatenate((torch.exp(outputs).cpu().numpy()[:, 0],
                                         y_0_scores), axis=0)
            batches = i
        # Compute metrics
        loss = loss / batches

        # no option to use 0 (confused) as positive class label,
        # so flip lables so that the true positive class (0) is
        # represented with 1s:
        y_true_flipped = y_true.copy()
        y_true_flipped[y_true == 1] = 0
        y_true_flipped[y_true == 0] = 1

        auc = roc_auc_score(y_true_flipped, y_0_scores)

        # roc_curve expects y_scores to be probability values of the positive class
        fpr, tpr, thresholds = roc_curve(y_true, y_0_scores, pos_label=0)
        if threshold is not None:
            print("Calculating metrics with given threshold")
            y_pred = y_0_scores < threshold # use < so that neg class maintains the 1 label
            correct = sum(y_pred == y_true)
            accuracy = correct/len(y_pred)
            recall = recall_score(y_true, y_pred, pos_label=0)
            specificity = recall_score(y_true, y_pred, pos_label=1)
        else:
            if return_threshold:
                recall, specificity, accuracy, \
                opt_thresh = optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                       tpr[1:],
                                                                       fpr[1:],
                                                                       y_true,
                                                                       y_0_scores,
                                                                       True)
            else:
                recall, specificity, \
                accuracy = optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                     tpr[1:],
                                                                     fpr[1:],
                                                                     y_true,
                                                                     y_0_scores)
        if verbose:
            print('Accuracy of the network on the ' + str(total) +
                  ' data items: %f %%' % (correct / total))
            print("Loss: ", loss)
            print("Recall/Sensitivity: ", recall)
            print("Specificity: ", specificity)
            print("AUC: ", auc)

    model = model.train()
    if return_threshold:
        metrics = (accuracy, recall, specificity, auc, loss, opt_thresh)
    else:
        metrics = (accuracy, recall, specificity, auc, loss)

    return metrics


def optimal_threshold_sensitivity_specificity(thresholds,
                                              true_pos_rates,
                                              false_pos_rates,
                                              y_true,
                                              y_0_scores,
                                              return_thresh=False):
    """ Finds the optimal threshold then calculates sensitivity and specificity.

    Args:
        thresholds (list): the list of thresholds used in computing the ROC score.
        true_pos_rates (list): the TP rate corresponding to each thresholds.
        false_pos_rates (list): the FP rate corresponding to each threshold.
        y_true (list): the ground truth labels of the dataset over which
            sensitivity and specificity will be calculated.
        y_0_scores (list): the model's probability that each item in the dataset is
            class 0, (i.e. confused).
        return_thresh (boolean): if True, the calculated optimal threshold is returned

    Returns:
        sensitivity (float): True positive rate when optimal threshold is used
        specificity (float): True negative rate when optimal threshold is used
        accuracy (float): the percentage of lables that were correctly predicted, in [0,1]
        best_threshold (float): if return_thresh is true, this value is the
            decition threshold that maximized combined sensitivity and specificity
    """

    best_threshold = 0.5
    dist = -1
    for i, thresh in enumerate(thresholds):
        current_dist = np.sqrt((np.power(1-true_pos_rates[i], 2)) +
                               (np.power(false_pos_rates[i], 2)))
        if dist == -1 or current_dist <= dist:
            dist = current_dist
            best_threshold = thresh

    y_pred = (y_0_scores >= best_threshold) == False
    y_pred = np.array(y_pred, dtype=np.int)
    accuracy = sum(y_pred == y_true)/len(y_true)
    sensitivity = recall_score(y_true, y_pred, pos_label=0)
    specificity = recall_score(y_true, y_pred, pos_label=1)

    if return_thresh:
        metrics = (sensitivity, specificity, accuracy, best_threshold)
    else:
        metrics = (sensitivity, specificity, accuracy)

    return metrics


def batch_accuracy(predictions, ground_truth):
    """ Calculate accuracy of predictions over items in a single batch.

    Args:
        predictions (PyTorch Tensor): the logit output of datum in the batch
        ground_truth (PyTorch): the correct class index of each datum

    Returns
        (float): the % of correct predictions as a value in [0,1]
    """

    correct_predictions = torch.argmax(predictions, dim=1) == ground_truth

    return torch.sum(correct_predictions).item()/len(correct_predictions)



def get_grouped_k_fold_splits(confused_list, not_confused_list, num_folds):
    """ Splits data ensuring no users have data in training and eval sets.

        Args:
            confused_list (list): list of data item names labelled as confused
            not_confused_list (list): list of data item names labelled as not_confused
            num_folds (int): number of folds for cross validation.

        Returns: (in following order)
            train_confused_splits (list): each element is a list containing the
                file names of the data items for this partition of the dataset
            test_confused_splits (list): as above
            train_not_confused_splits (list): as above
            test_not_confused_splits (list): as above
    """

    train_confused_splits = []
    test_confused_splits = []

    # make list where each index corresponds to the "group" (userID)
    confused_groups = [uid.split('_')[0][:-1] for uid in confused_list]
    not_confused_groups = [uid.split('_')[0][:-1] for uid in not_confused_list]

    # get train test splits for confused class
    dummy_y = [1 for i in range(len(confused_list))]
    gkf = GroupKFold(n_splits=num_folds)
    gkf.get_n_splits(X=confused_list, y=dummy_y, groups=confused_groups)
    for train, test in gkf.split(X=confused_list, y=dummy_y, groups=confused_groups):
        train_confused_splits.append([confused_list[i] for i in train])
        test_confused_splits.append([confused_list[i] for i in test])

    train_not_confused_splits = []
    test_not_confused_splits = []

    # get train test splits for not_confused class
    dummy_y = [1 for i in range(len(not_confused_list))]
    gkf = GroupKFold(n_splits=num_folds)
    gkf.get_n_splits(X=not_confused_list, y=dummy_y, groups=not_confused_groups)
    for train, test in gkf.split(X=not_confused_list, y=dummy_y, groups=not_confused_groups):
        train_not_confused_splits.append([not_confused_list[i] for i in train])
        test_not_confused_splits.append([not_confused_list[i] for i in test])

    split = (train_confused_splits, test_confused_splits,
             train_not_confused_splits, test_not_confused_splits)

    return split

def get_train_val_split(confused_items, not_confused_items, percent_val_set):
    """ Grouped split the training set into a training and validation set.

        Args:
            confused_items (list): list of strings; each of which is the name of
                a file containing a data item labelled confused.
            not_confused_items (list): list of strings; each of which is the
                name of a file containing a data item labelled not_confused.

        Returns:
            train_confused (list): list of strings; each is the name of a data
                item in the training set.
            train_not_confused (list): list of strings; each is the name of a
                data item in the training set.
            val_confused (list): list of strings; each is the name of a data
                item in the training set.
            val_not_confused (list): list of strings; each is the name of a
                data item in the training set.
    """

    train_confused = []
    val_confused = []
    train_not_confused = []
    val_not_confused = []

    # make list where each index corresponds to the "group" (userID)
    confused_groups = [uid.split('_')[0][:-1] for uid in confused_items]
    not_confused_groups = [uid.split('_')[0][:-1] for uid in not_confused_items]

    # get train test splits for confused class
    dummy_y = [1 for i in range(len(confused_items))]
    gkf = GroupShuffleSplit(n_splits=1, test_size=percent_val_set, random_state=MANUAL_SEED)
    gkf.get_n_splits(X=confused_items, y=dummy_y, groups=confused_groups)
    for train, test in gkf.split(X=confused_items, y=dummy_y, groups=confused_groups):
        train_confused = [confused_items[i] for i in train]
        val_confused = [confused_items[i] for i in test]


    # get train test splits for not_confused class
    dummy_y = [1 for i in range(len(not_confused_items))]
    gkf = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=MANUAL_SEED)
    gkf.get_n_splits(X=not_confused_items, y=dummy_y, groups=not_confused_groups)
    for train, test in gkf.split(X=not_confused_items, y=dummy_y, groups=not_confused_groups):
        train_not_confused = [not_confused_items[i] for i in train]
        val_not_confused = [not_confused_items[i] for i in test]

    return train_confused, train_not_confused, val_confused, val_not_confused


def plot_metrics(training_accs,
                 training_losses,
                 training_aucs,
                 validation_accs,
                 validation_losses,
                 validation_recalls,
                 validation_specifs,
                 validation_aucs):
    """
        Outputs four graphs showing changes in metric values over training.
        1. Training and validation loss
        2. Training and validation accuracy
        3. Training and validation AUC ROC score
        4. Validation AUC ROC, sensitivity, and specificity
    """

    # Training loss and validation loss
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    num_training_accs = np.arange(0, len(training_accs), 1)
    color = 'tab:red'
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Training Loss (%)', color=color)
    ax1.plot(num_training_accs, training_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.1])

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(num_training_accs, validation_losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.1])
    fig.tight_layout()
    plt.show()

    # Training and validation accuracy
    fig, ax1 = plt.subplots()
    ax1.set_ylim([0.1, 1.0])
    fig.set_size_inches(12, 6)
    num_val_accs = np.arange(0, len(validation_accs), 1)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation accuracy (%)', color=color)
    ax1.plot(num_val_accs, validation_accs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Training accuracy (%)', color=color)
    ax2.plot(num_val_accs, training_accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    # Training and validation AUC ROC score
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    num_val_aucs = np.arange(0, len(validation_aucs), 1)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation AUC', color=color)
    ax1.plot(num_val_aucs, validation_aucs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.0])

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Training AUC', color=color)
    ax2.plot(num_val_aucs, training_aucs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.1])
    fig.tight_layout()
    plt.show()

    # Validation AUC, specificity and sensitivity
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation AUC', color=color)
    ax1.plot(num_val_aucs, validation_aucs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.0])

    ax2 = ax1.twinx()
    ax2.set_ylim([0.0, 1.0])


    ybox1 = TextArea("Sensitivity ", textprops=dict(color='tab:blue',
                                                    rotation=90, ha='left',
                                                    va='bottom'))
    ybox2 = TextArea("and ", textprops=dict(color="k", rotation=90, ha='left',
                                            va='bottom'))
    ybox3 = TextArea("Specificity ", textprops=dict(color='xkcd:azure',
                                                    rotation=90, ha='left',
                                                    va='bottom'))

    ybox = VPacker(children=[ybox1, ybox2, ybox3], align="bottom", pad=0, sep=5)

    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                      bbox_to_anchor=(1.13, 0.25),
                                      bbox_transform=ax2.transAxes, borderpad=0.)

    ax2.add_artist(anchored_ybox)

    color = 'tab:blue'
    ax2.plot(num_val_aucs, validation_recalls, color=color)
    ax2.plot(num_val_aucs, validation_specifs, color='xkcd:azure')

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def get_users(file_names):
    """ Returns the users whose items make up a given list of data items.

        Args:
            file_names (list): list of strings naming data items

        Returns:
            users (list): list of strings where each is a user whose data
                is in file_names
    """

    users = []
    for item in file_names:
        user_number = item.split('_')[0][:-1]
        if user_number not in users:
            users.append(user_number)
    return users
