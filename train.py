"""
This module contains the training function required to train the RNN models in
the jupyter notebooks associated with the "Predicting Confusion from
Eye-Tracking Data with Recurrent Neural Networks" experiments. The functions
herein are common to all experiments.
"""

import random
import numpy as np

import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

from utils import check_metrics
from utils import batch_accuracy


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

def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through model during training.

        Usage: after loss.backwards()
        "plot_grad_flow(self.model.named_parameters())" to visualize the
        gradient flow.

        source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8

        Args:
            named_parameters (PyTorch Parameters): named parameters to be visualized
    """

    ave_grads = []
    layers = []
    for param_name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in param_name):
            layers.append(param_name)
            ave_grads.append(param.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def train(model,
          epochs,
          criterion_type,
          optimizer_type,
          train_loader,
          val_loader,
          print_every,
          plot_every,
          early_stopping=False,
          early_stopping_metric='val_auc',
          early_stopping_patience=4,
          rate_decay=False,
          rate_decay_patience=5,
          initial_learning_rate=0.001,
          model_name='best_rnn',
          min_auc=0.0,
          verbose=False,
          return_thresh=False):

    """
        Trains the given model according to the input arguments.

    Args:
        models (PyTorch model): The RNN to be trained.
        epochs (int): max number of iterations over the training dataset
        criterion (PyTorch loss): the loss function to use while backpropagating
            to train model
        optimizer (PyTorch optimizer): the optimization algorithm to be used
            train parameters of model
        train_loader (PyTorch DataLoader): the training dataloader that will be
            iterated through
        val_loader (PyTorch DataLoader): the validation dataloader that will be
            used for testing generalization
        print_every (int): the number of batches to pass between printing
            average loss per batch for the batches since the last print
        model_name (string): the name with which to save the model to the
            working directory, which will be model_name.pt
        early_stopping (boolean): stop when validation metric of choice doesn't
            improve for patience epochs
        early_stopping_metric (string): the validation set metric to be
            monitored. Can be either 'val_auc' or 'val_loss'.
        patience (int): number of epochs with no validation accuracy imporvement
            to do before stopping
        min_auc (float): the min. auc over validation set that must be achieved
            before saving the model.
        verbose (boolean): print metrics each epoch if True.
        val_thresh (boolean): when True, the function returns the optimal
            threshold used to compute metrics over the validation set.

    Returns:

        if return_thresh:
             model (Pytorch model): the trained RNN
             training_accs (list)
             validation_accs (list)
             training_losses (list)
             training_aucs (list)
             validation_losses (list)
             validation_recalls (list)
             validation_specifs (list)
             validation_aucs (list)
             val_thresh (float)

        else val_thresh is not returned

    """

    model = model.train()

    best_auc = min_auc
    best_val_loss = 10000
    epochs_no_improvement = 0

    # hold metrics to be tracked across training
    training_losses = []
    validation_recalls = []
    training_accs = []
    training_aucs = []
    validation_accs = []
    validation_losses = []
    validation_specifs = []
    validation_aucs = []

    # Check untrained metrics
    val_acc, \
    val_recall, \
    val_specif, \
    auc, \
    val_loss = check_metrics(model, val_loader, verbose=False)

    validation_accs.append(val_acc)
    validation_losses.append(val_loss)
    validation_recalls.append(val_recall)
    validation_specifs.append(val_specif)
    validation_aucs.append(auc)

    training_acc, \
    training_recall, \
    training_specif, \
    training_auc, \
    training_loss = check_metrics(model, train_loader, verbose=False)

    training_accs.append(training_acc)
    training_losses.append(training_loss)
    training_aucs.append(training_auc)

    if verbose:
        print("METRICS OF UNTRAINED MODEL")
        print("validation accuracy: ", val_acc)
        print("validation loss: ", val_loss)
        print("validation recall: ", val_recall)
        print("validation specificity: ", val_specif)
        print("validation AUC:, ", auc)
        print("training AUC: ", training_auc)

    learning_rate = initial_learning_rate
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("ERROR: optimizer " + str(optimizer_type) + " not supported.")
        return
    if criterion_type == 'NLLLoss':
        criterion = nn.NLLLoss()
        criterion = criterion.to(DEVICE)
    else:
        print("ERROR: criterion " + str(criterion_type) + " not supported.")
        return

    for epoch in range(epochs):
        # batch-wise metrics to be tracked
        training_accuracy = 0.0 # for printing accuracy over training set over epoch w/o recomputing
        running_acc = 0.0
        running_loss = 0.0
        plot_running_acc = 0.0
        plot_running_loss = 0.0
        num_batches = 0
        torch.manual_seed(MANUAL_SEED)

        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            hidden = model.init_hidden(batch_size=labels.shape[0])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            for j in range(MAX_SEQUENCE_LENGTH):
                outputs, hidden = model(inputs[:, j].unsqueeze(1).float(), hidden)

            loss = criterion(outputs, labels.squeeze(0).long()).sum()
            loss.backward()
            if verbose:
                plot_grad_flow(model.named_parameters())
            optimizer.step()

            # Update tracked metrics for batch
            batch_acc = batch_accuracy(outputs, labels)
            training_accuracy += batch_acc
            running_loss += loss.item()
            running_acc += batch_acc
            plot_running_loss += loss.item()
            plot_running_acc += batch_acc

            if i % print_every == (print_every-1):
                print('[epoch: %d, batches: %5d] loss: %.5f | accuracy: %.5f'
                      % (epoch + 1, i + 1, running_loss/print_every,
                         running_acc/print_every))

                running_loss = 0.0
                running_acc = 0.0

            if i % plot_every == (plot_every-1):
                training_accs.append(plot_running_acc/plot_every)
                training_losses.append(plot_running_loss/plot_every)

                plot_running_loss = 0.0
                plot_running_acc = 0.0

                val_acc, \
                val_recall, \
                val_specif, \
                auc, \
                val_loss = check_metrics(model, val_loader, verbose=False)


                validation_accs.append(val_acc)
                validation_losses.append(val_loss)
                validation_recalls.append(val_recall)
                validation_specifs.append(val_specif)
                validation_aucs.append(auc)

                training_acc, \
                training_recall, \
                training_specif, \
                training_auc, \
                training_loss = check_metrics(model, train_loader, verbose=False)
                training_aucs.append(training_auc)

            num_batches += 1

        # Update tracked metrics for epoch: accuracy, recall, specificity, auc, loss
        val_acc, \
        val_recall, \
        val_specif, \
        val_auc, \
        val_loss = check_metrics(model, val_loader, verbose=False)

        train_acc = training_accuracy/num_batches

        if verbose:
            print("Training accuracy for epoch: ", train_acc)
            print("validation accuracy: ", val_acc)
            print("validation loss: ", val_loss)
            print("validation recall: ", val_recall)
            print("validation specificity: ", val_specif)
            print("validation AUC: ", val_auc)

        if early_stopping:
            if early_stopping_metric == 'val_loss' and val_loss < best_val_loss:
                print("Old best val_loss: ", best_val_loss)
                print("New best val_loss: ", val_loss)
                print("Validation AUC: ", val_auc)
                best_val_loss = val_loss
                torch.save(model.state_dict(), './'+ model_name +'.pt')
                print("New best model found. Saving now.")
                epochs_no_improvement = 0
            elif early_stopping_metric == 'val_auc' and val_auc > best_auc:
                print("Old best val AUC: ", best_auc)
                print("New best val AUC: ", val_auc)
                best_auc = val_auc
                torch.save(model.state_dict(), './'+ model_name +'.pt')
                print("New best model found. Saving now.")
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
            if epochs_no_improvement == early_stopping_patience:
                print("No decrease in validation loss in %d epochs. Stopping" +
                      "training early." % early_stopping_patience)
                break
        if rate_decay and epochs_no_improvement > 0 and \
           (epochs_no_improvement % rate_decay_patience == 0):
            print("No increase in validation AUC score in %d epochs. " +
                  "Reducing learning rate." % rate_decay_patience)

            print("Old learning rate:", learning_rate)
            learning_rate = learning_rate/2.0
            print("New learning rate:", learning_rate)
            model.load_state_dict(torch.load('./'+ model_name +'.pt'))
            if optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            else:
                print("ERROR: optimizer " + str(optimizer_type) +
                      " not supported.")
                return

            val_acc, \
            val_recall, \
            val_specif, \
            auc, \
            val_loss = check_metrics(model, val_loader, verbose=False)

            print("Validation AUC: ", auc)
            print("Validation Loss: ", val_loss)
        print("Epochs without improvement: ", epochs_no_improvement)


    print('Finished Training')
    model.load_state_dict(torch.load('./'+ model_name +'.pt'))
    val_acc, \
    val_recall, \
    val_specif, \
    auc, \
    val_loss, \
    val_thresh = check_metrics(model, val_loader, verbose=False, return_threshold=True)

    validation_accs.append(val_acc)
    validation_losses.append(val_loss)
    validation_recalls.append(val_recall)
    validation_specifs.append(val_specif)
    validation_aucs.append(auc)

    training_acc, \
    training_recall, \
    training_specif, \
    training_auc, \
    training_loss = check_metrics(model, train_loader, verbose=False)

    training_accs.append(training_acc)
    training_losses.append(training_loss)
    training_aucs.append(training_auc)

    if verbose:
        #print("Training accuracy for epoch: ", train_acc)
        print("validation accuracy: ", val_acc)
        print("validation loss: ", val_loss)
        print("validation recall: ", val_recall)
        print("validation specificity: ", val_specif)
        print("validation AUC: ", auc)
        if return_thresh:
            metrics = (training_accs, validation_accs, training_losses,
                       training_aucs, validation_losses, validation_recalls,
                       validation_specifs, validation_aucs, val_thresh)
        else:
            metrics = (training_accs, validation_accs, training_losses,
                       training_aucs, validation_losses, validation_recalls,
                       validation_specifs, validation_aucs)

    return model, metrics
