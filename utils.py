import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit  # Necessary for using its functions


# task_train = Progress().add_task("[red]Training...", total=100)
# task_test = Progress().add_task("[green]Testing...", total=100)
# 
class learningStat():
    '''
    This class collect the learning statistics over the epoch.

    Usage:

    This class is designed to be used with learningStats instance although it can be used separately.

    >>> trainingStat = learningStat()
    '''
    def __init__(self):
        self.lossSum = 0
        self.correctSamples = 0
        self.numSamples = 0
        self.minloss = None
        self.maxAccuracy = None
        self.lossLog = []
        self.accuracyLog = []
        self.bestLoss = False
        self.bestAccuracy = False

    def reset(self):
        '''
        Reset the learning staistics.
        This should usually be done before the start of an epoch so that new statistics counts can be accumulated.

        Usage:

        >>> trainingStat.reset()
        '''
        self.lossSum = 0
        self.correctSamples = 0
        self.numSamples = 0

    def loss(self):
        '''
        Returns the average loss calculated from the point the stats was reset.

        Usage:

        >>> loss = trainingStat.loss()
        '''
        if self.numSamples > 0:
            return self.lossSum/self.numSamples
        else:
            return None

    def accuracy(self):
        '''
        Returns the average accuracy calculated from the point the stats was reset.

        Usage:

        >>> accuracy = trainingStat.accuracy()
        '''
        if self.numSamples > 0 and self.correctSamples > 0:
            return self.correctSamples/self.numSamples
        else:
            return None

    def update(self):
        '''
        Updates the stats of the current session and resets the measures for next session.

        Usage:

        >>> trainingStat.update()
        '''
        currentLoss = self.loss()
        self.lossLog.append(currentLoss)
        if self.minloss is None:
            self.minloss = currentLoss
        else:
            if currentLoss < self.minloss:
                self.minloss = currentLoss
                self.bestLoss = True
            else:
                self.bestLoss = False
            # self.minloss = self.minloss if self.minloss < currentLoss else currentLoss

        currentAccuracy = self.accuracy()
        self.accuracyLog.append(currentAccuracy)
        if self.maxAccuracy is None:
            self.maxAccuracy = currentAccuracy
        else:
            if currentAccuracy > self.maxAccuracy:
                self.maxAccuracy = currentAccuracy
                self.bestAccuracy = True
            else:
                self.bestAccuracy = False
            # self.maxAccuracy = self.maxAccuracy if self.maxAccuracy > currentAccuracy else currentAccuracy

    def displayString(self):
        loss = self.loss()
        accuracy = self.accuracy()
        minloss = self.minloss
        maxAccuracy = self.maxAccuracy

        if loss is None:    # no stats available
            return 'No testing results'
        elif accuracy is None:
            if minloss is None: # accuracy and minloss stats is not available
                return 'loss = %-11.5g'%(loss)
            else:   # accuracy is not available but minloss is available
                return 'loss = %-11.5g (min = %-11.5g)'%(loss, minloss)
        else:
            if minloss is None and maxAccuracy is None: # minloss and maxAccuracy is available
                return 'loss = %-11.5g        %-11s     accuracy = %.2f%%        %-8s '%(loss, ' ', accuracy*100, ' ')
            else:   # all stats are available
                return 'loss = %-11.5g (min = %-11.5g)    accuracy = %.2f%% (max = %.2f%%)'\
                       %(loss, minloss, accuracy*100, maxAccuracy*100)


class learningStats():
    '''
    This class provides mechanism to collect learning stats for training and testing, and displaying them efficiently.

    Usage:

    .. code-block:: python

        stats = learningStats()

        for epoch in range(100):
            tSt = datetime.now()

            stats.training.reset()
            for i in trainingLoop:
                # other main stuffs
                stats.training.correctSamples += numberOfCorrectClassification
                stats.training.numSamples     += numberOfSamplesProcessed
                stats.training.lossSum        += currentLoss
                stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
            stats.training.update()

            stats.testing.reset()
            for i in testingLoop
                # other main stuffs
                stats.testing.correctSamples += numberOfCorrectClassification
                stats.testing.numSamples     += numberOfSamplesProcessed
                stats.testing.lossSum        += currentLoss
                stats.print(epoch, i)
            stats.training.update()

    '''

    def __init__(self):
        self.linesPrinted = 0
        self.training = learningStat()
        self.testing = learningStat()

    def update(self):
        '''
        Updates the stats for training and testing and resets the measures for next session.

        Usage:

        >>> stats.update()
        '''
        self.training.update()
        self.training.reset()
        self.testing.update()
        self.testing.reset()

    def print(self, epoch, iter=None, timeElapsed=None, header=None, footer=None):
        '''
        Prints the available learning statistics from the current session on the console.
        For Linux systems, prints the data on same terminal space (might not work properly on other systems).

        Arguments:
            * ``epoch``: epoch counter to display (required).
            * ``iter``: iteration counter to display (not required).
            * ``timeElapsed``: runtime information (not required).
            * ``header``: things to be printed before printing learning statistics. Default: ``None``.
            * ``footer``: things to be printed after printing learning statistics. Default: ``None``.

        Usage:

        .. code-block:: python

            # prints stats with epoch index provided
            stats.print(epoch)

            # prints stats with epoch index and iteration index provided
            stats.print(epoch, iter=i)

            # prints stats with epoch index, iteration index and time elapsed information provided
            stats.print(epoch, iter=i, timeElapsed=time)
        '''
        print('\033[%dA' % (self.linesPrinted))

        self.linesPrinted = 1

        epochStr = 'Epoch : %10d' % (epoch)
        iterStr = '' if iter is None else '(i = %7d)' % (iter)
        profileStr = '' if timeElapsed is None else ', %12.4f s elapsed' % timeElapsed

        if header is not None:
            for h in header:
                print('\033[2K' + str(h))
                self.linesPrinted += 1

        print(epochStr + iterStr + profileStr)
        print(self.training.displayString())
        print(self.testing.displayString())
        self.linesPrinted += 3

        if footer is not None:
            for f in footer:
                print('\033[2K' + str(f))
                self.linesPrinted += 1

    def plot(self, figures=(1, 2), saveFig=False, path=''):
        '''
        Plots the available learning statistics.

        Arguments:
            * ``figures``: Index of figure ID to plot on. Default is figure(1) for loss plot and figure(2) for accuracy plot.
            * ``saveFig``(``bool``): flag to save figure into a file.
            * ``path``: path to save the file. Defaule is ``''``.

        Usage:

        .. code-block:: python

            # plot stats
            stats.plot()

            # plot stats figures specified
            stats.print(figures=(10, 11))
        '''
        plt.figure(figures[0])
        plt.cla()
        if len(self.training.lossLog) > 0:
            plt.semilogy(self.training.lossLog, label='Training')
        if len(self.testing.lossLog) > 0:
            plt.semilogy(self.testing.lossLog, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if saveFig is True:
            plt.savefig(path + 'loss.png')
            # plt.close()

        plt.figure(figures[1])
        plt.cla()
        if len(self.training.accuracyLog) > 0:
            plt.plot(self.training.accuracyLog, label='Training')
        if len(self.testing.accuracyLog) > 0:
            plt.plot(self.testing.accuracyLog, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        if saveFig is True:
            plt.savefig(path + 'accuracy.png')
            # plt.close()

    def save(self, filename=''):
        '''
        Saves the learning satatistics logs.

        Arguments:
            * ``filename``: filename to save the logs. ``accuracy.txt`` and ``loss.txt`` will be appended.

        Usage:

        .. code-block:: python

            # save stats
            stats.save()

            # save stats filename specified
            stats.save(filename='Run101-0.001-') # Run101-0.001-accuracy.txt and Run101-0.001-loss.txt
        '''

        with open(filename + 'loss.txt', 'wt') as loss:
            loss.write('#%11s %11s\r\n' % ('Train', 'Test'))
            for i in range(len(self.training.lossLog)):
                loss.write('%12.6g %12.6g \r\n' % (self.training.lossLog[i], self.testing.lossLog[i]))

        with open(filename + 'accuracy.txt', 'wt') as accuracy:
            accuracy.write('#%11s %11s\r\n' % ('Train', 'Test'))
            if self.training.accuracyLog != [None] * len(self.training.accuracyLog):
                for i in range(len(self.training.accuracyLog)):
                    accuracy.write('%12.6g %12.6g \r\n' % (
                        self.training.accuracyLog[i],
                        self.testing.accuracyLog[i] if self.testing.accuracyLog[i] is not None else 0,
                    ))

    def load(self, filename='', numEpoch=None, modulo=1):
        '''
        Loads the learning statistics logs from saved files.

        Arguments:
            * ``filename``: filename to save the logs. ``accuracy.txt`` and ``loss.txt`` will be appended.
            * ``numEpoch``: number of epochs of logs to load. Default: None. ``numEpoch`` will be automatically determined from saved files.
            * ``modulo``: the gap in number of epoch before model was saved.

        Usage:

        .. code-block:: python

            # save stats
            stats.load(epoch=10)

            # save stats filename specified
            stats.save(filename='Run101-0.001-', epoch=50) # Run101-0.001-accuracy.txt and Run101-0.001-loss.txt
        '''
        saved = {}
        saved['accuracy'] = np.loadtxt(filename + 'accuracy.txt')
        saved['loss'] = np.loadtxt(filename + 'loss.txt')
        if numEpoch is None:
            saved['epoch'] = saved['loss'].shape[0] // modulo * modulo + 1
        else:
            saved['epoch'] = numEpoch

        self.training.lossLog = saved['loss'][:saved['epoch'], 0].tolist()
        self.testing.lossLog = saved['loss'][:saved['epoch'], 1].tolist()
        self.training.minloss = saved['loss'][:saved['epoch'], 0].min()
        self.testing.minloss = saved['loss'][:saved['epoch'], 1].min()
        self.training.accuracyLog = saved['accuracy'][:saved['epoch'], 0].tolist()
        self.testing.accuracyLog = saved['accuracy'][:saved['epoch'], 1].tolist()
        self.training.maxAccuracy = saved['accuracy'][:saved['epoch'], 0].max()
        self.testing.maxAccuracy = saved['accuracy'][:saved['epoch'], 1].max()

        return saved['epoch']


class aboutCudaDevices():
    def __init__(self):
        pass

    def num_devices(self):
        """Return number of devices connected."""
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected."""
        num = cuda.Device.count()
        print("%d device(s) found:" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    def mem_info(self):
        """Get available and total memory of all devices."""
        available, total = cuda.mem_get_info()
        print("Available: %.2f GB\nTotal:     %.2f GB" % (available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def info(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1), cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" % (cuda.Device(i).total_memory() / 1e9))
        return string


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = np.Inf
        self.delta = delta

    def __call__(self, val, model, epoch):

        score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, val, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, val, epoch)
            self.counter = 0

    def save_checkpoint(self, network, val, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Accuracy increased ({self.val_min:.6f} --> {val:.6f}).  Saving model ...')
        state = {
            'net': network.state_dict(),
            'loss': val,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        self.val_min = val
