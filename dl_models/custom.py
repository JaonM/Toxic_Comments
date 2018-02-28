# -*- coding:utf-8 -*-
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class RocCallback(Callback):
    def __init__(self, x_train, y_train, x_valid, y_valid):
        Callback.__init__()
        self.x = x_train
        self.y = y_train
        self.x_val = x_valid
        self.y_val = y_valid

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
