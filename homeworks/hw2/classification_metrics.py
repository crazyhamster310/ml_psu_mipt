import numpy as np


class ClassificationMetrics:
    @staticmethod
    def accuracy(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with accuracy score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        return np.mean(labels == preds)

    @staticmethod
    def precision(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with precision score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        tp = np.sum(np.logical_and(labels == 1, preds == 1))
        fp = np.sum(np.logical_and(labels == 0, preds == 1))
        return tp / (tp + fp)

    @staticmethod
    def recall(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with recall score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """
        
        tp = np.sum(np.logical_and(labels == 1, preds == 1))
        fn = np.sum(np.logical_and(labels == 1, preds == 0))
        return tp / (tp + fn)

    @staticmethod
    def f1(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with f1 score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        precision = ClassificationMetrics.precision(labels, preds)
        recall = ClassificationMetrics.recall(labels, preds)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def f_beta(labels, preds, beta=1):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)
        beta : float

        Return : float
            single number with f_beta score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        precision = ClassificationMetrics.precision(labels, preds)
        recall = ClassificationMetrics.recall(labels, preds)
        beta_sqr = beta ** 2
        return (1 + beta_sqr) * precision * recall / (beta_sqr * precision + recall)
    