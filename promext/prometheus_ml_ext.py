"""
models.py

This file contains the classes that represent ML-related metrics 
for users to monitor.
"""

from abc import ABC, abstractmethod

import prometheus_client as prom


class MLMetric(ABC):
    """
    An abstract class that defines the interface for a metric.
    """

    def __init__(self, name, description, keys):
        self.name = name
        self.description = description
        self.keys = keys
        self.create_prometheus_metrics()

    def log(self, metric, value, keys):
        """
        Logs a metric to Prometheus.
        """
        if isinstance(keys, list) or isinstance(keys, tuple):
            metric.labels(**keys).set(value)
        else:
            metric.labels(keys).set(value)

    def logBatch(self, metric, values, keys):
        """
        Logs a batch of metrics to Prometheus.
        """
        for value, key in zip(values, keys):
            self.log(metric, value, key)

    @abstractmethod
    def create_prometheus_metrics(self):
        """
        Creates the prometheus metrics.
        """
        pass

    @abstractmethod
    def logOutput(self, pred, keys):
        """
        Logs a prediction.
        """
        pass

    @abstractmethod
    def logOutputs(self, preds, keys):
        """
        Logs a batch of predictions.
        """
        pass

    @abstractmethod
    def logFeedback(self, true, keys):
        """
        Logs true label.
        """
        pass

    @abstractmethod
    def logFeedbacks(self, trues, keys):
        """
        Logs a batch of true labels.
        """
        pass

    @abstractmethod
    def get_query_strings(self):
        """
        Returns the query strings for the metrics.
        """
        pass


class BinaryClassificationMetric(MLMetric):
    """
    A class that represents a binary classification metric.
    """

    def __init__(self, name, description, keys, threshold=0.5):
        self.threshold = threshold
        self.pred_metric_name = name + "_prediction"
        self.label_metric_name = name + "_label"
        super().__init__(name, description, keys)

    def create_prometheus_metrics(self):
        self.pred_metric = prom.Gauge(
            self.pred_metric_name, self.description, labelnames=self.keys
        )
        self.label_metric = prom.Gauge(
            self.label_metric_name,
            self.description,
            labelnames=self.keys,
        )

    def logOutput(self, pred, keys):
        self.log(self.pred_metric, pred, keys)

    def logOutputs(self, preds, keys):
        self.logBatch(self.pred_metric, preds, keys)

    def _check_label_validity(self, label):
        """
        Checks if a label is valid.

        Returns True if label is 0 or 1.
        """
        return label == 0 or label == 1

    def logFeedback(self, true, keys):
        assert self._check_label_validity(true), "Label must be 0 or 1."
        self.log(self.label_metric, true, keys)

    def logFeedbacks(self, trues, keys):
        for true in trues:
            assert self._check_label_validity(true), "Label must be 0 or 1."
        self.logBatch(self.label_metric, trues, keys)

    def get_query_strings(self):
        """
        Returns the query strings for the metrics.

        Accuracy, precision, recall
        """
        accuracy_query = f"""
            count(abs({self.label_metric_name} - on ({','.join(self.keys)}) {self.pred_metric_name}) < {self.threshold}) / count({self.label_metric_name} - on ({','.join(self.keys)}) {self.pred_metric_name})
        """

        precision_query = f"""
            count( ({self.label_metric_name} * on ({','.join(self.keys)}) {self.pred_metric_name}) > {self.threshold}) / count(({self.pred_metric_name} and on ({','.join(self.keys)}) {self.label_metric_name}) > {self.threshold})
        """

        recall_query = f"""
            count( ({self.label_metric_name} * on ({','.join(self.keys)}) {self.pred_metric_name}) > {self.threshold}) / count(({self.label_metric_name} and on ({','.join(self.keys)}) {self.pred_metric_name}) == 1)
        """

        return {
            "accuracy": accuracy_query.strip(),
            "precision": precision_query.strip(),
            "recall": recall_query.strip(),
        }
