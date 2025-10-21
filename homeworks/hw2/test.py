import classification_metrics
import numpy as np

print(classification_metrics.ClassificationMetrics.accuracy(np.array([1] * 99 + [0]), np.array([1] * 100)))
print()

print(classification_metrics.ClassificationMetrics.precision(np.array([1] * 99 + [0]), np.array([1] * 100)))
print(classification_metrics.ClassificationMetrics.precision(np.array([1] * 99 + [0]), np.array([1] * 99 + [0])))
print()

print(classification_metrics.ClassificationMetrics.recall(np.array([1] * 99 + [0]), np.array([1] * 100)))
print(classification_metrics.ClassificationMetrics.recall(np.array([1] * 100), np.array([1] * 99 + [0])))
print()

print(classification_metrics.ClassificationMetrics.f1(np.array([1] * 99 + [0]), np.array([1] * 100)))
print(classification_metrics.ClassificationMetrics.f_beta(np.array([1] * 99 + [0]), np.array([1] * 100)))