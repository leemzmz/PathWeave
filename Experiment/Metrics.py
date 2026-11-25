import numpy as np


def _rankData(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index
        while (
            end + 1 < len(values)
            and values[order[end + 1]] == values[order[index]]
        ):
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def calculatePearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator == 0:
        return 0.0
    return float(np.sum(x * y) / denominator)


def calculateSpearman(x, y):
    return calculatePearson(_rankData(x), _rankData(y))
