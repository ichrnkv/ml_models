def class_counts(rows):
    counts = dict()
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Condition:
    """
    Условие для разбиения данных
    """
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if isinstance(val, int) or isinstance(val, float):
            # если вещественный признак
            return val >= self.value
        else:
            # если категориальный признак
            return val == self.value

    def __repr__(self):
        condition = '=='
        if isinstance(self.value, int) or isinstance(self.value, float):
            condition = '>='
        return 'Is {} {} {}?'.format(header[self.column], condition, str(self.value))


def partition(rows, question):
    """
    Разбивает датасет по ответу на вопрос
    """
    true_rows, false_rows = list(), list()
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """
    Расчет коэффициента (неопределенности) Джини
    """
    counts = class_counts(rows)
    gini_imp = 1
    for label in counts:
        prob = counts[label]/float(len(rows))
        gini_imp -= prob**2
    return gini_imp


def info_gain(left, right, current):
    """
    Расчет прироста информации после разбиения
    """
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1 - p) * gini(right)


def best_split(rows):
    """
    Находит наилучшше по IG (приросту информации)
    разделение данных на 2 части
    """
    best_gain, best_condition = 0, None
    current = gini(rows)
    n_features = len(rows[0]) - 1  # кол-во признаков

    for column in range(n_features):  # для каждого признака
        values = set(row[column] for row in rows)  # уникальные значения признака
        for value in values:  # для каждого значения признака
            condition = Condition(column, value)
            true_rows, false_rows = partition(rows, condition)

            # если нет разделения:
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current)
            if gain >= best_gain:
                best_gain, best_condition = gain, condition

    return best_gain, best_condition


class Leaf:
    """
    Сохраняет лист дерева
    """
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class DecisionNode:
    """
    Сохраняет элемент разбиения: услове и две части данных,
    удовлетворяющую (true) условию и нет (false)
    """

    def __init__(self,
                 condition,
                 true_part,
                 false_part):
        self.condition = condition
        self.true_part = true_part
        self.false_part = false_part


def fit_tree(rows):
    """
    Рекурсивно строит дерево решений
    """
    gain, condition = best_split(rows)

    if gain == 0:  # если нет прироста информации
        return Leaf(rows)

    true_rows, false_rows = partition(rows, condition)

    true_part = fit_tree(true_rows)
    false_part = fit_tree(false_rows)

    return DecisionNode(condition, true_part, false_part)


def predict(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.condition.match(row):
        return predict(row, node.true_part)
    else:
        return predict(row, node.false_part)


if __name__ == '__main__':
    training_data = [
        ['Green', 3, 4, 'Apple'],
        ['Yellow', 3, 5, 'Apple'],
        ['Red', 1,  1, 'Grape'],
        ['Red', 1, 7, 'Grape'],
        ['Yellow', 3, 5, 'Lemon'],
    ]
    header = ["color", "diameter", "weight", "label"]

    tree = fit_tree(training_data)

    for row in training_data:
        print("Actual: {}. Predicted: {}".format(row[-1], predict(row, tree)))
