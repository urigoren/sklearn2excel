from string import ascii_uppercase
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def argmax(x_range, y_range):
    return "XLOOKUP(MAX({x}),{x},{y})".format(x=x_range, y=y_range)


def argmin(x_range, y_range):
    return "XLOOKUP(MIN({x}),{x},{y})".format(x=x_range, y=y_range)


def xl_str(s: str) -> str:
    return '"{x}"'.format(x=str(s).replace('"', '""'))


def dict_lookup(lookup, keys, values) -> str:
    return 'XLOOKUP("{l}","{k}","{v}")'.format(l=xl_str(lookup), v=xl_str(values), k=xl_str(keys))


def column_letter(i: int) -> str:
    if i <= len(ascii_uppercase):
        return ascii_uppercase[i]
    return column_letter(i // 26) + ascii_uppercase[i % 26]


def row_range(start_col: int, end_col: int, row_num: int) -> str:
    return "$" + column_letter(start_col) + str(row_num) + ":$" + column_letter(end_col) + str(row_num)


def xl_array(lst) -> str:
    return "{" + ",".join([xl_str(c) for c in lst]) + "}"


def np2array(X) -> str:
    return "{" + ";".join([",".join([f"{a:0.3f}" for a in row]) for row in X]) + "}"


def translate_log_reg(model: LogisticRegression) -> str:
    sigmoid = "1/(1+EXP(-{intercept:0.3f}-(SUM({{{weights}}}*{range}))))"
    if model.coef_.shape[0] == 1:  # Binary classification
        xl_range = row_range(0, len(model.coef_) - 1, first_column)
        weights = ",".join([f"{w:0.3f}" for w in model.coef_[0]])
        ret = sigmoid.format(intercept=float(model.intercept_), weights=weights, range=xl_range)
        ret = "ROUND({f},0)".format(f=ret)
        ret = "INDEX({a}, {i}, 1)".format(a=xl_array(model.classes_), i=ret)
    else:  # multiclass
        xl_range = row_range(0, model.coef_.shape[1] - 1, first_column)
        intercept = "{" + ",".join([f"{c:0.3f}" for c in model.intercept_]) + "}"
        sigmoids = "1/(1-EXP(-{i}-MMULT({x},{w})))".format(w=np2array(model.coef_.T), x=xl_range, i=intercept)
        ret = argmax(sigmoids, xl_array(model.classes_))
    return "=" + ret.replace("(--", "(")


def translate_decision_tree(model: DecisionTreeClassifier) -> str:
    pass


def translate(model) -> str:
    if isinstance(model, LogisticRegression):
        return translate_log_reg(model)
    if isinstance(model, DecisionTreeClassifier):
        return translate_decision_tree(model)
    raise NotImplementedError("{t} is not supported".format(t=type(model).__name__))


first_column = 2
