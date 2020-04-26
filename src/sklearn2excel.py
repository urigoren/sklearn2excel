from string import ascii_uppercase
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def argmax(x_range, y_range):
    return "XLOOKUP(MAX({x}),{x},{y})".format(x=x_range, y=y_range)


def argmin(x_range, y_range):
    return "XLOOKUP(MIN({x}),{x},{y})".format(x=x_range, y=y_range)


def xl_str(s: str) -> str:
    if isinstance(s, str) and s.startswith('"') and s.endswith('"') and len(s) > 2 and '"' not in s[1:-1]:
        return s
    return '"{x}"'.format(x=str(s).replace('"', '""'))


def xl_num(f: float) -> str:
    return ("{f:0." + str(fp_precision) + "f}").format(f=f)


def xl_array(lst, numeric=False) -> str:
    if numeric==True:
        return "{" + ",".join([xl_num(c) for c in lst]) + "}"
    return "{" + ",".join([xl_str(c) for c in lst]) + "}"


def np2array(X) -> str:
    return "{" + ";".join([",".join([xl_num(a) for a in row]) for row in X]) + "}"


def dict_lookup(lookup, keys, values) -> str:
    return 'XLOOKUP({l},{k},{v})'.format(l=xl_str(lookup), v=xl_array(values), k=xl_array(keys))


def column_letter(i: int) -> str:
    if i <= len(ascii_uppercase):
        return ascii_uppercase[i]
    return column_letter(i // 26) + ascii_uppercase[i % 26]


def row_range(start_col: int, end_col: int, row_num: int) -> str:
    return "$" + column_letter(start_col) + str(row_num) + ":$" + column_letter(end_col) + str(row_num)


def translate_log_reg(model: LogisticRegression) -> str:
    if model.coef_.shape[0] == 1:  # Binary classification
        sigmoid = "1/(1+EXP(-{intercept:0." + str(fp_precision) + "f}-(SUM({weights}*{range}))))"
        xl_range = row_range(0, model.coef_.shape[1] - 1, first_column)
        ret = sigmoid.format(intercept=float(model.intercept_), weights=np2array(model.coef_), range=xl_range)
        ret = "ROUND({f},0)".format(f=ret)
        ret = "INDEX({a}, {i}, 1)".format(a=xl_array(model.classes_), i=ret)
    else:  # multiclass
        xl_range = row_range(0, model.coef_.shape[1] - 1, first_column)
        intercept = "{" + ",".join([xl_num(c) for c in model.intercept_]) + "}"
        sigmoid_matrix = "1/(1-EXP(-{i}-MMULT({x},{w})))".format(w=np2array(model.coef_.T), x=xl_range, i=intercept)
        ret = argmax(sigmoid_matrix, xl_array(model.classes_))
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
fp_precision = 3
