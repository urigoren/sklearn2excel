from string import ascii_uppercase
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def argmax(xl_range):
    return "MATCH(MAX({r}),{r},0)".format(r=xl_range)


def argmin(xl_range):
    return "MATCH(MIN({r}),{r},0)".format(r=xl_range)


def column_letter(i: int) -> str:
    if i <= len(ascii_uppercase):
        return ascii_uppercase[i]
    return column_letter(i // 26) + ascii_uppercase[i % 26]


def row_range(start_col: int, end_col: int, row_num: int) -> str:
    return column_letter(start_col) + str(row_num) + ":" + column_letter(end_col) + str(row_num)


def classes_array(model)->str:
    return "{"+",".join(['"'+str(c).replace('"', '""')+'"' for c in model.classes_])+"}"


def translate_log_reg(model: LogisticRegression) -> str:
    sigmoid="1/(1+EXP(-{intercept:0.3f}-(SUM({{{weights}}}*{range}))))"
    if model.coef_.shape[0] ==1: #Binary classification
        xl_range = row_range(0, len(model.classes_), 1)
        ret = sigmoid.format(intercept=float(model.intercept_), weights=",".join([f"{w:0.3f}" for w in model.coef_[0]]), range=xl_range)
        ret = "ROUND({f},0)".format(f=ret)
        ret = "INDEX({a}, {i}, 1)".format(a=classes_array(model),i=ret)
    else: #multiclass
        xl_range = row_range(0, len(model.classes_), 1)
        sigmoids=[]
        for i in range(model.coef_.shape[0]):
            sigmoids.append(sigmoid.format(intercept=model.intercept_[i], weights=",".join([f"{w:0.3f}" for w in model.coef_[i]]), range=xl_range))
        sigmoids = "{" + ",".join(sigmoids) + "}"
        ret = argmax(sigmoids)
        ret = "INDEX({a}, {i}, 1)".format(a=classes_array(model),i=ret)
    return "=" + ret.replace("(--", "(")


def translate_decision_tree(model: DecisionTreeClassifier) -> str:
    pass


def translate(model) -> str:
    if isinstance(model, LogisticRegression):
        return translate_log_reg(model)
    if isinstance(model, DecisionTreeClassifier):
        return translate_decision_tree(model)
    raise NotImplementedError("{t} is not supported".format(t=type(model).__name__))
