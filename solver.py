import cv2 as cv
import numpy as np

from classifier import get_expression, load_best_model
from detector import detect_characters


def solve_and_draw(model, image_path, kernel=(5, 5)):
    bbs = detect_characters(image_path, kernel=kernel)
    limits = [(x, y, x + w, y + h) for (x, y, w, h) in bbs]
    [_, _, x_max, y_max] = list(np.amax(limits, axis=0))
    [x_min, y_min, _, _] = list(np.amin(limits, axis=0))
    exp = get_expression(model, image_path, kernel=kernel)
    img = cv.imread(image_path)
    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv.putText(img, str(pretty(exp)), (x_min - 5, y_min - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (200, 15, 0), 1)
    try:
        cv.putText(img, str(evaluate_expression(exp)), (x_max + 5, y_min - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (200, 15, 0), 1)
    except ValueError:
        print("Unable to evaluate expression")
    cv.imshow('Expression', img)
    cv.waitKey(0)


def op(digit1, operator, digit2):
    digit1, digit2 = float(digit1), float(digit2)
    if operator == '_':
        res = digit1 * digit2
    elif operator == '%':
        res = digit1 / digit2
    elif operator == '+':
        res = digit1 + digit2
    elif operator == '-':
        res = digit1 - digit2
    return res


def is_operator(char):
    return char == '+' or char == '-' or char == '%' or char == '_'


def is_float(slice):
    if is_operator(slice[0]):
        return False
    try:
        float(slice)
        return True
    except ValueError:
        return False


def numbers_op(exp: str, op: str):
    op_ind = exp.find(op)
    start_ind = -1
    end_ind = -1
    for ind in range(1, op_ind + 1):
        i = op_ind - ind
        slice = exp[i:op_ind]
        if is_float(slice):
            start_ind = i
        else:
            break
    for i in range(op_ind + 1, len(exp)):
        slice = exp[op_ind + 1:i + 1]
        if is_float(slice):
            end_ind = i
        else:
            break
    return exp[:start_ind], exp[start_ind:end_ind + 1], exp[end_ind + 1:]


def branch(exp, operator):
    f, s, t = numbers_op(exp, operator)
    res = my_solver(s)
    if f != "":
        first = my_solver(f[:-1])
        res = op(first, f[-1], res)
    if t != "":
        third = my_solver(t[1:])
        res = op(res, t[0], third)
    return res


def terminal(exp):
    ops = ['+', '-', '%', '_']
    brackets = ['[', ']']
    count_ops = {op: exp.count(op) for op in ops}
    count_br = {br: exp.count(br) for br in brackets}
    if sum(count_ops.values()) == 1 and sum(count_br.values()) == 0:
        for key in count_ops.keys():
            if count_ops[key] == 1:
                return key
    return None


def remove_brackets(exp):
    closed_ind = exp.find(']')
    open_ind = exp.rfind('[', 0, closed_ind)
    while open_ind != -1 and closed_ind != -1:
        bracket_exp = exp[open_ind + 1:closed_ind]
        left_exp = exp[:open_ind]
        right_exp = exp[closed_ind + 1:]
        exp = left_exp + str(my_solver(bracket_exp)) + right_exp
        open_ind, closed_ind = exp.rfind('['), exp.find(']')
    return exp


def my_solver(exp: str):
    operator = terminal(exp)
    if operator is not None:
        [first, second] = exp.split(operator)
        return op(first, operator, second)
    mul_ind = exp.find('_')
    div_ind = exp.find('%')
    if mul_ind < div_ind and mul_ind != -1:
        return branch(exp, '_')
    if div_ind != -1:
        return branch(exp, '%')
    add_ind = exp.find('+')
    sub_ind = exp.find('-')
    if add_ind < sub_ind and add_ind != -1:
        return branch(exp, '+')
    if sub_ind != -1:
        return branch(exp, '-')
    if exp != "":
        return int(exp)


def uglify(exp):
    new = exp.replace('/', '%')
    new = new.replace('*', '_')
    new = new.replace('(', '[')
    new = new.replace(')', ']')
    return new


def pretty(exp):
    new = exp.replace('%', '/')
    new = new.replace('_', '*')
    new = new.replace('[', '(')
    new = new.replace(']', ')')
    return new


def evaluate_expression(exp):
    return my_solver(remove_brackets(exp))


def test():
    examples = ['10+1-1+0/1', '5-3+8/2*1', '4+4+4/8', '(7*1)-(4-1)+5', '8-2/1+3*(3-1)']
    examples = [(uglify(ex), float(eval(ex))) for ex in examples]
    for i in range(0, len(examples)):
        example = examples[i]
        res = evaluate_expression(example[0])
        print(f"Calculated:{res}, expected:{example[1]}")

# test()

# model = load_best_model()
# solve_and_draw(model, 'notebooks/data/handwritten_ex/20211231_191020.jpg')
