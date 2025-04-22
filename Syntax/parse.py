from lark import Lark

"""Syntax for first-order logic with quantifiers and mondadic and binary predicates"""

logic_grammar = r"""
start: wff

wff: atom
   | "¬" wff                       -> neg
   | "(" wff CONNECTIVE wff ")"    -> binary
   | QUANTIFIER VARIABLE wff            -> quantified

atom: PREDICATE TERM               -> pred1
    | PREDICATE TERM TERM          -> pred2

QUANTIFIER: "∀" | "∃"
CONNECTIVE: "∧" | "∨" | "→" | "↔"

PREDICATE: /[A-Z]/
VARIABLE: /[s-z]/
CONSTANT: /[a-r]/
TERM: VARIABLE | CONSTANT

%import common.WS_INLINE
%ignore WS_INLINE
"""

# setup parser
parser = Lark(logic_grammar, start="start", parser="lalr")
