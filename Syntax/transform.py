from lark import Transformer


class FOLTransformer(Transformer):
    """
    Transformer for First-Order Logic expressions.
    Processes parsed items from the grammar and converts them into a structured tuple format
    """

    def start(self, items):
        return items[0]

    def wff(self, items):
        return items[0]

    def pred1(self, items):
        return ("pred1", items[0].value, items[1].value)

    def pred2(self, items):
        return ("pred2", items[0].value, items[1].value, items[2].value)

    def neg(self, items):
        return ("not", items[0])

    def binary(self, items):
        op_map = {"∧": "and", "∨": "or", "→": "imp", "↔": "iff"}
        op = op_map[items[1].value]
        return (op, items[0], items[2])

    def quantified(self, items):
        quant = "forall" if items[0].value == "∀" else "exists"
        var = items[1].value
        scope = items[2]
        return (quant, var, scope)


transformer = FOLTransformer()
