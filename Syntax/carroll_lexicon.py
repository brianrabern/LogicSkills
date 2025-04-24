class CarrollLexicon:
    def __init__(self):
        self.names = {
            "a": {"name": "Zindle", "gender": "female"},
            "b": {"name": "Rafin", "gender": "male"},
            "c": {"name": "Bungo", "gender": "male"},
        }

        self.predicates = {
            "F": {
                "template": "[1] will whiffle",
                "negated_template": "[1] won't whiffle",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "future",
            },
            "G": {
                "template": "[1] will burble",
                "negated_template": "[1] won't burble",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "future",
            },
            "H": {
                "template": "[1] gyred",
                "negated_template": "[1] didn't gyre",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "I": {
                "template": "[1] gimbled",
                "negated_template": "[1] didn't gimble",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "J": {
                "template": "[1] is mimsy",
                "negated_template": "[1] isn't mimsy",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "K": {
                "template": "[1] is uffish",
                "negated_template": "[1] isn't uffish",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "L": {
                "template": "[1] is beamish",
                "negated_template": "[1] isn't beamish",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "M": {
                "template": "[1] is a tove",
                "negated_template": "[1] isn't a tove",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "N": {
                "template": "[1] is a borogove",
                "negated_template": "[1] isn't a borogove",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "O": {
                "template": "[1] is a rath",
                "negated_template": "[1] isn't a rath",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "P": {
                "template": "[1] chortled at [2]",
                "negated_template": "[1] didn't chortle at [2]",
                "arity": 2,
                "structure": "transitive_verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "Q": {
                "template": "[1] galumphed over [2]",
                "negated_template": "[1] didn't galumph over [2]",
                "arity": 2,
                "structure": "transitive_verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "R": {
                "template": "[1] snicker-snacked [2]",
                "negated_template": "[1] didn't snicker-snacked [2]",
                "arity": 2,
                "structure": "transitive_verb",
                "semantic_type": "action",
                "tense": "past",
            },
        }

    def get_name(self, symbol):
        return self.names[symbol]["name"]

    def get_gender(self, symbol):
        return self.names[symbol]["gender"]

    def get_predicate_template(self, symbol):
        return self.predicates[symbol]["template"]

    def get_predicate_negated_template(self, symbol):
        return self.predicates[symbol]["negated_template"]

    def get_predicate_arity(self, symbol):
        return self.predicates[symbol]["arity"]

    def is_binary(self, symbol):
        return self.predicates[symbol]["arity"] == 2

    def is_unary(self, symbol):
        return self.predicates[symbol]["arity"] == 1
