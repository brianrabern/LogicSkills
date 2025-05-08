class EnglishLexicon:
    def __init__(self):
        self.names = {
            "a": {"name": "Alfred", "gender": "male"},
            "b": {"name": "Lewis", "gender": "male"},
            "c": {"name": "Hazel", "gender": "female"},
        }

        self.predicates = {
            "F": {
                "template": "[1] will run",
                "negated_template": "[1] won't run",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "future",
            },
            "G": {
                "template": "[1] will attack",
                "negated_template": "[1] won't attack",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "future",
            },
            "H": {
                "template": "[1] drank",
                "negated_template": "[1] didn't drink",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "I": {
                "template": "[1] ate",
                "negated_template": "[1] didn't eat",
                "arity": 1,
                "structure": "verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "J": {
                "template": "[1] is asleep",
                "negated_template": "[1] isn't asleep",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "K": {
                "template": "[1] is happy",
                "negated_template": "[1] isn't happy",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "L": {
                "template": "[1] is hungry",
                "negated_template": "[1] isn't hungry",
                "arity": 1,
                "structure": "copula+adjective",
                "semantic_type": "state",
                "tense": "present",
            },
            "M": {
                "template": "[1] is a donkey",
                "negated_template": "[1] isn't a donkey",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "N": {
                "template": "[1] is a human",
                "negated_template": "[1] isn't a human",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "O": {
                "template": "[1] is a monkey",
                "negated_template": "[1] isn't a monkey",
                "arity": 1,
                "structure": "copula+noun",
                "semantic_type": "kind",
                "tense": "present",
            },
            "P": {
                "template": "[1] saw [2]",
                "negated_template": "[1] doesn't see [2]",
                "arity": 2,
                "structure": "transitive_verb",
                "semantic_type": "action",
                "tense": "present",
            },
            "Q": {
                "template": "[1] kicked [2]",
                "negated_template": "[1] didn't kick [2]",
                "arity": 2,
                "structure": "transitive_verb",
                "semantic_type": "action",
                "tense": "past",
            },
            "R": {
                "template": "[1] chased [2]",
                "negated_template": "[1] didn't chase [2]",
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
