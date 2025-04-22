import json

lexicon = {
    "names": {
        "a": {"name": "Alfred", "gender": "male"},
        "b": {"name": "Lewis", "gender": "male"},
        "c": {"name": "Hazel", "gender": "female"},
    },
    "predicates": {
        "F": {
            "label": "[1] will run",
            "arity": 1,
            "structure": "verb",
            "semantic_type": "action",
            "tense": "future",
        },
        "G": {
            "label": "[1] will attack",
            "arity": 1,
            "structure": "verb",
            "semantic_type": "action",
            "tense": "future",
        },
        "H": {
            "label": "[1] drank",
            "arity": 1,
            "structure": "verb",
            "semantic_type": "action",
            "tense": "past",
        },
        "I": {
            "label": "[1] ate",
            "arity": 1,
            "structure": "verb",
            "semantic_type": "action",
            "tense": "past",
        },
        "J": {
            "label": "[1] is asleep",
            "arity": 1,
            "structure": "copula+adjective",
            "semantic_type": "state",
            "tense": "present",
        },
        "K": {
            "label": "[1] is happy",
            "arity": 1,
            "structure": "copula+adjective",
            "semantic_type": "state",
            "tense": "present",
        },
        "L": {
            "label": "[1] is hungry",
            "arity": 1,
            "structure": "copula+adjective",
            "semantic_type": "state",
            "tense": "present",
        },
        "M": {
            "label": "[1] is a donkey",
            "arity": 1,
            "structure": "copula+noun",
            "semantic_type": "type",
            "tense": "present",
        },
        "N": {
            "label": "[1] is a human",
            "arity": 1,
            "structure": "copula+noun",
            "semantic_type": "type",
            "tense": "present",
        },
        "O": {
            "label": "[1] is a monkey",
            "arity": 1,
            "structure": "copula+noun",
            "semantic_type": "type",
            "tense": "present",
        },
        "P": {
            "label": "[1] sees [2]",
            "arity": 2,
            "structure": "transitive_verb",
            "semantic_type": "action",
            "tense": "present",
        },
        "Q": {
            "label": "[1] kicked [2]",
            "arity": 2,
            "structure": "transitive_verb",
            "semantic_type": "action",
            "tense": "past",
        },
        "R": {
            "label": "[1] chased [2]",
            "arity": 2,
            "structure": "transitive_verb",
            "semantic_type": "action",
            "tense": "past",
        },
    },
}


# Generate atomic sentences
sentences = []

for pred_symbol, pred_template in lexicon["predicates"].items():
    if "[2]" in pred_template:
        for name1 in lexicon["names"]:
            for name2 in lexicon["names"]:
                sentence = (
                    pred_template.replace("[1]", lexicon["names"][name1]).replace(
                        "[2]", lexicon["names"][name2]
                    )
                    + "."
                )
                form = f"{pred_symbol}{name1}{name2}"
                soa = {
                    pred_symbol: pred_template,
                    name1: lexicon["names"][name1],
                    name2: lexicon["names"][name2],
                }
                sentences.append({"sentence": sentence, "soa": soa, "form": form})
                neg_pred_template = neg_predicates[pred_symbol]
                negated_sentence = (
                    neg_pred_template.replace("[1]", lexicon["names"][name1]).replace(
                        "[2]", lexicon["names"][name2]
                    )
                    + "."
                )
                negated_form = f"¬{pred_symbol}{name1}{name2}"
                sentences.append(
                    {"sentence": negated_sentence, "soa": soa, "form": negated_form}
                )
    else:
        for name in lexicon["names"]:
            sentence = pred_template.replace("[1]", lexicon["names"][name]) + "."
            form = f"{pred_symbol}{name}"
            soa = {pred_symbol: pred_template, name: lexicon["names"][name]}
            sentences.append({"sentence": sentence, "soa": soa, "form": form})
            neg_pred_template = neg_predicates[pred_symbol]
            negated_sentence = (
                neg_pred_template.replace("[1]", lexicon["names"][name]) + "."
            )
            negated_form = f"¬{pred_symbol}{name}"
            sentences.append(
                {"sentence": negated_sentence, "soa": soa, "form": negated_form}
            )


# binary connective sentences
def make_connective_entries(entry1, entry2):
    # Combine sentences
    s1 = entry1["sentence"].rstrip(".")
    s2 = entry2["sentence"].rstrip(".")
    conjunction = f"{s1} and {s2}."
    disjunction = f"{s1} or {s2}."
    conditional = f"If {s1}, then {s2}."
    biconditional = f"{s1} if and only if {s2}."

    # Merge soa
    soa = {**entry1["soa"], **entry2["soa"]}

    # Combine forms
    conjunction_form = f"({entry1['form']}∧{entry2['form']})"
    disjunction_form = f"({entry1['form']}∨{entry2['form']})"
    conditional_form = f"({entry1['form']}→{entry2['form']})"
    biconditional_form = f"({entry1['form']}↔{entry2['form']})"

    return [
        {"sentence": conjunction, "soa": soa, "form": conjunction_form},
        {"sentence": disjunction, "soa": soa, "form": disjunction_form},
        {"sentence": conditional, "soa": soa, "form": conditional_form},
        {"sentence": biconditional, "soa": soa, "form": biconditional_form},
    ]


def make_quantified_entries():
    restrictors = {"elf": "M", "human": "N", "monkey": "O"}
    quantified_sentences = []
    for pred_symbol, pred_template in lexicon["predicates"].items():
        if "[2]" in pred_template:
            for restrictor, restrictor_symbol in restrictors.items():
                for name in lexicon["names"]:
                    universal_sentence = (
                        pred_template.replace("[1]", f"Every {restrictor}").replace(
                            "[2]", lexicon["names"][name]
                        )
                        + "."
                    )
                    universal_form = f"∀x({restrictor_symbol}x→{pred_symbol}{name})"
                    soa = {
                        restrictor_symbol: lexicon["predicates"][restrictor_symbol],
                        pred_symbol: pred_template,
                        name: lexicon["names"][name],
                    }
                    quantified_sentences.append(
                        {
                            "sentence": universal_sentence,
                            "soa": soa,
                            "form": universal_form,
                        }
                    )

                    if restrictor == "human" or restrictor == "monkey":
                        existential_sentence = (
                            pred_template.replace("[1]", f"A {restrictor}").replace(
                                "[2]", lexicon["names"][name]
                            )
                            + "."
                        )
                    else:
                        existential_sentence = (
                            pred_template.replace("[1]", f"An {restrictor}").replace(
                                "[2]", lexicon["names"][name]
                            )
                            + "."
                        )

                    existential_form = f"∃x({restrictor_symbol}x∧{pred_symbol}{name})"
                    soa = {
                        restrictor_symbol: lexicon["predicates"][restrictor_symbol],
                        pred_symbol: pred_template,
                        name: lexicon["names"][name],
                    }
                    quantified_sentences.append(
                        {
                            "sentence": existential_sentence,
                            "soa": soa,
                            "form": existential_form,
                        }
                    )

        else:
            if pred_symbol in restrictors.values():
                continue
            for restrictor, restrictor_symbol in restrictors.items():
                universal_sentence = (
                    pred_template.replace("[1]", f"Every {restrictor}") + "."
                )
                form = f"∀x({restrictor_symbol}x→{pred_symbol}x)"
                soa = {
                    restrictor_symbol: lexicon["predicates"][restrictor_symbol],
                    pred_symbol: pred_template,
                }
                quantified_sentences.append(
                    {"sentence": universal_sentence, "soa": soa, "form": form}
                )
                if restrictor == "human" or restrictor == "monkey":
                    existential_sentence = (
                        pred_template.replace("[1]", f"A {restrictor}") + "."
                    )
                else:
                    existential_sentence = (
                        pred_template.replace("[1]", f"An {restrictor}") + "."
                    )
                form = f"∃x({restrictor_symbol}x∧{pred_symbol}x)"
                soa = {
                    restrictor_symbol: lexicon["predicates"][restrictor_symbol],
                    pred_symbol: pred_template,
                }
                quantified_sentences.append(
                    {"sentence": existential_sentence, "soa": soa, "form": form}
                )

    return quantified_sentences


c = make_quantified_entries()
print(json.dumps(c, indent=2, ensure_ascii=False))
print(len(c))
