import hashlib


def ast_from_json(data):
    if isinstance(data, list):
        return tuple(ast_from_json(x) for x in data)
    return data


def canonical_premise_str(premise_ids):
    return ",".join(map(str, sorted(premise_ids)))


def generate_argument_id(premise_ids, conclusion_id):
    full_key = canonical_premise_str(premise_ids) + f",{conclusion_id}"
    return hashlib.sha256(full_key.encode()).hexdigest()[:16]
