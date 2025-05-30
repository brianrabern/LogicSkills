# Countermodel Checker

Evaluate whether a user-supplied structure is a **countermodel** to a given sentence/argument

A **countermodel** is a fully specified interpretation that makes the sentence **false**. This system checks countermodels by converting them to SMT-LIB and verifying satisfiability using Z3.

---

## Overview

Given:

* A **FOL sentence** φ (e.g., `∀x(Mx → Pcx)`)
* A **Schema of Abbreviation** (SoA): the set of non-logical symbols in φ (e.g., `{M, P, c}`)
* A **user-provided model** specifying interpretations for all symbols in the SoA

The checker determines whether the model falsifies the sentence.

---

### User Model (Python dict)

Fully specifies a finite model:

```python
user_model = {
    "Domain": [0, 1, 2],
    "c": 0,                    # Constant interpretation
    "M": [1, 2],               # Unary predicate extension
    "P": [[0, 1], [0, 2]],     # Binary predicate extension
}
```

---

## Method

The system performs the following steps:

### 1. Convert the model to SMT-LIB assertions

Includes:

* Constant and predicate declarations
* Ground facts matching the user’s specification
* **Domain closure** to restrict quantifiers:

  ```lisp
  (assert (forall ((x Object)) (or (= x d0) (= x d1) (= x d2))))
  ```

### 2. Negate the sentence

To test whether the sentence is false in the model:

```lisp
(assert (not (forall ((x Object)) (=> (M x) (P c x)))))
```

### 3. Merge all into a single SMT-LIB script

Includes:

* Sort and function declarations
* Model assertions
* Negated sentence

### 4. Run Z3

Use Z3 to evaluate satisfiability of the combined assertions.

---

## Interpreting Results

| Z3 Output | Meaning                                              | Verdict              |
| --------- | ---------------------------------------------------- | -------------------- |
| `sat`     | Model is consistent with `¬φ` → φ is **false**       | ✅ Valid countermodel |
| `unsat`   | Model contradicts `¬φ` → φ is **true** in this model | ❌ Not a countermodel |

---

## Requirements for a Valid Model

A valid model must:

* Assign values to all SoA symbols (`c`, `M`, `P`, etc.)
* Explicitly declare the finite domain (e.g., `[0, 1, 2]`)
* Fully specify the extension of each predicate
* Enforce domain closure (e.g., quantifiers range only over declared elements)

Underspecified models (e.g., missing `M` or `P`, or ambiguous domain) are **not** accepted as valid countermodels.
