import json


def clean_sentences(entries):
    for entry in entries:
        entry["sentence"] = (
            entry["sentence"].replace("[1] ", "").replace("[2]", "").strip()
        )
    return entries


# Load your file
with open("combined.json", "r") as f:
    data = json.load(f)

# Clean the sentences
cleaned_data = clean_sentences(data)

# Save the result
with open("combined1.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
CREATE TABLE `sentences` (
  `id` varchar(56) NOT NULL,
  `sentence` varchar(56) DEFAULT NULL,
  `type` varchar(255) DEFAULT NULL,
  `soa` varchar(255) DEFAULT NULL,
  `form` varchar(255) DEFAULT NULL,
  `ast` varchar(255) DEFAULT NULL,
  `base` tinyint(1) DEFAULT NULL,
  `time_created` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
{"id": "1",
"sentence": "If Sam is not a knight, then he is not honest.",
"type": "conditional",
"soa": {
      "c": "Sam",
      "F": "[1] is a knight",
      "G": "[1] is honest"
    },
"form": [
      "(¬Fc→¬Gc)"
    ],
"ast": [
      "imp",
      [
        "not",
        [
          "pred1",
          "F",
          "c"
        ]
      ],
      [
        "not",
        [
          "pred1",
          "G",
          "c"
        ]
      ]
    ],
"base": 1,
"time_created": 1697059200}
