from Database.DB import db
from collections import defaultdict

def count_sentences():
    counts = defaultdict(lambda: defaultdict(int))
    for row in db.get_all_sentences():
        counts[row['type']][row['subtype']] += 1

    print('\nSentence Counts by Type and Subtype:')
    print('=' * 50)

    for type_name, subtypes in sorted(counts.items()):
        print(f'\n{type_name.upper()}:')
        for subtype, count in sorted(subtypes.items()):
            print(f'  {subtype or "None"}: {count}')

if __name__ == "__main__":
    count_sentences()
