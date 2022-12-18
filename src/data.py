# Load in the data
import pandas as pd
import itertools as it

EXPERIMENTS = ['cookies', 'cat', 'laundry', 'apples']

def read_form_b():
    form_b = pd.read_csv('../form-b.csv')

    # remove the extra columns and rows
    form_b = form_b.iloc[:,:-6] # remove delta columns

    # fix the column names to use hierarchical indexing
    form_b_questions = [
        ('plan', 'time', 1), ('plan', 'time', 2), 
        ('complete', 'time', 1), ('complete', 'time', 2),
        ('plan', 'prob', 1), ('plan', 'prob', 2),
        ('complete', 'prob', 1), ('complete', 'prob', 2),
        ('partial', 'difficulty', 'main'),
        ('execute', 'difficulty', 'main'),
    ]
    form_b_expanded_toolset = [2, 1, 2, 1]
    form_b_columns = pd.MultiIndex.from_tuples(
        [(e,t,m,
            'main' if n in [form_b_expanded_toolset[i], 'main'] else 'other'
        ) for (i,e),(t,m,n) in it.product(enumerate(EXPERIMENTS), form_b_questions)],
        names=['experiment', 'type', 'measurement', 'toolset'])
    form_b.columns = form_b_columns
    return form_b

if __name__ == '__main__':
    form_b = read_form_b()
    print(form_b.head())
    print(form_b.columns)