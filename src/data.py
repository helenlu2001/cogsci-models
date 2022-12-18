# Load in the data
import pandas as pd
import itertools as it

def read_form_b():
    form_b = pd.read_csv('../form-b.csv')

    # remove the extra columns and rows
    form_b = form_b.iloc[:,:-6] # remove delta columns
    print(form_b.shape)

    # fix the column names to use hierarchical indexing
    experiments = ['cookies', 'cat', 'laundry', 'apples']
    form_b_questions = [
        ('plan time', 1), ('plan time', 2), 
        ('complete time', 1), ('complete time', 2),
        ('plan prob', 1), ('plan prob', 2),
        ('complete prob', 1), ('complete prob', 2),
        ('partial difficulty', 'main'),
        ('complete difficulty', 'main'),
    ]
    form_b_expanded_toolset = [2, 1, 2, 1]
    form_b_columns = pd.MultiIndex.from_tuples(
        [(e,t,
            'main' if n in [form_b_expanded_toolset[i], 'main'] else 'other'
        ) for (i,e),(t,n) in it.product(enumerate(experiments), form_b_questions)],
        names=['experiment', 'question', 'toolset'])
    form_b.columns = form_b_columns
    return form_b

if __name__ == '__main__':
    form_b = read_form_b()
    print(form_b.head())
    print(form_b.columns)