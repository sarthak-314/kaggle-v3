import sys

from sklearn.manifold import trustworthiness 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 
from kaggle_utils.utils import solve_environment
from chai.tensorflow_qa import *

# Competition Specific Constants
COMP_NAME = 'chaii-hindi-and-tamil-question-answering'
DRIVE_DIR = Path('/content/drive/MyDrive/Chai')

INTERNET_AVAILIBLE = True 
try: 
    os.system('pip install wandb')
    import wandb
except: 
    INTERNET_AVAILIBLE = False
    
    
# Termcolor Colors
red = lambda str: colored(str, 'red')
blue = lambda str: colored(str, 'blue')
green = lambda str: colored(str, 'green')
yellow = lambda str: colored(str, 'yellow')


WORD_LENS = [0, 10, 25, 50, 100, 200, 400, 600, 1000, 2000, 4000, 10000, 250000]
SPLIT_ON = '\n' # \n, ।, .

def get_word_len_tokens(word_lens): 
    print(f'Word Lengths:', ','.join(word_lens))
    return [f'[WORD={word_len}]' for word_len in word_lens]

def add_word_len_tokens(df, word_lens=WORD_LENS, split_on=SPLIT_ON): 
    df_dict = {'context_with_token': [], 'id': [], 'answer_start_temp': []}
    for i, row in df.iterrows(): 
        lines = []
        word_count = 0
        answer_found = False
        answer_start = row.answer_start
        for line in row.context.split(split_on):
            for lower, upper in zip(word_lens, word_lens[1:]): 
                if word_count < upper: 
                    break
            token = f'[WORD={lower}]'
            add_token = len(line) > 8
            if add_token: 
                word_count += len(line) + 1
                line = token + line
                lines.append(line)
                if not answer_found: 
                    answer_start += 1
            else: 
                word_count += len(line)
                lines.append(line)
            
            if row.answer_text in line: 
                answer_found = True
        context = split_on.join(lines)
        df_dict['context_with_token'].append(context)
        df_dict['id'].append(row.id)
        df_dict['answer_start_temp'].append(answer_start)
    df = df.merge(pd.DataFrame(df_dict))
    return df