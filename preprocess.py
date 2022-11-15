import pandas as pd
from tqdm import tqdm

kw_num = 10

essay_len_list = []
len_list = []

for dataset in ['train', 'dev', 'test']:
    file_path = 'dataset/ef_{}_w_kw.csv'.format(dataset)
    data_df = pd.read_csv(file_path)
    essay_with_keywords_list = []
    for keywords_list, essay in tqdm(zip(data_df['essay_keywords_list'], data_df['essay'])):
        keywords_list = eval(keywords_list)
        keywords = [kw[0] for kw in keywords_list[:kw_num]]
        keywords = [keyword + ' # ' + str(i+1) for i, keyword in enumerate(keywords)]
        # keywords_str = ' | '.join(keywords)
        # keywords_str += ' | '
        keywords_str = ' | '.join(keywords) + ' |'
        essay_with_keywords = '[KEYWORDS] ' + keywords_str \
                                + ' [ESSAY] [ESSAY] ' + essay
        essay_with_keywords_list.append(essay_with_keywords)
    data_df['essay_with_keywords'] = essay_with_keywords_list
    data_df.to_csv(file_path, index=False)