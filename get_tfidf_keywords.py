
import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from tqdm import tqdm
del_words = ['toefl', 'writing', 'task', 'ielts', '2', 'essay', 'topic', 'sat',
            'ibt', 'gre', 'clep']
# nltk.data.path = ['nltk_data']
nltk.download('stopwords')
nltk.download('punkt')
stopwords_list = nltk.corpus.stopwords.words('english')

def get_kws(data):
    orig_essay_list = data['essay'].apply(lambda x: ' '.join(x.split(' Ċ ')))
    essays_list = []
    for essay in orig_essay_list:
        words = nltk.word_tokenize(essay.lower())
        filtered_words = filter(lambda _:_ not in stopwords_list and _ not in del_words, words)
        filtered_para = ' '.join(filtered_words)
        essays_list.append(filtered_para)
    
    tfidf_vec = TfidfVectorizer()
    tfidf = tfidf_vec.fit_transform(essays_list)

    key_words_list = []
    tfidfs_list = []
    for doc_tfidf in tqdm(tfidf, ncols=30, total=tfidf.shape[0]):
        word_for_sort = {
            'word': tfidf_vec.get_feature_names_out(),
            'tfidf': doc_tfidf.toarray()[0].tolist()
        }
        word_for_sort_df = pd.DataFrame(word_for_sort)
        word_for_sort_df = word_for_sort_df[word_for_sort_df['tfidf'] != 0.0]
        word_for_sort_df = word_for_sort_df.sort_values(by='tfidf', ascending=False)
        selected_word_df = word_for_sort_df
        key_words = selected_word_df['word'].tolist()
        tfidfs = selected_word_df['tfidf'].tolist()
        key_words_list.append(key_words)
        tfidfs_list.append(tfidfs)

    kw_with_score_list = [list(zip(key_words, tfidfs)) for key_words, tfidfs in zip(key_words_list, tfidfs_list)]
    return kw_with_score_list


if __name__ == "__main__":

    train_data = pd.read_csv('dataset/ef_train.csv')
    val_data = pd.read_csv('dataset/ef_dev.csv')
    test_data = pd.read_csv('dataset/ef_test.csv')
    all_data = pd.concat([train_data, val_data, test_data])

    train_data['essay_keywords_list'] = get_kws(train_data)
    train_data.to_csv('dataset/ef_train_w_kw.csv', index=False)

    all_data['essay_keywords_list'] = get_kws(all_data)
    new_val_data = all_data[len(train_data):len(train_data)+len(val_data)]
    new_test_data = all_data[len(train_data)+len(val_data):]
    new_val_data.to_csv('dataset/ef_dev_w_kw.csv', index=False)
    new_test_data.to_csv('dataset/ef_test_w_kw.csv', index=False)
