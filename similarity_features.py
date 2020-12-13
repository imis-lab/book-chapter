import argparse
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from GraphOfDocs_Representation import select
from GraphOfDocs_Representation import utils

tags_per_community = {}
filenames_community = {}


def _tfidf_feature_selector(x, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit_transform(x)
    return set(vectorizer.get_feature_names())


def _get_text_from_paper_file(input_dir, sha):
    with open(f'{input_dir}/dataset/{sha}.json') as f:
        data = json.load(f)
        title = data['metadata']['title']
        abstract = ' '.join([t['text'] for t in data['abstract']])
    text = title + ' ' + abstract
    return text


def _get_important_terms_of_author(database, vocabulary: set, author_id, input_dir):
    """Create a list of the most representative terms for a given author.

    :param database: the database connector
    :param vocabulary: the vocabulary with the most important terms for the whole corpus of text based on GraFS
    :param author_id: the author_id
    :returns: a list of unique terms
    """
    result = select.get_author_filenames(database, author_id)
    if result is None or len(result) == 0:
        return []

    author_filenames = result[0][1]
    important_terms = set()
    for sha in author_filenames:
        with open(f'{input_dir}/dataset/{sha}.json') as f:
            data = json.load(f)
            title = data['metadata']['title']
            abstract = ' '.join([t['text'] for t in data['abstract']])
        text = title + ' ' + abstract
        text = text.lower()
        tokens = text.split()
        for token in tokens:
            if token in vocabulary:
                important_terms.add(token)
    return list(important_terms)


def _calculate_similarities(database, vocabulary: set, df, input_dir):
    """Calculate the similarity between the two authors of each row of the given dataframe.

    The similarity score is based on the Jaccard index.

    :param database: the database connector
    :param vocabulary: the vocabulary with the most important terms for the whole corpus of text based on GraFS
    :returns: a list of similarity scores
    """
    similarities = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        author_id1 = int(row['node1'])
        author_id2 = int(row['node2'])

        author_terms1 = _get_important_terms_of_author(database, vocabulary, author_id1, input_dir)
        author_terms2 = _get_important_terms_of_author(database, vocabulary, author_id2, input_dir)

        similarities.append(utils.jaccard_similarity(author_terms1, author_terms2))
    return similarities


def run(args):
    datasets = [
        ['datasets/dataset1/fully_balanced/train_balanced_668.csv',
         'datasets/dataset1/fully_balanced/test_balanced_840.csv'],
        ['datasets/dataset2/fully_balanced/train_balanced_858.csv',
         'datasets/dataset2/fully_balanced/test_balanced_1566.csv'],
        ['datasets/dataset3/fully_balanced/train_balanced_1726.csv',
         'datasets/dataset3/fully_balanced/test_balanced_2636.csv'],
        ['datasets/dataset4/fully_balanced/train_balanced_3346.csv',
         'datasets/dataset4/fully_balanced/test_balanced_7798.csv'],
        ['datasets/dataset5/fully_balanced/train_balanced_5042.csv',
         'datasets/dataset5/fully_balanced/test_balanced_12976.csv'],
        ['datasets/dataset6/fully_balanced/train_balanced_5296.csv',
         'datasets/dataset6/fully_balanced/test_balanced_16276.csv'],
        ['datasets/dataset7/fully_balanced/train_balanced_6210.csv',
         'datasets/dataset7/fully_balanced/test_balanced_25900.csv'],
        ['datasets/dataset8/fully_balanced/train_balanced_8578.csv',
         'datasets/dataset8/fully_balanced/test_balanced_34586.csv'],
        ['datasets/dataset9/fully_balanced/train_balanced_13034.csv',
         'datasets/dataset9/fully_balanced/test_balanced_49236.csv']
    ]

    database = utils.connect_to_the_database()
    global filenames_community
    filenames_community = dict(select.get_filenames_community(database))
    filenames_per_community = select.get_communities_filenames(database)
    filenames_per_community = {f[0]: f[1] for f in filenames_per_community}

    top_x = [5, 100, 250]
    for train_file, test_file in datasets:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Create a corpus of files for the tfidf feature selection process.
        corpus = []
        for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
            author_id1 = int(row['node1'])
            result = select.get_author_filenames(database, int(author_id1))
            if not (result is None or len(result) == 0):
                author_filenames = result[0][1]
                for filename in author_filenames:
                    corpus.append(_get_text_from_paper_file(args.input_dir, filename))
            author_id2 = int(row['node2'])
            result = select.get_author_filenames(database, int(author_id2))
            if not (result is None or len(result) == 0):
                author_filenames = result[0][1]
                for filename in author_filenames:
                    corpus.append(_get_text_from_paper_file(args.input_dir, filename))
        tfidf_vocabulary = _tfidf_feature_selector(corpus, max_features=2000)
        train_df['similarity_tfidf'] = _calculate_similarities(database, tfidf_vocabulary, train_df, args.input_dir)
        test_df['similarity_tfidf'] = _calculate_similarities(database, tfidf_vocabulary, test_df, args.input_dir)

        for top_n in top_x:
            global tags_per_community
            tags_per_community = select.get_communities_tags(database, top_terms=top_n)
            vocabulary = sum([tags_per_community[key] for key in tags_per_community.keys()], [])
            vocabulary = set(vocabulary)
            print('Number of features:%s %s |%s|' % (train_file, top_n, len(vocabulary)))
            for key in tags_per_community.keys():
                tags_per_community[key] = set(tags_per_community[key])
            field_name = f'similarity_top_{top_n}'
            print(train_file, top_n)
            train_df[field_name] = _calculate_similarities(database, vocabulary, train_df, args.input_dir)
            print(test_file, top_n)
            test_df[field_name] = _calculate_similarities(database, vocabulary, test_df, args.input_dir)

        train_file = train_file.replace('.csv', '_enriched.csv')
        train_df.to_csv(train_file)
        test_file = test_file.replace('.csv', '_enriched.csv')
        test_df.to_csv(test_file)
    utils.disconnect_from_the_database(database)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create the similarity features for each edges of the nine datasets based on the feature selection"
                    " made by the GraFS model",
    )
    parser.add_argument(
        "--input-directory",
        help="Input directory containing the metadata and the raw text data of the dataset",
        dest="input_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    run(args)
