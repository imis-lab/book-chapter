"""
This script contains functions that 
select data from the Neo4j database.
"""

def get_communities_filenames(database):
    """
    This function retrieves all filenames (and the file count) 
    for every community of similar documents.
    """
    query = ('MATCH (p:Paper) RETURN p.community, '
             'collect(p.filename) AS files, '
             'count(p.filename) AS file_count '
             'ORDER BY file_count DESC')
    results = database.execute(query, 'r')
    return results

def get_communities_tags(database, top_terms = None):
    """
    This function generates the most important terms that describe
    each community of similar documents, and returns them for all communities.
    """
    # Get all intersecting nodes of the speficied community, 
    # ranked by their in-degree (which shows to how many documents they belong to).
    # and pagerank score in descending order.
    top_tags = {}
    query = ('MATCH p=((p1:Paper)-[:includes]->(w:Word)) '
             'WITH p1.community as community, w, count(p) as degree '
             'WHERE degree > 1 '
             'WITH community as com, w.key as word, w.pagerank as pagerank, degree as deg '
             'ORDER BY com, deg DESC, pagerank DESC '
             'RETURN com, collect([word, pagerank, deg])')
    communities = database.execute(query, 'r')

    # Get the top tags from the tags and scores list.
    for [community, tags_scores] in communities:
        # Get all top terms for this community.
        if top_terms is None: 
            top_tags[community] = [tag[0] for tag in tags_scores]
        else:
            top_tags[community] = [tag[0] for tag in tags_scores[:top_terms]]
    return top_tags


def get_positive_examples(database, limit, train_set = True):
    relationship = 'co_author_early' if train_set else 'co_author_late'
    query = (
        f'MATCH (author:Author)-[:{relationship}]->(other:Author) '
         'WITH id(author) AS node1, id(other) AS node2, 1 AS label, rand() AS random '
         'WHERE random > 0.5 '
        f'RETURN node1, node2, label LIMIT {limit}'
    )
    return database.execute(query, 'r')

def get_negative_examples(database, limit, train_set = True, min_hops = 2, max_hops = 3):
    relationship = 'co_author_early' if train_set else 'co_author_late'
    query = (
        f'MATCH (author:Author)-[:{relationship}*{min_hops}..{max_hops}]-(other:Author) '
        f'WHERE author <> other AND NOT (author)-[:{relationship}]-(other) '
         'WITH id(author) AS node1, id(other) AS node2, 0 AS label, rand() AS random '
         'WHERE random > 0.5 '
        f'RETURN node1, node2, label LIMIT {limit}'
    )
    return database.execute(query, 'r')

def create_graph_features(database, data, train_set):
    relationship = 'co_author_early' if train_set else 'co_author_late'
    similarity_edge = 'is_similar_early' if train_set else 'is_similar_late'
    #relationship = 'co_author'

    query = (
    f'UNWIND {data} AS pair '
    'MATCH (p1) WHERE id(p1) = pair[0] '
    'MATCH (p2) WHERE id(p2) = pair[1] '
    f'OPTIONAL MATCH (p1)-[r:{similarity_edge}]-(p2) '
    'RETURN pair[0] AS node1, '
    '       pair[1] AS node2, '
    '       gds.alpha.linkprediction.adamicAdar('
    f'       p1, p2, {{relationshipQuery: "{relationship}"}}) AS aa, '
    '       gds.alpha.linkprediction.commonNeighbors('
    f'       p1, p2, {{relationshipQuery: "{relationship}"}}) AS cn, '
    '       gds.alpha.linkprediction.preferentialAttachment('
    f'       p1, p2, {{relationshipQuery: "{relationship}"}}) AS pa, '
    '       gds.alpha.linkprediction.totalNeighbors('
    f'       p1, p2, {{relationshipQuery: "{relationship}"}}) AS tn, '
    '       r.score AS similarity, '
    '       pair[2] AS label       '
    )
    return database.execute(query, 'r')

def get_author_filenames(database, author_id):
    query = (
    'MATCH (a:Author)-[:writes]->(p:Paper) '
    f'WHERE id(a)={author_id} RETURN id(a), collect(p.filename)'
    )
    return database.execute(query, 'r')

def get_filename_community(database, filename):
    query = (
    f"MATCH (p:Paper) WHERE p.filename='{filename}' RETURN p.community"
    )
    return database.execute(query, 'r')

def get_filenames_community(database):
    query = (
    f'MATCH (p:Paper) RETURN p.filename, p.community'
    )
    return database.execute(query, 'r')