import json

import elasticsearch.helpers
import requests

from utils import elastic_connection, INDEX_NAME

API_URL = 'https://en.wikipedia.org/w/api.php'


def batch(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def api_req(params, session=requests.Session()):
    res = session.get(API_URL, params=dict(params, format='json', formatversion=2))
    if res.status_code != 200:
        raise Exception("api completely failed. try later?")
    return res.json()
    


TITLE_TO_ID = {}


def fetch_docs(labeled, batch_size=20):
    all_titles = set(s['title'] for s in labeled)
    for titles in batch(all_titles, batch_size):
        res = api_req({
            'action': 'query',
            'prop': 'cirrusdoc',
            'titles': '|'.join(titles)
        })
        norm = {n['to']: n['from'] for n in res['query'].get('normalized', [])}
        for page in res['query']['pages']:
            try:
                doc = page['cirrusdoc'][0]
            except (KeyError, IndexError):
                pass
            else:
                title = page['title']
                if title in norm:
                    title = norm['title']
                TITLE_TO_ID[title] = doc['id']
                yield {
                    "_index": INDEX_NAME,
                    "_type": "page",
                    "_id": doc['id'],
                    "_source": doc['source'],
                }

def clean_analysis(analysis):
    """Remove non-default analysis elements"""
    analysis['analyzer'] = {
        name: dict(config, filter=[x for x in config['filter'] if 'preserve_original' not in x])
        for name, config in analysis['analyzer'].items()}
    return analysis

def reindex(es_connection, labeled, batch_size=5):
    analysis_settings = api_req({'action': 'cirrus-settings-dump'})['content']
    mapping_settings = api_req({'action': 'cirrus-mapping-dump'})['content']
    settings = {
        'mappings': {
            'page': {
                'properties': mapping_settings['page']['properties'],
            },
        },
        'settings': {
            'index': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'similarity': analysis_settings['page']['index']['similarity'],
                'analysis': clean_analysis(analysis_settings['page']['index']['analysis']),
            },
        }
    }
    es_connection.indices.delete(INDEX_NAME, ignore=[400, 404])
    es_connection.indices.create(INDEX_NAME, body=settings)
    elasticsearch.helpers.bulk(es_connection, fetch_docs(labeled, batch_size), chunk_size=batch_size)


def write_judgements(labeled, title_to_id, f):
    from itertools import count, groupby

    all_queries = set(s['query'] for s in labeled)
    qids = dict(zip(all_queries, count()))
    for query, qid in sorted(qids.items(), key=lambda x: x[1]):
        line = '# qid:{:d}: {}\n'.format(qid, query)
        f.write(line)

    labeled = sorted(labeled, key=lambda x: qids[x['query']])
    for query, scores in groupby(labeled, lambda x: x['query']):
        qid = 'qid:{:d}'.format(qids[query])
        for score in scores:
            try:
                docid = title_to_id[score['title']]
            except KeyError:
                continue
            line = '{:d}\t{} #\t{}\t{}\n'.format(
                round(float(score['score'])),
                qid, docid, score['title']
            )
            f.write(line)


if __name__ == "__main__":
    es = elastic_connection(timeout=600)
    labeled = json.load(open('discernatron.json'))['scores']
    labeled = [s for s in labeled if s['score'] is not None]

    reindex(es, labeled=labeled)
   
    with open('sample_judgments.txt', 'w') as f:
        write_judgements(labeled, TITLE_TO_ID, f)
