import predictionio
import argparse
import os

categories = os.popen('ls ./data/20_newsgroups').read().split('\n')[: -1]


def import_events(client):
    count = 0
    print('Importing data.....')

    for k in range(len(categories)):
        cat_files = os.popen(
            'ls ./data/20_newsgroups/' + categories[k] + '/*'
        ).read().split('\n')[: -1]
        for file_ in cat_files:
            try:
                client.create_event(
                    event = "documents",
                    entity_id = count,
                    entity_type = "source",
                    properties = {
                        "label": k,
                        "text": open(file_).read(),
                        "category" : categories[k]
                })
                count += 1
            except UnicodeDecodeError:
                pass
            except predictionio.NotCreatedError:
                pass
    print("Imported {0} events.".format(count))


stop_words = open('./data/stopwords.txt').read().split('\n')

def import_stopwords(client):
    count = 0
    print("Importing stop words.....")
    for elem in stop_words:
        count += 1
        client.create_event(
            event = "stopwords",
            entity_id = count,
            entity_type = "resource",
            properties = {
                "word" : elem
            })
    print("Imported {0} stop words.".format(count))
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import sample data for text manipulation engine")
    parser.add_argument('--access_key', default='invald_access_key')
    parser.add_argument('--url', default="http://localhost:7070")
    args = parser.parse_args()

    client = predictionio.EventClient(
        access_key=args.access_key,
        url=args.url,
        threads=20,
        qsize=5000)

    import_events(client)
    import_stopwords(client)

