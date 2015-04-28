from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text 
import predictionio
import argparse







categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset = 'train',
                                  shuffle=True,
                                  random_state=10,
                                  categories = categories)
stop_words = text.ENGLISH_STOP_WORDS


def import_stopwords(client):
    count = 0
    for elem in stop_words:
        count += 1
        client.create_event(
            event = "stopwords",
            entity_id = count,
            entity_type = "rsc",
            properties = {
                "word" : elem
            })
    print("Imported {0} stopwords.".format(count))
        

def import_events(client):
    train = ((float(twenty_train.target[k]),
              twenty_train.data[k])
             for k in range(len(twenty_train.data)))
    count = 0
    print('Importing data.....')
    for elem in train:
        count += 1
        client.create_event(
            event = "documents",
            entity_id = count,
            entity_type = "source",
            properties = {
                "label": elem[0],
                "text": elem[1]
            })
    print("Imported {0} events.".format(count))

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

