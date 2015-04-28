import predictionio
import argparse


def import_words(client):
    words = open('./data/common-english-words.txt', 'r').read().split(",")
    count = 0
    for word in words:
        count += 1
        client.create_event(
            event = "stopwords",
            entity_id = "word" + str(count),
            entity_type = "stopword",
            properties = {
                "word": word
            })
    print("Imported {0} stopwords.".format(count))



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

    import_words(client)
