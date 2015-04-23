from sklearn.datasets import fetch_20newsgroups
import predictionio
import argparse


twenty_train = fetch_20newsgroups(subset = 'train',
                                    shuffle = True,
                                    random_state = 10)



def import_events(client):
    train = ((twenty_train.target_names[twenty_train.target[k]],
              float(twenty_train.target[k]),
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
                "category": elem[0],
                "label": float(elem[1]),
                "text": elem[2]
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

