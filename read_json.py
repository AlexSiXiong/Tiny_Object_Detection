import json

with open('./val2017.json', 'r') as f:
    dataset = json.load(f)
    if isinstance(dataset, dict):

        for i in dataset.keys():
            print('----------')
            print(i)

            if len(dataset[i]) < 10:
                print(dataset[i])
            else:
                print(dataset[i][0])
                print(dataset[i][1])
                continue
