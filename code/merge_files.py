import json_lines
import json

def tweet_union(filenames, OUTPUT_FILE):
    data = []
    for filename in filenames:
        with open('../data/{}'.format(filename)) as f:
            for line in f:
                data.append(json.loads(line))

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out:
        for item in data:
            json.dump(item, out)
            out.write('\n')

filenames=["results_11june2018.jsonl", "results_22april2018.jsonl"]
OUTPUT_FILE = 'preGreta.jsonl'

print("Merging files...")
tweet_union(filenames, OUTPUT_FILE)
print("Done!")
