import csv, json

with open("./assertions.csv", "r", encoding="utf-8") as f_in, open("edges.jsonl", "w", encoding="utf-8") as f_out:
    reader = csv.reader(f_in, delimiter='\t')
    for row in reader:
        uri, rel, start, end, data = row
        if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
            continue
        weight = json.loads(data).get("weight", 1.0)
        if weight < 1.0:
            continue
        head = start.split("/")[3].replace("_", " ")
        tail = end.split("/")[3].replace("_", " ")
        relation = rel.split("/")[2]
        f_out.write(json.dumps({
            "source": head, "target": tail,
            "relation": relation, "weight": weight
        }) + "\n")
