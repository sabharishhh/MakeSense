import json, re

def is_clean(term):
    return bool(re.search(r"[a-zA-Z]", term)) and not re.match(r"^[\d\s\W_]+$", term)

with open("EDGES") as f_in, open("edges_clean.jsonl", "w") as f_out:
    kept = 0
    for line in f_in:
        e = json.loads(line)
        if is_clean(e["source"]) and is_clean(e["target"]):
            f_out.write(json.dumps(e) + "\n")
            kept += 1
print("Cleaned edges written:", kept)
