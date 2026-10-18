# job.json -> GitHub Actions matrix. Frames of each shot stay contiguous; split into ~equal chunks.
import json
def rng(s):
    if isinstance(s, list): return s
    out = []
    for part in str(s).split(','):
        if '-' in part: a, b = map(int, part.split('-')); out += list(range(a, b + 1))
        elif part: out.append(int(part))
    return out
j = json.load(open('job.json'))
items = []
for t in j['tasks']:
    w = float(t.get('weight', 1.0))
    for f in rng(t['frames']): items.append((t['shot'], f, w))
n = min(int(j.get('chunks', 20)), len(items))
tot = sum(w for _, _, w in items); per = tot / n
chunks, cur, acc = [], [], 0.0
for it in items:
    cur.append(it); acc += it[2]
    if acc >= per * (len(chunks) + 1) - 1e-9 and len(chunks) < n - 1:
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
inc = []
for i, c in enumerate(chunks):
    by = {}
    for s, f, _ in c: by.setdefault(s, []).append(f)
    inc.append({'id': i, 'tasks': ';'.join(f"{s}:{','.join(map(str, fs))}" for s, fs in by.items())})
print('matrix=' + json.dumps({'include': inc}))
