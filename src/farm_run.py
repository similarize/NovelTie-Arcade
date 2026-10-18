# One matrix runner: for each shot in its slice, mesh (background producer) + render (Blender consumer).
import sys, os, json, subprocess, time
tasks, blender, assets, out = sys.argv[1:5]
j = json.load(open('job.json'))
os.makedirs(out, exist_ok=True)
os.environ['FROG_CACHE'] = os.path.expanduser('~/frogcache')
for t in tasks.split(';'):
    shot, frames = t.split(':')
    t0 = time.time()
    md = f'/tmp/mesh/{shot}'; os.makedirs(md, exist_ok=True)
    meshp = subprocess.Popen(['python3', 'src/mesh_frames.py', '--shot', f'shots/{shot}.json', '--list', frames, '--out', md, '--throttle', '6'])
    cmd = [blender, '-b', '-t', '0', '-P', 'src/render_shot.py', '--', '--shot', f'shots/{shot}.json', '--list', frames,
           '--mesh', md, '--out', out, '--assets', assets, '--wait', '1']
    if j.get('pct'): cmd += ['--pct', str(j['pct'])]
    if j.get('samples'): cmd += ['--samples', str(j['samples'])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if any(k in line for k in ('RENDERED', 'Error', 'Traceback', 'rror:', 'ALLDONE', 'SETUP')): print(line, flush=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:])
    if meshp.poll() is None: meshp.kill()
    meshp.wait()
    for f in os.listdir(md): os.remove(os.path.join(md, f))
    print('SHOT', shot, 'frames', len(frames.split(',')), 'sec', round(time.time() - t0), flush=True)
