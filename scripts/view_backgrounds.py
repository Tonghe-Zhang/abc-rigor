"""Browser viewer: every abc_sim task, each in its own background variation.

    python scripts/view_backgrounds.py --background-variation-asset-dir <package dir or s3:// URI> \\
        [--port 8765] [--max-tasks 40] [--background-seed 0] [--tasks spell_cat sort_lego_blocks ...]

Open http://localhost:<port>/ (forward the port from a remote host). Rendering uses the CPU MuJoCo
renderer (set MUJOCO_GL=egl or osmesa on a headless machine). The grid shows every task's top camera;
click a tile for a large view with the top / wrist cameras or a free orbit camera, draw the next
background, pin a background id, or re-sample the task. Task i uses background seed base + i, so
every task has its own background stream (task seeds stay independent).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np

import abc_sim


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Tasks:
    """All envs and every MuJoCo call live on one worker thread (one GL owner)."""

    def __init__(self, args):
        self.args = args
        names = args.tasks or [s.name for s in abc_sim.list_task_specs()]
        self.names = names[: args.max_tasks]
        self.envs: dict[str, object] = {}
        self.errors: dict[str, str] = {}
        self.seeds = {n: 0 for n in self.names}
        self.thumbs: dict[str, bytes] = {}
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def call(self, fn, *a, timeout=600):
        box = {"ev": threading.Event()}
        self.q.put((fn, a, box))
        box["ev"].wait(timeout)
        if "err" in box:
            raise RuntimeError(box["err"])
        return box.get("res")

    def _loop(self):
        for i, name in enumerate(self.names):  # build in the background; the page shows progress
            self.q.put((self._build, (i, name), {"ev": threading.Event()}))
        while True:
            fn, a, box = self.q.get()
            try:
                box["res"] = fn(*a)
            except Exception:  # noqa: BLE001 - shown in the page
                box["err"] = traceback.format_exc(limit=3)
            box["ev"].set()

    def _build(self, i, name):
        try:
            env = abc_sim.make_env(
                task=name,
                render_cameras=False,  # the viewer owns its renderers
                randomize_background=True,
                background_seed=self.args.background_seed + i,
                background_variation_asset_dir=self.args.background_variation_asset_dir,
            )
            env.reset(seed=0)
            self.envs[name] = env
            self.thumbs[name] = self._render(name, "top", 360, 240)
            print(f"[{i + 1}/{len(self.names)}] {name}: {env.background_info}", flush=True)
        except Exception as e:  # noqa: BLE001
            self.errors[name] = f"{type(e).__name__}: {e}"
            print(f"[{i + 1}/{len(self.names)}] {name}: FAILED {self.errors[name]}", flush=True)

    def _render(self, name, cam, w, h, free=None):
        env = self.envs[name]
        r = env._viewer_renderers.get((w, h)) if hasattr(env, "_viewer_renderers") else None
        if r is None:
            env._viewer_renderers = getattr(env, "_viewer_renderers", {})
            r = env._viewer_renderers[(w, h)] = mujoco.Renderer(env.model, h, w)
        if free is not None:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.azimuth, c.elevation, c.distance = free["az"], free["el"], free["dist"]
            c.lookat[:] = [free["lx"], free["ly"], free["lz"]]
            r.update_scene(env.data, camera=c)
        else:
            r.update_scene(env.data, camera=cam)
        return _jpeg(r.render())

    def _rebind(self, name):
        env = self.envs[name]
        for r in getattr(env, "_viewer_renderers", {}).values():
            r.close()
        env._viewer_renderers = {}
        self.thumbs[name] = self._render(name, "top", 360, 240)

    def action(self, name, what, value=None):
        env = self.envs[name]
        if what == "next_background":
            env.set_background()
        elif what == "background_id":
            env.set_background(int(value))
        elif what == "resample_task":
            self.seeds[name] += 1
            bid = env.background_info["background_id"]
            env.reset(seed=self.seeds[name], options={"background_id": bid})
        self._rebind(name)  # the model may have been reloaded: fresh renderers upload its textures
        return self.info(name)

    def info(self, name):
        env = self.envs.get(name)
        if env is None:
            return {"name": name, "error": self.errors.get(name), "ready": False}
        return {"name": name, "ready": True, "task_seed": self.seeds[name], **(env.background_info or {})}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>abc_sim backgrounds</title><style>
body{margin:0;background:#111;color:#ddd;font:13px sans-serif}#top{padding:8px}#grid{display:flex;flex-wrap:wrap;gap:6px;padding:6px}
.t{width:360px;cursor:pointer;background:#1c1c1c;border-radius:4px;overflow:hidden}.t img{width:360px;height:240px;display:block;background:#222}
.t div{padding:4px 6px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.err{color:#f88}
#big{display:none;padding:8px}#v{display:block;max-width:100%;cursor:grab;user-select:none}button,select,input{background:#222;color:#ddd;border:1px solid #444;padding:3px 6px}
</style></head><body><div id="top"><b>abc_sim tasks with background variations</b> <span id="prog"></span>
<span style="opacity:.6"> - click a task; free camera: drag to orbit, shift/right-drag to pan, wheel to zoom</span></div>
<div id="grid"></div>
<div id="big"><button onclick="back()">back to grid</button> <b id="bname"></b>
 view <select id="cam"><option>top</option><option>left</option><option>right</option><option value="free">free camera</option></select>
 <button onclick="act('next_background')">next background</button>
 id <input id="bid" type="number" min="0" style="width:70px"><button onclick="act('background_id',document.getElementById('bid').value)">set</button>
 <button onclick="act('resample_task')">re-sample task (same background)</button> <span id="binfo"></span><br><br>
 <img id="v" draggable="false"></div>
<script>
let cur=null,S={az:0,el:-35,dist:1.9,lx:0.6,ly:0,lz:0.75},busy=false,dirty=false,N=0;
const g=document.getElementById('grid');
function fmt(i){return i.ready?`#${i.background_id} ${i.wallpaper.split('/')[0]} + ${i.table.split('/')[0]}`:(i.error?'<span class=err>'+i.error+'</span>':'building...')}
async function refresh(){const L=await (await fetch('/tasks')).json();N=L.length;let ready=0;
 for(const i of L){if(i.ready)ready++;let el=document.getElementById('t_'+i.name);
  if(!el){el=document.createElement('div');el.className='t';el.id='t_'+i.name;el.innerHTML='<img><div class=n></div><div class=b></div>';el.onclick=()=>open_(i.name);g.appendChild(el)}
  el.querySelector('.n').textContent=i.name;el.querySelector('.b').innerHTML=fmt(i);
  if(i.ready&&!el.dataset.v){el.dataset.v=1;el.querySelector('img').src='/thumb?task='+i.name+'&t='+Date.now()}}
 document.getElementById('prog').textContent=` ${ready}/${N} tasks ready`;if(ready<N)setTimeout(refresh,2000)}
function open_(n){cur=n;g.style.display='none';document.getElementById('big').style.display='block';document.getElementById('bname').textContent=n;info();upd()}
function back(){cur=null;document.getElementById('big').style.display='none';g.style.display='flex'}
async function info(){const i=await (await fetch('/info?task='+cur)).json();document.getElementById('binfo').innerHTML=fmt(i)+(i.ready?` (task seed ${i.task_seed})`:'')}
function q(){const c=document.getElementById('cam').value;return new URLSearchParams({task:cur,cam:c,...S,t:Date.now()})}
function upd(){dirty=true;pump()}
function pump(){if(busy||!dirty||!cur)return;busy=true;dirty=false;const im=new Image();im.onload=()=>{document.getElementById('v').src=im.src;busy=false;pump()};
 im.onerror=()=>{busy=false};im.src='/frame?'+q()}
async function act(w,v){await fetch('/act?'+new URLSearchParams({task:cur,what:w,value:v||''}));info();upd();
 const el=document.getElementById('t_'+cur);el.querySelector('img').src='/thumb?task='+cur+'&t='+Date.now();el.querySelector('.b').innerHTML=document.getElementById('binfo').innerHTML}
document.getElementById('cam').onchange=upd;const v=document.getElementById('v');let drag=null;
v.onmousedown=e=>{document.getElementById('cam').value='free';drag={x:e.clientX,y:e.clientY,p:e.shiftKey||e.button==2}};window.onmouseup=()=>drag=null;v.oncontextmenu=e=>e.preventDefault();
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.p){const a=S.az*Math.PI/180,k=S.dist*0.0015;S.lx-=(-Math.sin(a)*dx+Math.cos(a)*dy)*k;S.ly-=(Math.cos(a)*dx+Math.sin(a)*dy)*k}else{S.az-=dx*0.3;S.el=Math.max(-89,Math.min(-2,S.el-dy*0.3))}upd()};
v.onwheel=e=>{e.preventDefault();document.getElementById('cam').value='free';S.dist=Math.max(0.3,Math.min(8,S.dist*Math.exp(e.deltaY*0.001)));upd()};
refresh();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background-variation-asset-dir", "--background_variation_asset_dir", required=True)
    ap.add_argument("--background-seed", "--background_seed", type=int, default=0, help="task i uses seed + i")
    ap.add_argument("--tasks", nargs="*", help="task names (default: the whole catalogue)")
    ap.add_argument("--max-tasks", "--max_tasks", type=int, default=40, help="cap on simultaneously loaded tasks (CPU / memory)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    T = Tasks(args)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            p = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, PAGE.encode(), "text/html")
                if u.path == "/tasks":
                    return self.send(200, json.dumps([T.info(n) for n in T.names]).encode(), "application/json")
                name = p.get("task", "")
                if name not in T.envs:
                    return self.send(404, b"task not ready", "text/plain")
                if u.path == "/info":
                    return self.send(200, json.dumps(T.info(name)).encode(), "application/json")
                if u.path == "/thumb":
                    return self.send(200, T.thumbs[name], "image/jpeg")
                if u.path == "/frame":
                    cam = p.get("cam", "top")
                    free = None
                    if cam == "free":
                        free = {k: float(p.get(k, d)) for k, d in (("az", 0), ("el", -35), ("dist", 1.9), ("lx", 0.6), ("ly", 0), ("lz", 0.75))}
                    img = T.call(T._render, name, cam, args.width * 3 // 2, args.height * 3 // 2, free)
                    return self.send(200, img, "image/jpeg")
                if u.path == "/act":
                    res = T.call(T.action, name, p.get("what"), p.get("value") or None)
                    return self.send(200, json.dumps(res).encode(), "application/json")
            except Exception as e:  # noqa: BLE001
                return self.send(500, str(e).encode(), "text/plain")
            self.send(404, b"not found", "text/plain")

    print(f"serving http://{args.host}:{args.port}/  ({len(T.names)} tasks)", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
