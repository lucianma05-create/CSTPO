"""种子草稿人工预审前端服务（仅标准库，无第三方依赖）。

用法：
    cd CSTPO && python cstpo/review_server.py --port 8765
    浏览器打开 http://127.0.0.1:8765

API：
    GET  /                            审阅单页界面
    GET  /api/tasks                   任务列表 + 审阅进度
    GET  /api/seeds?task=<task>       种子摘要列表
    GET  /api/seed/<task>/<seed_id>   单个种子全文
    POST /api/seed/<task>/<seed_id>/review   保存审阅结果
        body: {"reviewer": str, "review_notes": str, "custom": dict,
               "checklist": {条目: true/false/null}}

安全约束：仅允许写 seed["review"] 的四个字段；其余字段只读；
task 与 seed_id 白名单校验，防路径穿越。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "data" / "seeds_draft"
UI = Path(__file__).resolve().parent / "review_ui" / "index.html"
TASKS = {"esconv", "p4g", "craigslistbargain"}
ID_RE = re.compile(r"^[a-z0-9_]+$")

REVIEW_KEYS = {"reviewer", "review_notes", "custom", "checklist"}


def task_dir(task: str) -> Path:
    return SEEDS / task


def seed_path(task: str, seed_id: str) -> Path:
    p = task_dir(task) / f"{seed_id}.json"
    if not p.resolve().is_relative_to(task_dir(task).resolve()):
        raise ValueError("非法路径")
    return p


def load_seed(task: str, seed_id: str) -> dict:
    return json.loads(seed_path(task, seed_id).read_text())


def seed_list(task: str) -> list[dict]:
    rows = []
    for p in sorted(task_dir(task).glob("*.json")):
        if p.name == "manifest.json":
            continue
        s = json.loads(p.read_text())
        cl = s["review"].get("checklist", {})
        done = sum(1 for v in cl.values() if v is not None)
        rows.append({"seed_id": s["seed_id"], "low_info": s["context"].get("low_info"),
                     "cutoff_rule": s["context"]["cutoff"].get("rule", ""),
                     "emotion": s["initial_emotion"]["category"],
                     "group_id": s["provenance"]["group_id"],
                     "reviewed": s["review"].get("reviewer") != "" or done > 0,
                     "checklist_done": done, "checklist_total": len(cl)})
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "SeedReview/1.0"

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: Path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._html(UI)
        if u.path == "/api/tasks":
            out = []
            for t in sorted(TASKS):
                rows = seed_list(t)
                out.append({"task": t, "total": len(rows),
                            "reviewed": sum(1 for r in rows if r["reviewed"]),
                            "low_info": sum(1 for r in rows if r["low_info"])})
            return self._json(out)
        if u.path == "/api/seeds":
            task = parse_qs(u.query).get("task", [""])[0]
            if task not in TASKS:
                return self._json({"error": "unknown task"}, 400)
            return self._json(seed_list(task))
        m = re.match(r"^/api/seed/([a-z0-9]+)/([a-z0-9_]+)$", u.path)
        if m:
            task, sid = m.groups()
            if task not in TASKS or not ID_RE.match(sid):
                return self._json({"error": "bad path"}, 400)
            try:
                return self._json(load_seed(task, sid))
            except FileNotFoundError:
                return self._json({"error": "not found"}, 404)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        m = re.match(r"^/api/seed/([a-z0-9]+)/([a-z0-9_]+)/review$", self.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        task, sid = m.groups()
        if task not in TASKS or not ID_RE.match(sid):
            return self._json({"error": "bad path"}, 400)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict) or not set(payload).issubset(REVIEW_KEYS):
                return self._json({"error": "只允许 reviewer/review_notes/custom/checklist"}, 400)
            seed = load_seed(task, sid)
            if "checklist" in payload:
                if not isinstance(payload["checklist"], dict):
                    return self._json({"error": "checklist 需为对象"}, 400)
                for k, v in payload["checklist"].items():
                    if v not in (True, False, None):
                        return self._json({"error": f"checklist 值非法: {k}"}, 400)
            seed["review"].update(payload)
            seed["review"]["review_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            seed_path(task, sid).write_text(
                json.dumps(seed, ensure_ascii=False, indent=2) + "\n")
            return self._json({"ok": True, "saved": seed["review"]["review_date"]})
        except (json.JSONDecodeError, ValueError) as e:
            return self._json({"error": str(e)}, 400)

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"种子审阅前端已启动: http://{args.host}:{args.port}")
    print(f"种子目录: {SEEDS}  |  Ctrl-C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
