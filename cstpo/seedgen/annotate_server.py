"""校准对话人工标注前端服务（仅标准库，无第三方依赖；参照 review_server）。

用法：
    cd CSTPO && python cstpo/annotate_server.py --port 8766
    浏览器打开 http://127.0.0.1:8766

API：
    GET  /                                 标注单页界面
    GET  /api/tasks                        对话列表 + 标注进度（不显示 judge 评分，盲标）
    GET  /api/dialogue/<split>/<task>/<did> 对话（中文为主 + 英文原文对照）
    POST /api/dialogue/<split>/<task>/<did>/annotate   保存标注
        body: {"annotator": str, "scores": dict}

评分字段（04§19 协议）：
    esconv: E(0-4), A(0-4), evidence_sufficient(bool)
    p4g:    commitment(bool), conditional(bool), withdrawn(bool), amount
    craigslistbargain: deal(bool), final_price, parse_error(bool)

安全约束：路径白名单（split/task/did 校验）；只写 annotations 字段。
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
OUT = ROOT / "data" / "judge_calibration"
ANNO = OUT / "annotations"
UI = Path(__file__).resolve().parent / "annotate_ui" / "index.html"

TASKS = {"esconv", "p4g", "craigslistbargain"}
SPLITS = {"dev", "heldout"}
ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def anno_path(split, task, did) -> Path:
    return ANNO / split / task / f"{did}.json"


def load_dialogue(split, task, did) -> dict:
    for p in (OUT / split / task).glob(f"{did}.json"):
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        return d
    raise FileNotFoundError(did)


def load_judge() -> dict:
    """smoke_summary.json 的 judge 评分（dev 1 次、heldout 3 次取众数）。"""
    p = OUT / "smoke_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("by_dialogue", {})


def judge_view(did: str) -> dict | None:
    vs = load_judge().get(did)
    if not vs:
        return None
    # 优先用三次聚合评分（数值均值/布尔多数）
    for v in vs:
        if "aggregated" in v:
            return v["aggregated"]
    vs = [v for v in vs if "aggregated" not in v]
    if len(vs) == 1:
        return vs[0]
    # heldout 3 次：E/A 取众数元组；二元字段取众数
    from collections import Counter
    v = dict(vs[0])
    if "E" in v and "A" in v:
        (e, a), _ = Counter((x["E"], x["A"]) for x in vs).most_common(1)[0]
        v["E"], v["A"] = e, a
    for k in ("commitment", "deal"):
        if k in v:
            v[k] = Counter(bool(x.get(k)) for x in vs).most_common(1)[0][0]
    return v


def is_conflict(task: str, sc: dict, jv: dict) -> bool:
    if task == "esconv" and "E" in sc:
        return abs(sc["E"] - jv["E"]) >= 2 or abs(sc["A"] - jv["A"]) >= 2
    if task == "p4g" and "commitment" in sc:
        return bool(sc["commitment"]) != bool(jv["commitment"])
    if task == "craigslistbargain" and "deal" in sc:
        return bool(sc["deal"]) != bool(jv["deal"])
    return False


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, s, allowed):
        return s if s in allowed else None

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = UI.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/tasks":
            tasks = []
            for fp in sorted(OUT.rglob("*.json")):
                if fp.name == "smoke_summary.json" or "annotations" in fp.parts:
                    continue
                d = json.loads(fp.read_text())
                rel = fp.relative_to(OUT)
                ap = anno_path(rel.parts[0], rel.parts[1], d["dialogue_id"])
                annotated = ap.exists() and bool(
                    json.loads(ap.read_text()).get("annotations"))
                tasks.append({
                    "id": d["dialogue_id"], "task": d["task"],
                    "split": rel.parts[0], "profile": d["profile"],
                    "n_turns": d["n_turns"], "annotated": annotated,
                })
            self._json(200, tasks)
            return
        m = re.match(r"^/api/dialogue/(dev|heldout)/(\w+)/([A-Za-z0-9_]+)$", u.path)
        if m:
            split, task, did = m.group(1), m.group(2), m.group(3)
            if task not in TASKS or split not in SPLITS:
                self._json(404, {"error": "unknown"})
                return
            try:
                d = load_dialogue(split, task, did)
            except FileNotFoundError:
                self._json(404, {"error": "not found"})
                return
            ap = anno_path(split, task, did)
            zh = (json.loads(ap.read_text()) if ap.exists()
                  else {"turns_zh": [], "annotations": {}})
            self._json(200, {
                "dialogue_id": did, "task": task, "split": split,
                "profile": d["profile"], "n_turns": d["n_turns"],
                "prefix_n": d.get("prefix_n", 0),
                "situation": d["seed"].get("situation"),
                "turns": d["turns"], "turns_zh": zh.get("turns_zh", []),
                "existing": zh.get("annotations", {}),
            })
            return
        if u.path == "/api/conflicts":
            out = []
            for fp in sorted(ANNO.rglob("*.json")):
                d = json.loads(fp.read_text())
                did = d["dialogue_id"]
                jv = judge_view(did)
                if jv is None:
                    continue
                split = fp.parent.parent.name
                task = fp.parent.name
                dial = None
                for dp in (OUT / split / task).glob(f"{did}.json"):
                    dial = json.loads(dp.read_text())
                for who, a in d.get("annotations", {}).items():
                    sc = a.get("scores", {})
                    if is_conflict(task, sc, jv):
                        out.append({
                            "id": did, "split": split, "task": task,
                            "annotator": who, "human": sc, "judge": jv,
                            "feedback": a.get("feedback", ""),
                            "profile": dial["profile"] if dial else "",
                            "n_turns": dial["n_turns"] if dial else 0,
                            "arbitrated": bool(d.get("arbitrations")),
                        })
            self._json(200, out)
            return
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        u = urlparse(self.path)
        m = re.match(r"^/api/dialogue/(dev|heldout)/(\w+)/([A-Za-z0-9_]+)/annotate$",
                     u.path)
        if not m:
            self._json(404, {"error": "unknown path"})
            return
        split, task, did = m.group(1), m.group(2), m.group(3)
        if task not in TASKS or split not in SPLITS or not ID_RE.match(did):
            self._json(400, {"error": "invalid"})
            return
        if u.path.endswith("/arbitrate"):
            self.do_arbitrate(split, task, did)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        annotator = str(body.get("annotator", "anonymous"))[:50]
        scores = body.get("scores", {})
        feedback = str(body.get("feedback", ""))[:2000]
        if not isinstance(scores, dict) or not annotator.strip():
            self._json(400, {"error": "bad payload"})
            return
        ap = anno_path(split, task, did)
        ap.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(ap.read_text()) if ap.exists() else {
            "dialogue_id": did, "turns_zh": [], "annotations": {}}
        data["annotations"][annotator] = {
            "scores": scores, "feedback": feedback,
            "ts": datetime.now().isoformat(timespec="seconds")}
        ap.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        self._json(200, {"ok": True, "saved": len(data["annotations"])})

    def do_arbitrate(self, split, task, did):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        arbiter = str(body.get("arbiter", "arbiter"))[:50]
        scores = body.get("scores", {})
        feedback = str(body.get("feedback", ""))[:2000]
        if not isinstance(scores, dict) or not arbiter.strip():
            self._json(400, {"error": "bad payload"})
            return
        ap = anno_path(split, task, did)
        data = json.loads(ap.read_text()) if ap.exists() else {
            "dialogue_id": did, "turns_zh": [], "annotations": {}}
        data.setdefault("arbitrations", {})[arbiter] = {
            "scores": scores, "feedback": feedback,
            "ts": datetime.now().isoformat(timespec="seconds")}
        ap.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        self._json(200, {"ok": True, "saved": len(data["arbitrations"])})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    print(f"标注服务: http://127.0.0.1:{args.port}（Ctrl+C 退出）")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
