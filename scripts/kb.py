# -*- coding: utf-8 -*-
"""训练经验知识库 CLI：踩坑库 / 结论台账 / 决策日志 的统一读写入口。

解决的问题：项目里的经验此前散落在对话、commit message 和某个报告的第 5 节里，
下次开训练要么想不起来，要么想起一个**已经被推翻的结论**继续当依据用。
本脚本把三类知识固化成三个 markdown 文件，并提供「写入 + 检索 + 校验 + 训练前自检」：

    experiments/LESSONS.md    踩坑库   每条：现象/根因/解法/证据
    experiments/FINDINGS.md   结论台账 每条：状态/证据/前提（被证伪的永不删除）
    experiments/DECISIONS.md  决策日志 每条：背景/备选/结论/复核触发条件

用法（全部用项目自带 Python）：

    # 1) 记一个坑
    runtime\\python.exe scripts\\kb.py add-lesson --tag 训练 ^
        --title "改 imgsz 后忘了同步 meta" ^
        --symptom "Web 端输出全废但不报错" ^
        --cause "推理 imgsz 取自 _train_meta.json，训练脚本没写" ^
        --fix "train_run.py 结束时写 meta，archive --backfill-meta 幂等纠正" ^
        --evidence "runs/detect/CT_pro/V52_ROI_1536_REG/_train_meta.json"

    # 2) 记一条结论（含被证伪的）
    runtime\\python.exe scripts\\kb.py add-finding --title "ROI 对亮斑有效" ^
        --status 成立 --confidence 高 ^
        --evidence "亮斑 AP 0.487→0.517" --premise "亮斑孤立、就地可判"

    # 3) 记一个决策
    runtime\\python.exe scripts\\kb.py add-decision --title "生产保持 V5" ^
        --status 生效 --background "..." --conclusion "..." --trigger "节拍<12ms"

    # 4) 查
    runtime\\python.exe scripts\\kb.py query 置信度
    runtime\\python.exe scripts\\kb.py list --type finding --status 已证伪

    # 5) 格式校验（写坏了会在这里报出来）
    runtime\\python.exe scripts\\kb.py check

    # 6) 训练前自检：打印相关坑 + 静态检查
    runtime\\python.exe scripts\\kb.py preflight --data datasets_v51/CT_DATASET_V51_ROI/data.yaml

设计约定：
- 编号（L-/F-/D-）自动生成，取现有最大号 +1，**不复用、不删除**
- 只追加不改写历史条目；确需更正时新增一条并在新条目里注明取代关系
- 不引入第三方依赖，纯标准库；中文路径全部按 utf-8 读写
"""
from __future__ import print_function

import argparse
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
F_LESSON = os.path.join(EXP, "LESSONS.md")
F_FINDING = os.path.join(EXP, "FINDINGS.md")
F_DECISION = os.path.join(EXP, "DECISIONS.md")

# LESSONS 的标签 == 二级小节名
TAGS = ["工程", "数据", "训练", "评测", "部署"]

# FINDINGS 的状态 -> 二级小节标题前缀
STATUS_SECTION = {
    "已证伪": "## 已证伪",
    "成立": "## 成立",
    "有条件成立": "## 有条件成立",
    "待验": "## 待验",
}
VALID_STATUS = list(STATUS_SECTION.keys())

PREFIX = {"lesson": "L", "finding": "F", "decision": "D"}


# ------------------------------------------------------------------ 基础 IO
def read(path):
    if not os.path.exists(path):
        sys.exit("[x] 文件不存在：%s" % path)
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def next_id(text, prefix):
    ids = [int(m) for m in re.findall(r"^###\s+%s-(\d+)" % prefix, text, flags=re.M)]
    return "%s-%03d" % (prefix, (max(ids) + 1) if ids else 1)


# ------------------------------------------------------------------ 解析
def iter_entries(text):
    """把 markdown 切成条目：返回 [(id, title, body)]，body 不含标题行。"""
    out = []
    cur = None
    for line in text.splitlines():
        m = re.match(r"^###\s+([LFD]-\d+)\s+(.*)$", line)
        if m:
            if cur:
                out.append(cur)
            cur = [m.group(1), m.group(2).strip(), []]
            continue
        if cur is not None:
            if re.match(r"^#{2,3}\s", line):
                out.append(cur)
                cur = None
                continue
            cur[2].append(line)
    if cur:
        out.append(cur)
    return [(i, t, "\n".join(b)) for i, t, b in out]


def insert_into_section(text, section_prefix, block):
    """把 block 插到以 section_prefix 开头的二级小节的末尾；小节不存在则新建。"""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(section_prefix):
            start = i
            break
    if start is None:
        return text.rstrip("\n") + "\n\n" + section_prefix + "\n\n" + block + "\n"

    # 找该小节内最后一个 ### 条目的结尾
    end = len(lines)
    last_entry_end = start + 1
    for j in range(start + 1, len(lines)):
        if re.match(r"^##\s", lines[j]):        # 下一个二级小节
            end = j
            break
        if re.match(r"^###\s", lines[j]):
            last_entry_end = j
    # 从 last_entry_end 往后，直到下一个 ### 或小节结束，取最后一行非空
    cut = last_entry_end
    for j in range(last_entry_end + 1, end):
        if re.match(r"^###\s", lines[j]):
            break
        if lines[j].strip():
            cut = j
    head = lines[:cut + 1]
    tail = lines[cut + 1:]
    return "\n".join(head) + "\n\n" + block + "\n" + ("\n".join(tail) if tail else "")


# ------------------------------------------------------------------ 写入
def cmd_add_lesson(a):
    text = read(F_LESSON)
    eid = next_id(text, PREFIX["lesson"])
    tag = a.tag if a.tag in TAGS else "训练"
    block = (
        "### %s %s\n"
        "- **标签**：%s\n"
        "- **现象**：%s\n"
        "- **根因**：%s\n"
        "- **解法**：%s\n"
        "- **证据**：%s\n"
        "- **首次**：%s\n"
    ) % (eid, a.title, tag, a.symptom, a.cause, a.fix, a.evidence, a.date)
    write(F_LESSON, insert_into_section(text, "## " + tag, block))
    print("[ok] 已记入 LESSONS.md：%s  %s" % (eid, a.title))


def cmd_add_finding(a):
    if a.status not in VALID_STATUS:
        sys.exit("[x] --status 只能是：%s" % " / ".join(VALID_STATUS))
    text = read(F_FINDING)
    eid = next_id(text, PREFIX["finding"])
    title = ("~~%s~~" % a.title) if a.status == "已证伪" else a.title
    parts = [
        "### %s %s" % (eid, title),
        "- **状态**：%s" % a.status,
        "- **置信度**：%s" % a.confidence,
        "- **证据**：%s" % a.evidence,
    ]
    if a.premise:
        parts.append("- **前提**：%s" % a.premise)
    if a.refuted_by:
        parts.append("- **推翻依据**：%s" % a.refuted_by)
    if a.run:
        parts.append("- **关联 run**：`%s`" % a.run)
    parts.append("- **登记**：%s" % a.date)
    block = "\n".join(parts) + "\n"
    write(F_FINDING, insert_into_section(text, STATUS_SECTION[a.status], block))
    print("[ok] 已记入 FINDINGS.md：%s  [%s] %s" % (eid, a.status, a.title))


def cmd_add_decision(a):
    text = read(F_DECISION)
    eid = next_id(text, PREFIX["decision"])
    block = (
        "### %s %s\n"
        "- **日期**：%s · **状态**：%s\n"
        "- **背景**：%s\n"
        "- **备选**：%s\n"
        "- **结论**：%s\n"
        "- **代价**：%s\n"
        "- **复核触发条件**：%s\n"
    ) % (eid, a.title, a.date, a.status, a.background,
         a.alternatives or "（未记录）", a.conclusion, a.cost or "（未记录）", a.trigger)
    write(F_DECISION, text.rstrip("\n") + "\n\n" + block + "\n")
    print("[ok] 已记入 DECISIONS.md：%s  [%s] %s" % (eid, a.status, a.title))


# ------------------------------------------------------------------ 检索
def _fmt(eid, title, body, max_lines=8):
    lines = [l for l in body.splitlines() if l.strip()][:max_lines]
    return "  %-7s %s\n%s" % (eid, title, "\n".join("          " + l for l in lines))


def cmd_query(a):
    kw = a.keyword
    hits = 0
    for path, label in ((F_LESSON, "踩坑"), (F_FINDING, "结论"), (F_DECISION, "决策")):
        text = read(path)
        for eid, title, body in iter_entries(text):
            if kw.lower() in (eid + title + body).lower():
                if hits == 0:
                    print("")
                hits += 1
                print("[%s] %s  (%s)" % (label, path.split(os.sep)[-1], eid))
                print(_fmt(eid, title, body))
                print("")
    if hits == 0:
        print("[i] 没有匹配「%s」的条目" % kw)
    else:
        print("[i] 共 %d 条" % hits)


def cmd_list(a):
    src = {"lesson": F_LESSON, "finding": F_FINDING, "decision": F_DECISION}
    paths = src.values() if a.type == "all" else [src[a.type]]
    for path in paths:
        text = read(path)
        rows = iter_entries(text)
        if a.status:
            rows = [r for r in rows if ("**状态**：%s" % a.status) in r[2]]
        print("\n== %s (%d 条) ==" % (path.split(os.sep)[-1], len(rows)))
        for eid, title, body in rows:
            state = ""
            m = re.search(r"\*\*状态\*\*：(\S+)", body)
            if m:
                state = "[%s] " % m.group(1)
            print("  %-7s %s%s" % (eid, state, title))


# ------------------------------------------------------------------ 校验
def cmd_check(a):
    errs, warns = [], []

    # LESSONS
    text = read(F_LESSON)
    rows = iter_entries(text)
    seen = set()
    for eid, title, body in rows:
        if eid in seen:
            errs.append("LESSONS %s 编号重复" % eid)
        seen.add(eid)
        for key in ("现象", "根因", "解法", "证据"):
            if ("**%s**" % key) not in body:
                errs.append("LESSONS %s 缺字段「%s」" % (eid, key))
        if re.search(r"(待补充|TODO|TBD|xxx)", body, flags=re.I):
            warns.append("LESSONS %s 含占位符" % eid)
    print("LESSONS.md  %d 条" % len(rows))

    # FINDINGS
    text = read(F_FINDING)
    rows = iter_entries(text)
    seen = set()
    n_refuted = 0
    for eid, title, body in rows:
        if eid in seen:
            errs.append("FINDINGS %s 编号重复" % eid)
        seen.add(eid)
        m = re.search(r"\*\*状态\*\*：(\S+)", body)
        if not m:
            errs.append("FINDINGS %s 缺「状态」" % eid)
            continue
        st = m.group(1)
        if st not in VALID_STATUS:
            errs.append("FINDINGS %s 状态非法：%s" % (eid, st))
        if st == "已证伪":
            n_refuted += 1
            if "**推翻依据**" not in body:
                errs.append("FINDINGS %s 已证伪但没有「推翻依据」（这是最关键的字段）" % eid)
        if st == "有条件成立" and "**前提**" not in body:
            errs.append("FINDINGS %s 有条件成立但没有「前提」" % eid)
        if "**证据**" not in body:
            errs.append("FINDINGS %s 缺「证据」" % eid)
    print("FINDINGS.md %d 条（其中已证伪 %d 条）" % (len(rows), n_refuted))

    # DECISIONS
    text = read(F_DECISION)
    rows = iter_entries(text)
    for eid, title, body in rows:
        if "**复核触发条件**" not in body:
            errs.append("DECISIONS %s 缺「复核触发条件」" % eid)
        if "**状态**" not in body:
            errs.append("DECISIONS %s 缺「状态」" % eid)
    print("DECISIONS.md %d 条" % len(rows))

    print("")
    for w in warns:
        print("[!] " + w)
    if errs:
        for e in errs:
            print("[x] " + e)
        sys.exit("\n校验失败：%d 个错误" % len(errs))
    print("[ok] 三个文件格式校验通过")


# ------------------------------------------------------------------ 训练前自检
def cmd_preflight(a):
    print("=" * 66)
    print("训练前自检（只读，不改任何文件）")
    print("=" * 66)

    sys.path.insert(0, os.path.join(ROOT, 'core'))
    from dataset_contract import inspect_dataset
    from path_safety import run_output
    fatal, inspection = 0, None
    try:
        size = int(a.imgsz) if a.imgsz else None
        if size is not None and (size <= 0 or size % 32):
            raise ValueError('imgsz必须是32对齐的正整数')
        if a.data:
            inspection = inspect_dataset(a.data, size)
            print('[1] 严格数据契约通过：%s' % a.data)
            for split, counts in inspection['stats'].items():
                print('    %s: %s' % (split, counts))
            print('    数据契约SHA256: %s' % inspection['contract_sha256'])
        else:
            print('[1] 未指定--data，仅列知识库；这不构成开训许可')
        print('[2] ROI/元信息一致性（只读契约，不猜目录名）')
        if inspection:
            contract = inspection['contract']
            if a.roi is not None and a.roi != contract['roi']:
                raise ValueError('请求roi位与数据契约不一致')
            if contract['roi']:
                raise ValueError('D-010已终止当前ROI训练路线；数据历史复核请用物化器--verify-only')
            if a.run_name:
                import json
                meta_path = run_output(a.run_name) / '_train_meta.json'
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding='utf8'))
                    if (meta.get('roi') != contract['roi'] or meta.get('roi_k') != contract['roi_k']
                            or meta.get('hyperparams', {}).get('imgsz') != size):
                        raise ValueError('已有run规格与请求/数据不一致')
        print('[3] 画布：%s' % size)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        fatal += 1
        print('[x] %s' % exc)

    # 4) 相关坑
    print("[4] 与本次训练相关的坑（完整内容：kb.py query <关键词>）")
    print("    历史条目须结合复核：L-014的旧scaleup机制已撤回，erasing不进入当前检测增强链。")
    text = read(F_LESSON)
    want = {"训练", "数据", "评测"} if not a.tag else set(a.tag)
    n = 0
    for eid, title, body in iter_entries(text):
        m = re.search(r"\*\*标签\*\*：(\S+)", body)
        if m and m.group(1) in want:
            n += 1
            print("    %-7s %s" % (eid, title))
    print("    共 %d 条" % n)

    # 5) 禁止引用的结论
    print("[5] 已证伪结论（禁止当依据）")
    text = read(F_FINDING)
    for eid, title, body in iter_entries(text):
        if "**状态**：已证伪" in body:
            print("    %-7s %s" % (eid, title.replace("~~", "")))
    print("")
    if fatal:
        sys.exit("[x] 自检发现 %d 个必须修的问题，先修再开训" % fatal)
    print("[ok] 自检通过。记得：判据先预注册（DECISIONS D-003），一次只动一个变量（L-008）")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="训练经验知识库")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("add-lesson", help="记一个坑")
    p.add_argument("--tag", default="训练", choices=TAGS)
    p.add_argument("--title", required=True)
    p.add_argument("--symptom", required=True)
    p.add_argument("--cause", required=True)
    p.add_argument("--fix", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--date", default="")
    p.set_defaults(func=cmd_add_lesson)

    p = sub.add_parser("add-finding", help="记一条结论")
    p.add_argument("--title", required=True)
    p.add_argument("--status", required=True, choices=VALID_STATUS)
    p.add_argument("--confidence", default="中", choices=["高", "中", "低"])
    p.add_argument("--evidence", required=True)
    p.add_argument("--premise", default="")
    p.add_argument("--refuted-by", default="")
    p.add_argument("--run", default="")
    p.add_argument("--date", default="")
    p.set_defaults(func=cmd_add_finding)

    p = sub.add_parser("add-decision", help="记一个决策")
    p.add_argument("--title", required=True)
    p.add_argument("--status", default="生效", choices=["生效", "已撤回", "待定"])
    p.add_argument("--background", required=True)
    p.add_argument("--alternatives", default="")
    p.add_argument("--conclusion", required=True)
    p.add_argument("--cost", default="")
    p.add_argument("--trigger", required=True)
    p.add_argument("--date", default="")
    p.set_defaults(func=cmd_add_decision)

    p = sub.add_parser("query", help="全文检索三个文件")
    p.add_argument("keyword")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("list", help="列出条目")
    p.add_argument("--type", default="all", choices=["all", "lesson", "finding", "decision"])
    p.add_argument("--status", default="")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("check", help="格式校验")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("preflight", help="训练前自检")
    p.add_argument("--data", default="")
    p.add_argument("--imgsz", default="")
    p.add_argument("--roi", dest="roi", action="store_true", default=None)
    p.add_argument("--no-roi", dest="roi", action="store_false")
    p.add_argument("--run-name", default="")
    p.add_argument("--tag", action="append", default=[])
    p.set_defaults(func=cmd_preflight)

    a = ap.parse_args()
    if not a.cmd:
        ap.print_help()
        return
    if not getattr(a, "date", None):
        a.date = __import__("datetime").date.today().isoformat()
    a.func(a)


if __name__ == "__main__":
    os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
    main()
