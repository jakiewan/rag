"""
md_splitter.py — 按 day02「7 步法」切 Markdown。

Step1 输入 → Step2 标题切分（hierarchy 追踪 + 代码围栏保护 + preamble 并入首节）
       → Step3 无标题兜底 → Step4 表格原子化 + 长度控制 + 短块合并
       → Step5 组装 → Step6 备份

输出 dict 列表（不是 LangChain Document），便于 bge_m3_embedding_utils.py 读取。
"""
import json
import os
import re
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_LEN = 2000
MIN_LEN = 500

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+)")
_TABLE_RE = re.compile(r"(<table>.*?</table>)", re.DOTALL | re.IGNORECASE)


def _split_by_headings(content: str, file_title: str) -> Tuple[List[dict], bool]:
    """按标题切分，hierarchy[1..6] 追踪 parent_title，代码围栏内 # 跳过，
    第一个标题之前的 preamble（题图/引言）并入首节。"""
    sections, body, preamble = [], [], []
    cur_title, cur_level = "", 0
    hierarchy, in_fence, seen_first = [""] * 7, False, False

    def flush():
        b = "\n".join(body).strip()
        if not (cur_title or b):
            return
        parent = next(
            (hierarchy[l] for l in range(cur_level - 1, 0, -1) if hierarchy[l]),
            cur_title or file_title,
        )
        sections.append({
            "title": cur_title, "body": b,
            "file_title": file_title, "parent_title": parent,
            "level": cur_level,
        })

    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
        m = _HEADING_RE.match(line) if not in_fence else None
        if m:
            is_first = not seen_first
            seen_first = True
            if not is_first:
                flush()
            cur_level = len(m.group(1))
            cur_title = line.strip()
            hierarchy[cur_level] = cur_title
            for i in range(cur_level + 1, 7):
                hierarchy[i] = ""
            body = list(preamble) if is_first else []
        else:
            (preamble if not seen_first else body).append(line)
    flush()
    return sections, bool(sections)


def _atomize(body: str) -> List[Tuple[str, bool]]:
    """按表格边界拆 body，返回 [(text, is_table)]。"""
    parts = _TABLE_RE.split(body)
    return [(p, "<table" in p.lower() and "</table>" in p.lower())
            for p in parts if p.strip()]


def _split_section(section: dict) -> List[dict]:
    """拆 section：先按表格边界拆，表格独立成 has_table=True 块；
    超长文本用 RecursiveCharacterTextSplitter 切；不超长整块保留。"""
    title = section["title"]
    body = section["body"]
    full = f"{title}\n\n{body}" if title else body

    if len(full) <= MAX_LEN:
        return [{**section, "has_table": "<table" in body.lower()}]

    chunks, part = [], 0
    for text, is_table in _atomize(body):
        if is_table:
            part += 1
            chunks.append({
                "title": f"{title}-table" if title else "table",
                "body": text.strip(),
                "file_title": section["file_title"],
                "parent_title": section["parent_title"],
                "part": part, "has_table": True,
            })
            continue
        available = max(MAX_LEN - len(title) - 2, 1) if title else MAX_LEN
        if len(text) <= available:
            part += 1
            chunks.append({
                "title": title or f"chunk-{part}",
                "body": text.strip(),
                "file_title": section["file_title"],
                "parent_title": section["parent_title"],
                "part": part, "has_table": False,
            })
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available, chunk_overlap=0,
                separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            )
            for piece in splitter.split_text(text):
                part += 1
                chunks.append({
                    "title": f"{title}-{part}" if title else f"chunk-{part}",
                    "body": piece.strip(),
                    "file_title": section["file_title"],
                    "parent_title": section["parent_title"],
                    "part": part, "has_table": False,
                })
    return chunks


def _merge_short(chunks: List[dict]) -> List[dict]:
    """合并同 parent 下 < MIN_LEN 的相邻短块；表格不参与；
    被合并的章节标题进 body 作为内部分隔符（不丢标题）；
    合并后总长 > MAX_LEN*1.5 不合并。"""
    if not chunks:
        return []

    # 表格前的短文本并入表格（前文标题写入 body）
    buf, i = [], 0
    while i < len(chunks):
        c = chunks[i]
        if (i + 1 < len(chunks)
            and chunks[i + 1]["has_table"]
            and not c["has_table"]
            and len(c["body"]) < MIN_LEN):
            t = chunks[i + 1]
            t["body"] = (c["title"] + "\n\n" + c["body"] + "\n\n" + t["body"]).strip()
            buf.append(t)
            i += 2
        else:
            buf.append(c)
            i += 1

    # 同 parent 短块合并，被合并的标题写进 body
    merged = [buf[0]]
    for c in buf[1:]:
        if (not merged[-1]["has_table"]
            and not c["has_table"]
            and merged[-1]["parent_title"] == c["parent_title"]
            and len(merged[-1]["body"]) < MIN_LEN):
            new_body = (merged[-1]["body"] + "\n\n" + c["title"] + "\n\n" + c["body"]).strip()
            if len(new_body) + len(merged[-1]["title"]) <= MAX_LEN * 1.5:
                merged[-1]["body"] = new_body
                continue
        merged.append(c)
    return merged


def _assemble(sections: List[dict]) -> List[dict]:
    """title + body → content。"""
    out = []
    for s in sections:
        title, b = s["title"], s["body"]
        out.append({
            "title": title,
            "content": (f"{title}\n\n{b}").strip() if title else b,
            "file_title": s["file_title"],
            "parent_title": s["parent_title"],
            "part": s.get("part", 0),
            "has_table": s.get("has_table", False),
        })
    return out


def process_markdown(state: dict) -> dict:
    md = state["md_content"].replace("\r\n", "\n").replace("\r", "\n")
    title = state["file_title"]
    sections, has_title = _split_by_headings(md, title)
    if not has_title:
        sections = [{"title": "无标题", "body": md, "file_title": title,
                    "parent_title": title, "level": 1}]

    split = []
    for s in sections:
        split.extend(_split_section(s))
    state["chunks"] = _assemble(_merge_short(split))

    out_dir = state.get("file_dir", "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(state["chunks"], f, ensure_ascii=False, indent=2)
    return state


if __name__ == "__main__":
    sample_md = os.path.join(os.path.dirname(__file__), "..", "01-load", "result", "sample_new.md")
    state = {
        "file_title": "sample_new",
        "md_content": open(sample_md, encoding="utf-8").read().strip(),
        "file_dir": os.path.dirname(sample_md),
    }
    state = process_markdown(state)
    sizes = [len(c["content"]) for c in state["chunks"]]
    print(f"chunks={len(state['chunks'])} "
          f"len[min/avg/max]={min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)} "
          f"has_table={sum(1 for c in state['chunks'] if c['has_table'])}")