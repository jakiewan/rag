
from __future__ import annotations  # 启用未来注解语法

import argparse  # 命令行参数解析
import base64  # 图片 Base64 编码
import json  # JSON 读写
import logging  # 日志模块
import os  # 操作系统接口
import re  # 正则
import shutil  # 进程检查（which 替代）
import subprocess  # 子进程管理
import sys  # 系统相关
import time  # 计时
from dataclasses import dataclass  # 轻量数据类
from pathlib import Path  # 跨平台路径
from typing import Dict, List, Optional, Tuple  # 类型注解

# ---------------------------------------------------------------------------
# 第三方库 —— 在脚本入口再延迟 import，便于在缺包时给出友好提示
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # 读取 .env
except ImportError:  # pragma: no cover                                  # 缺包兜底
    load_dotenv = None  # type: ignore                                  # 占位


# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent  # 本文件所在目录（不 resolve，便于日志显示相对路径，跨机器一致）
PROJECT_ROOT = HERE.parent.parent  # 04-rag 根目录（个人/0914work/rag 的上两级）

IMAGE_EXTS: set = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}  # 支持的图片扩展名

DEFAULT_VL_MODEL = "MiniMax-M3"  # 默认视觉模型（MiniMax 多模态）


_EMBEDDED_MINIO_SALT: bytes = b"@}\x98\xd7\"s\xb9\xcbdQ\xcd\x7f0M\x02'"  # 16 字节随机盐
_EMBEDDED_MINIO_PASSPHARSE: bytes = b"rag-pipeline-2024-default"  # 固定 passpharse
_EMBEDDED_MINIO_TOKEN: bytes = (
    b"gAAAAABqp_XZIUHueubGmudTQzC1f8lJyzzT6AKN_WucFKeKmH91ADCxybUVm6CfdF"
    b"fDxd77y_Um9PNLDd9hdmCDGgfDXyrv8eSh56hJe9ORQi47I4t3ULM="
)  # Fernet 加密的 "access_key|secret_key"


def _decrypt_embedded_minio_creds() -> Tuple[str, str]:
    """解密代码内嵌的 MinIO 凭据，返回 (access_key, secret_key)。"""
    try:
        from cryptography.fernet import Fernet  # Fernet 对称加密
        from cryptography.hazmat.primitives import hashes  # 哈希算法
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # KDF
        import base64  # base64 编码
    except ImportError as e:  # 缺 cryptography 库
        raise RuntimeError("缺少 cryptography 库，请 pip install cryptography") from e

    kdf = PBKDF2HMAC(  # 用 PBKDF2 派生密钥
        algorithm=hashes.SHA256(),  # SHA-256
        length=32,  # 32 字节 = Fernet 需要的 256-bit
        salt=_EMBEDDED_MINIO_SALT,  # 盐
        iterations=100_000,  # 迭代次数（防暴力）
    )
    key = base64.urlsafe_b64encode(kdf.derive(_EMBEDDED_MINIO_PASSPHARSE))  # 派生密钥
    plain = Fernet(key).decrypt(_EMBEDDED_MINIO_TOKEN).decode("utf-8")  # 解密
    ak, sk = plain.split("|", 1)  # 拆分
    return ak, sk


logging.basicConfig(  # 配置日志
    level=logging.INFO,  # INFO 级别
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 日志格式
)
logger = logging.getLogger("rag_mineru_pipeline")  # 当前模块 logger


# ---------------------------------------------------------------------------
# 配置 dataclass —— 集中存放所有可调参数
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """脚本运行时的统一配置。"""

    # MinerU（默认自动查找 mineru / magic-pdf 任一存在的命令）
    mineru_bin: str = os.getenv("MINERU_BIN", "")  # 为空时 step1 自动探测
    mineru_backend: str = os.getenv("MINERU_BACKEND", "pipeline")  # 解析后端

    # MinIO：默认连远程服务器 43.143.93.152:9000，bucket=images
    # 凭据从代码内嵌的 Fernet 密文自动解密，无需 .env
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "43.143.93.152:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "")  # 为空时用内嵌
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "")  # 为空时用内嵌
    minio_bucket: str = os.getenv("MINIO_BUCKET", "images")
    minio_secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # MiniMax 多模态（OpenAI 兼容，https://api.minimaxi.com/v1）
    # 默认走 MiniMax-M3，用户只要 export MINIMAX_API_KEY=xxx 即可
    minimax_base_url: str = os.getenv(
        "MINIMAX_BASE_URL",
        "https://api.minimaxi.com/v1",
    )
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")  # 为空时回退到 DashScope

    # 阿里云百炼（OpenAI 兼容）：作为 MiniMax 不可用时的后备
    dashscope_base_url: str = os.getenv(  # 兼容接入点
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    dashscope_api_key: str = os.getenv(
        "DASHSCOPE_API_KEY", ""
    )  # API key（不提供则跳过 VLM）
    vl_model: str = os.getenv("VL_MODEL", DEFAULT_VL_MODEL)  # 视觉模型名

    # 运行时
    mode: str = "live"  # live / mock
    skip_mineru: bool = False  # 跳过 mineru 子进程
    skip_upload: bool = False  # 跳过 MinIO 上传
    skip_vlm: bool = False  # 跳过 VLM 摘要


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _which(bin_name: str) -> Optional[str]:
    """跨平台查找可执行文件（macOS/Linux 用 shutil.which，Windows 可后续扩展）。"""
    return shutil.which(bin_name)


def _rel(p: Path) -> str:
    """尽量把绝对路径转成相对 CWD 的形式，日志跨机器保持一致。

    Python 在 import 模块时会把 __file__ 内部绝对化，所以即使脚本以相对路径调用，
    Path(__file__) 仍然会得到绝对路径。这里从 CWD 向上找共同祖先，能转相对就转。
    """
    p = Path(p)
    if not p.is_absolute():
        return str(p)
    cwd = Path.cwd()
    cur, ups = cwd, 0
    while True:
        try:
            tail = p.relative_to(cur)
            prefix = Path(*([".."] * ups)) if ups else Path(".")
            return str(prefix / tail)
        except ValueError:
            if cur.parent == cur:  # 已到根仍找不到共同祖先
                return str(p)
            cur = cur.parent
            ups += 1


def _bootstrap_path() -> None:
    candidates: List[str] = []

    # 1) sys.executable 反推
    try:
        exe = Path(sys.executable).resolve()
        parts = exe.parts
        for i, p in enumerate(parts):
            if p == "envs" and i + 2 < len(parts) and parts[i + 2] == "bin":
                env_bin = Path(*parts[: i + 3])
                if env_bin.is_dir():
                    candidates.append(str(env_bin))
                break
    except OSError:
        pass

    # 2) CONDA_PREFIX/bin（兼容 activate 场景）
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin"))

    # 3) 常见 conda 根目录的所有 envs/*/bin
    for conda_root in (
        "/opt/anaconda3",
        "/usr/local/anaconda3",
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/miniforge3"),
    ):
        envs_dir = Path(conda_root) / "envs"
        if not envs_dir.is_dir():
            continue
        try:
            # sorted() 保证顺序稳定
            entries = sorted(envs_dir.iterdir(), key=lambda e: e.name)
        except OSError:
            continue
        for env in entries:
            try:
                cand = env / "bin"
                if cand.is_dir():
                    candidates.append(str(cand))
            except OSError:
                continue

    # 4) ~/.local/bin
    candidates.append(os.path.expanduser("~/.local/bin"))

    cur = os.environ.get("PATH", "")
    cur_parts = cur.split(":") if cur else []
    added: List[str] = []
    seen = set(cur_parts)
    for p in candidates:
        if p and p not in seen and Path(p).is_dir():
            added.append(p)
            seen.add(p)
    if added:
        os.environ["PATH"] = ":".join(added + cur_parts)
        logger.info("[PATH bootstrap] 已加入候选 bin 目录: %s", ", ".join(added))


def _resolve_mineru_bin(cfg: Config) -> Tuple[str, str]:
    """
    解析实际可用的 mineru 可执行文件。

    Returns:
        (real_path, flavor)  其中 flavor ∈ {"new", "old"}
            - new: 命令是 `mineru`，使用 `-b` 参数（新版 MinerU 2.x）
            - old: 命令是 `magic-pdf`，使用 `-m` 参数（magic-pdf 1.x 旧版）
    """
    if cfg.mineru_bin and _which(cfg.mineru_bin):  # 显式指定
        flavor = "new" if "mineru" in cfg.mineru_bin else "old"
        return cfg.mineru_bin, flavor
    for candidate in ("mineru", "magic-pdf"):  # 依次尝试
        found = _which(candidate)
        if found:
            flavor = "new" if candidate == "mineru" else "old"
            return found, flavor
    return "", "new"  # 默认按新版处理（找不到时给提示）


def _ensure_magic_pdf_config() -> Path:
    """
    为旧版 magic-pdf (1.x) 自动生成最小可用的 ~/magic-pdf.json。
    新版 mineru 2.x 不需要此文件。

    Returns:
        配置文件路径。
    """
    cfg_path = Path.home() / "magic-pdf.json"
    if cfg_path.exists():  # 已存在就不动
        return cfg_path
    minimal = {  # 最小可用配置
        "version": "1.0.0",
        "device-mode": "cpu",
        "table-config": {"is_table_recog_enable": False, "max_time": 400},
        "formula-config": {"is_formula_recog_enable": False},
        "config_version": "1.0.0",
    }
    cfg_path.write_text(
        json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[Step1] 已自动生成 %s", cfg_path)
    return cfg_path


def _build_mineru_cmd(
    real_bin: str, flavor: str, pdf_path: Path, output_dir: Path, backend: str
) -> List[str]:
    """
    根据 MinerU CLI 版本构造对应的命令行。
    """
    if flavor == "old":  # 旧版 magic-pdf
        # backend: pipeline/txt/ocr → method: auto/txt/ocr
        method_map = {"pipeline": "auto", "txt": "txt", "ocr": "ocr", "auto": "auto"}
        method = method_map.get(backend, "auto")
        return [real_bin, "-p", str(pdf_path), "-o", str(output_dir), "-m", method]
    # 新版 mineru
    return [real_bin, "-p", str(pdf_path), "-o", str(output_dir), "-b", backend]


def _is_mineru_available(cfg: Config) -> bool:
    """检测 mineru CLI 是否可用。"""
    found, flavor = _resolve_mineru_bin(cfg)
    if found:
        logger.info("检测到 MinerU 可执行文件: %s (flavor=%s)", found, flavor)
    else:
        logger.warning(
            "未在 PATH 中找到 mineru/magic-pdf；如需解析 PDF 请先 pip install -U magic-pdf"
        )
    return bool(found)


def _image_to_data_uri(image_path: Path) -> str:
    """把本地图片读取为 base64 data URI（用于 OpenAI 兼容的 image_url 字段）。"""
    with open(image_path, "rb") as f:  # 二进制打开
        b64 = base64.b64encode(f.read()).decode("utf-8")  # 编码
    ext = image_path.suffix.lower().lstrip(".") or "jpeg"  # 扩展名
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


# ---------------------------------------------------------------------------
# Step 1: MinerU 解析 PDF → Markdown
# ---------------------------------------------------------------------------
def step1_mineru_parse(pdf_path: Path, output_dir: Path, cfg: Config) -> Optional[Path]:
    """
    调用 mineru CLI 解析 PDF，返回生成的 sample.md 路径。

    与 04-mineru-pdf.py 完全等价，但补全了：
        - 文件存在性校验
        - 异常时打印 stderr
        - 返回的 md 路径用 glob 兜底（不同版本 mineru 输出目录可能略不同）
    """
    if cfg.skip_mineru:  # 用户显式跳过
        logger.info("[Step1] 已通过 --skip-mineru 跳过 PDF 解析。")
        return None

    if not _is_mineru_available(cfg):  # mineru 未安装
        logger.warning("[Step1] mineru 不可用，跳过 PDF 解析。")
        logger.warning("        安装命令: pip install -U magic-pdf")
        return None

    if not pdf_path.exists():  # 输入不存在
        logger.error("[Step1] PDF 不存在: %s", pdf_path)
        return None

    real_bin, flavor = _resolve_mineru_bin(cfg)  # 解析可执行文件 + 版本
    if flavor == "old":  # 旧版 magic-pdf
        _ensure_magic_pdf_config()  # 自动写 ~/magic-pdf.json
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录
    cmd = _build_mineru_cmd(
        real_bin, flavor, pdf_path, output_dir, cfg.mineru_backend  # 构造对应版本的命令
    )
    logger.info("[Step1] 执行: %s", " ".join(cmd))  # 打印命令

    start = time.time()  # 计时起点
    proc = subprocess.Popen(  # 启动子进程
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in proc.stdout:  # 实时输出
        logger.debug("[mineru] %s", line.rstrip())  # DEBUG 级别打印
    rc = proc.wait()  # 等待结束
    logger.info("[Step1] mineru 退出码=%s, 耗时=%.2fs", rc, time.time() - start)

    if rc != 0:  # 失败
        logger.error("[Step1] mineru 解析失败。")
        return None

    # 在输出目录里找 sample*.md（与 02-minerU-base.ipynb 默认文件名一致）
    candidates = sorted(output_dir.rglob("*.md"))  # 递归找所有 .md
    if not candidates:  # 兜底
        logger.error("[Step1] 未找到任何 .md 产物。")
        return None
    md_path = candidates[0]  # 取第一个
    logger.info("[Step1] MinerU 解析完成: %s", _rel(md_path))
    return md_path


# ---------------------------------------------------------------------------
# Step 2: 读取 Markdown 与同目录 images/
# ---------------------------------------------------------------------------
def step2_read_md(md_path: Path) -> Tuple[str, Path, Path]:
    """
    读取 Markdown 文本，并定位同名 images 目录。

    Returns:
        (md_content, md_path, images_dir)
    """
    md_content = md_path.read_text(encoding="utf-8")  # 读取正文
    images_dir = md_path.parent / "images"  # images 与 md 同级
    logger.info("[Step2] MD 长度=%d 字符, images_dir=%s", len(md_content), _rel(images_dir))
    return md_content, md_path, images_dir


# ---------------------------------------------------------------------------
# Step 3: 扫描 MD 中被引用的图片
# ---------------------------------------------------------------------------
def _find_image_contexts(
    md_content: str, image_filename: str, max_chars: int = 100
) -> List[Tuple[str, str, str]]:
    """
    在 Markdown 文本中查找某张图片被引用的所有上下文。

    策略：向上找最近标题，向下取 1-2 段，分别截 max_chars 字符。
    与 07-md_im.py 中 _find_image_contexts_in_md 同等行为。
    """
    lines = md_content.split("\n")  # 按行切
    img_pat = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    out: List[Tuple[str, str, str]] = []  # 结果列表

    for idx, line in enumerate(lines):  # 遍历行
        if not img_pat.search(line):  # 不含目标图
            continue

        # 向上找标题
        heading = ""
        heading_idx = -1
        for i in range(idx - 1, -1, -1):  # 向上扫
            if re.match(r"^#{1,6}\s+", lines[i]):  # 命中标题
                heading = lines[i].strip()
                heading_idx = i
                break

        # 向下找下一个标题
        next_idx = len(lines)
        for i in range(idx + 1, len(lines)):  # 向下扫
            if re.match(r"^#{1,6}\s+", lines[i]):  # 命中下一标题
                next_idx = i
                break

        pre = _extract_paragraphs(
            lines[heading_idx + 1 : idx] if heading_idx >= 0 else lines[:idx],
            max_chars,
            direction="backward",
        )  # 上文
        post = _extract_paragraphs(
            lines[idx + 1 : next_idx], max_chars, direction="forward"
        )  # 下文
        out.append((heading, pre, post))  # 收集

    return out


def _extract_paragraphs(
    lines: List[str], max_chars: int, direction: str = "forward"
) -> str:
    """
    在字符数限制内尽量取完整段落。

    direction: forward 从前往后取；backward 优先取靠近图片的段落，再还原顺序。
    """
    paragraphs: List[str] = []  # 段落列表
    cur: List[str] = []  # 当前段
    for raw in lines:  # 遍历
        s = raw.strip()
        if s == "":  # 空行：段落结束
            if cur:
                paragraphs.append("\n".join(cur))
                cur = []
        else:
            if re.match(r"^!\[.*?\]\(.*?\)$", s):  # 跳过纯图片行
                if cur:
                    paragraphs.append("\n".join(cur))
                    cur = []
                continue
            cur.append(s)
    if cur:
        paragraphs.append("\n".join(cur))
    paragraphs = [p for p in paragraphs if p.strip()]

    if not paragraphs:
        return ""

    if direction == "backward":  # 反向取
        paragraphs = list(reversed(paragraphs))

    selected: List[str] = []
    total = 0
    for p in paragraphs:
        if total + len(p) > max_chars and selected:
            break
        selected.append(p)
        total += len(p)

    if direction == "backward":  # 还原顺序
        selected = list(reversed(selected))
    return "\n\n".join(selected)


def step3_scan_images(
    md_content: str, images_dir: Path
) -> List[Tuple[Path, Tuple[str, str, str]]]:
    """
    扫描 images_dir 中的图片，过滤出在 MD 中被引用的，返回 [(Path, context), ...]。
    """
    if not images_dir.exists():  # 无 images 目录
        logger.warning("[Step3] images 目录不存在: %s", images_dir)
        return []

    result: List[Tuple[Path, Tuple[str, str, str]]] = []
    for img in sorted(images_dir.iterdir()):  # 遍历图片
        if img.suffix.lower() not in IMAGE_EXTS:  # 扩展名过滤
            continue
        ctxs = _find_image_contexts(md_content, img.name)  # 找上下文
        if not ctxs:  # 未被引用
            continue
        result.append((img, ctxs[0]))  # 取第一处
    logger.info("[Step3] 命中有效图片 %d 张。", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 4: 调用 VLM 生成图片中文标题
# ---------------------------------------------------------------------------
def step4_vlm_summaries(
    targets: List[Tuple[Path, Tuple[str, str, str]]], doc_title: str, cfg: Config
) -> Dict[str, str]:
    """
    调用视觉模型为每张图片生成中文摘要。

    优先级：
        1. MiniMax-M3（MINIMAX_API_KEY 环境变量，国内端点 api.minimaxi.com）
        2. 阿里云百炼 qwen-vl-plus（DASHSCOPE_API_KEY 环境变量）
        3. 都没有则用占位文本
    """
    if cfg.skip_vlm:  # 跳过
        logger.info("[Step4] 已通过 --skip-vlm 跳过 VLM。")
        return {p.name: "[skip-vlm]" for p, _ in targets}

    try:  # 尝试 import openai
        from openai import OpenAI  # 兼容客户端
    except ImportError:
        logger.error("[Step4] 未安装 openai，请 pip install openai>=1.0。")
        return {p.name: f"图片摘要-{p.stem[:8]}" for p, _ in targets}

    # 决定 provider：MiniMax 优先 → DashScope 兜底
    if cfg.minimax_api_key:  # 优先 MiniMax
        provider = "MiniMax"
        client = OpenAI(api_key=cfg.minimax_api_key, base_url=cfg.minimax_base_url)
        model_name = os.getenv("VL_MODEL", DEFAULT_VL_MODEL)
    elif cfg.dashscope_api_key:  # 兜底 DashScope
        provider = "DashScope"
        client = OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)
        model_name = cfg.vl_model
    else:  # 都没配
        logger.warning(
            "[Step4] MINIMAX_API_KEY / DASHSCOPE_API_KEY 均未配置，使用占位标题。"
        )
        return {p.name: f"图片摘要-{p.stem[:8]}" for p, _ in targets}

    logger.info("[Step4] 使用 provider=%s model=%s", provider, model_name)
    out: Dict[str, str] = {}
    for img_path, (heading, pre, post) in targets:  # 遍历
        try:
            data_uri = _image_to_data_uri(img_path)  # 编码
            ctx = (
                "\n".join(
                    filter(
                        None,
                        [  # 拼接上下文
                            f"所属章节标题：{heading}" if heading else "",
                            f"图片上文：{pre}" if pre else "",
                            f"图片下文：{post}" if post else "",
                        ],
                    )
                )
                or "无可用上下文"
            )

            prompt = (  # 给 VLM 的 prompt
                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                f'背景信息：\n1. 所属文档标题："{doc_title}"\n'
                f"2. 图片上下文：\n{ctx}\n"
                f"请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，"
                f'生成一个精准的中文标题（不要包含"图片"二字）。'
            )

            resp = client.chat.completions.create(  # 调模型
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=1024,  # 留足给推理 + 正文
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}},  # MiniMax-M3 关推理模式
            )
            msg = resp.choices[0].message
            raw_content = (msg.content or "").strip().replace("\n", " ")
            # MiniMax-M3 推理模型有时只输出 <think>，正文放 reasoning_content
            if not raw_content and getattr(msg, "reasoning_content", None):
                raw_content = msg.reasoning_content.strip().replace("\n", " ")
                logger.info("[Step4] %s 正文取自 reasoning_content", img_path.name)
            summary = raw_content
            # MiniMax-M3 / DeepSeek 等模型默认会带 <think>...</think> 思考块，剥掉
            summary = re.sub(
                r"<think>.*?</think>", "", summary, flags=re.DOTALL
            ).strip()
            # 截短到一句话（最多 60 字）
            if len(summary) > 60:
                summary = summary[:60].rstrip("，。、,. ") + "…"
            out[img_path.name] = summary or f"图片摘要-{img_path.stem[:8]}"
            logger.info("[Step4] %s -> %s", img_path.name, out[img_path.name])
        except Exception as e:  # 单图失败不影响整体
            logger.warning("[Step4] %s 摘要失败: %s", img_path.name, e)
            out[img_path.name] = f"图片摘要-{img_path.stem[:8]}"

    return out


# ---------------------------------------------------------------------------
# Step 5: 上传 MinIO 并替换 MD 链接
# ---------------------------------------------------------------------------
def _build_minio_client(cfg: Config):
    """构造 MinIO 客户端。导入失败或服务端连不上时返回 None。"""
    try:
        from minio import Minio  # MinIO SDK
    except ImportError:
        logger.error("[Step5] 未安装 minio，请 pip install minio。")
        return None

    # 凭据优先级：环境变量 > 代码内嵌加密凭据 > 兜底默认
    ak = cfg.minio_access_key
    sk = cfg.minio_secret_key
    if not ak or not sk or (ak == "minioadmin" and sk == "minioadmin"):
        # 默认值/未设时，尝试用代码内嵌的加密凭据
        try:
            ak, sk = _decrypt_embedded_minio_creds()
            logger.info("[Step5] 使用代码内嵌加密的 MinIO 凭据")
        except Exception as e:
            logger.warning("[Step5] 解密内嵌 MinIO 凭据失败: %s", e)

    try:
        client = Minio(  # 创建客户端
            cfg.minio_endpoint,
            access_key=ak or "minioadmin",
            secret_key=sk or "minioadmin",
            secure=cfg.minio_secure,
        )
        if not client.bucket_exists(cfg.minio_bucket):  # 桶不存在就建
            client.make_bucket(cfg.minio_bucket)
            logger.info("[Step5] 创建桶: %s", cfg.minio_bucket)
        logger.info(
            "[Step5] MinIO 客户端就绪: endpoint=%s bucket=%s",
            cfg.minio_endpoint,
            cfg.minio_bucket,
        )
        return client
    except Exception as e:  # 连接失败
        logger.warning("[Step5] MinIO 初始化失败: %s", e)
        return None


def step5_upload_and_replace(
    targets: List[Tuple[Path, Tuple[str, str, str]]],
    summaries: Dict[str, str],
    md_content: str,
    doc_stem: str,
    cfg: Config,
) -> Tuple[str, Dict[str, str]]:
    """
    上传图片到 MinIO，把 Markdown 中的本地图片链接替换为远端 URL。
    返回 (新 md_content, {文件名: 远端 url})。
    """
    if cfg.skip_upload:  # 跳过上传
        logger.info("[Step5] 已通过 --skip-upload 跳过 MinIO 上传。")
        remote = {p.name: f"http://mock-minio/{doc_stem}/{p.name}" for p, _ in targets}
        new_md = md_content
        for fn, s in summaries.items():
            new_md = re.sub(
                r"!\[(.*?)\]\((.*?" + re.escape(fn) + r".*?)\)",
                f"![{s}]({remote[fn]})",
                new_md,
            )
        return new_md, remote

    client = _build_minio_client(cfg)  # 客户端
    proto = "https://" if cfg.minio_secure else "http://"  # 协议
    base_url = f"{proto}{cfg.minio_endpoint}/{cfg.minio_bucket}"  # 桶基础 URL
    remote: Dict[str, str] = {}

    for img_path, _ in targets:  # 遍历图片
        object_name = f"{doc_stem}/{img_path.name}"  # 桶内路径
        ext = img_path.suffix.lower()  # 扩展名
        content_type = f"image/{ext.lstrip('.')}" if ext else "application/octet-stream"
        if ext == ".jpg":
            content_type = "image/jpeg"
        if client is not None:  # 有客户端
            try:
                client.fput_object(  # 上传
                    cfg.minio_bucket,
                    object_name,
                    str(img_path),
                    content_type=content_type,
                )
                remote[img_path.name] = f"{base_url}/{object_name}"  # 记录 URL
                logger.info("[Step5] 上传成功: %s", img_path.name)
            except Exception as e:
                logger.warning("[Step5] 上传失败 %s: %s", img_path.name, e)
                remote[img_path.name] = f"{base_url}/{object_name}"  # 占位 URL
        else:  # 无客户端 mock
            remote[img_path.name] = f"{base_url}/{object_name}"
            logger.info("[Step5] mock URL: %s", remote[img_path.name])

    new_md = md_content
    for fn, s in summaries.items():  # 替换链接
        if fn not in remote:
            continue
        new_md = re.sub(
            r"!\[(.*?)\]\((.*?" + re.escape(fn) + r".*?)\)",
            f"![{s}]({remote[fn]})",
            new_md,
            flags=re.IGNORECASE,
        )
    logger.info("[Step5] 链接替换完成。")
    return new_md, remote


# ---------------------------------------------------------------------------
# Step 6: 写新 MD 文件
# ---------------------------------------------------------------------------
def step6_write_md(md_path: Path, new_content: str) -> Path:
    """
    把替换后的内容写到 sample_new.md（同目录）。
    """
    out = md_path.with_name(f"{md_path.stem}_new{md_path.suffix}")
    out.write_text(new_content, encoding="utf-8")
    logger.info("[Step6] 已写入: %s", out)
    return out


def step7_load_as_documents(md_path: Path, images_dir: Path) -> List[Dict[str, Any]]:
    """
    按 Markdown 标题（# ... ######）切分文档，产出 Document 列表。
    对应 01-document_loader.ipynb 的 UnstructuredMarkdownLoader(mode="elements")：
    每块结构为 {"page_content": str, "metadata": {source, heading, heading_level}}。
    """
    logger.info(
        "[Step7] 按 Markdown 标题切分文档（对应 01 UnstructuredMarkdownLoader）"
    )
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    blocks: List[Dict[str, Any]] = []
    cur_heading = ""
    cur_level = 0
    cur_lines: List[str] = []

    def _flush() -> None:
        content = "\n".join(cur_lines).strip()
        if content or cur_heading:
            blocks.append(
                {
                    "page_content": content,
                    "metadata": {
                        "source": str(md_path),
                        "heading": cur_heading,
                        "heading_level": cur_level,
                        "images_dir": str(images_dir) if images_dir else "",
                    },
                }
            )

    for line in lines:
        m = heading_re.match(line)
        if m:
            _flush()
            cur_level = len(m.group(1))
            cur_heading = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    _flush()

    logger.info("[Step7] 共切出 %d 个 Document", len(blocks))
    return blocks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> int:
    """执行整条流水线。返回 0 表示成功，非 0 表示失败。"""
    if load_dotenv is not None:  # 加载 .env
        env_path = HERE / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("已加载 .env: %s", env_path)

    cfg = Config(  # 构造配置
        mode=args.mode,
        skip_mineru=args.skip_mineru,
        skip_upload=args.skip_upload,
        skip_vlm=args.skip_vlm,
    )
    logger.info("运行模式: %s", cfg.mode)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (HERE / "result")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[输出目录] %s", _rel(out_dir))

    # 输入：MD 优先于 PDF
    md_path: Optional[Path] = None
    if args.md:
        md_path = Path(args.md).resolve()
    elif args.pdf:
        pdf_path = Path(args.pdf).resolve()
        # 仅在 live 模式才尝试调用 mineru
        if cfg.mode == "live" and not cfg.skip_mineru:
            md_path = step1_mineru_parse(pdf_path, out_dir, cfg)
        else:
            logger.info(
                "[入口] mock 模式或 --skip-mineru，跳过 PDF 解析，请用 --md 指定。"
            )
            return 1
    else:
        # 默认使用 rag 目录自带的 mineru 产物（自包含，不依赖父级）
        # 默认值指向 rag/ 根目录（与 README 第 122-129 行、docstring 第 32-33 行一致；
        # HERE = 01-load/，HERE.parent = rag/）
        default_md = HERE.parent / "result" / "sample" / "auto" / "sample.md"
        if default_md.exists():
            md_path = default_md
            logger.info("[入口] 使用默认 MD: %s", _rel(md_path))
        else:
            # 默认 MD 不存在 → 自动回退到仓库自带的 sample.pdf 跑 mineru
            default_pdf = HERE.parent / "knowledge_base" / "sample.pdf"
            if default_pdf.exists() and cfg.mode == "live" and not cfg.skip_mineru:
                logger.info(
                    "[入口] 默认 MD 不存在，自动使用仓库自带 PDF 跑 mineru: %s",
                    _rel(default_pdf),
                )
                pdf_path = default_pdf
                md_path = step1_mineru_parse(pdf_path, out_dir, cfg)
            else:
                logger.error(
                    "[入口] 未指定 --md/--pdf，且默认 MD 与默认 PDF 均不存在: %s / %s",
                    _rel(default_md),
                    _rel(default_pdf),
                )
                return 1

    if not md_path or not md_path.exists():
        logger.error("Markdown 文件不存在，终止。")
        return 1

    # Step 2-3
    md_content, _, images_dir = step2_read_md(md_path)
    targets = step3_scan_images(md_content, images_dir)
    if not targets:
        logger.warning("未在 MD 中找到有效图片，结束。")
        return 0

    # Step 4
    summaries = step4_vlm_summaries(targets, doc_title=md_path.stem, cfg=cfg)

    # Step 5
    new_md, remote = step5_upload_and_replace(
        targets,
        summaries,
        md_content,
        md_path.stem,
        cfg,
    )

    # Step 6 —— 把新 MD / 摘要 JSON 都写到 out_dir，而不是原 MD 同目录
    out_md = out_dir / f"{md_path.stem}_new{md_path.suffix}"
    out_md.write_text(new_md, encoding="utf-8")
    logger.info("[Step6] 已写入: %s", _rel(out_md))

    summary_path = out_dir / f"{md_path.stem}_summary.json"
    summary_path.write_text(
        json.dumps(
            {"summaries": summaries, "remote_urls": remote},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("[摘要] %s", _rel(summary_path))

    # Step 7 —— 文档加载（对应 01 UnstructuredMarkdownLoader，按标题切 Document）
    documents = step7_load_as_documents(out_md, images_dir)
    documents_path = out_dir / f"{md_path.stem}_documents.json"
    documents_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[Step7] Document 列表: %s", _rel(documents_path))

    logger.info("===== 运行结束 =====")
    return 0


def parse_args() -> argparse.Namespace:
    """命令行参数解析。"""
    p = argparse.ArgumentParser(
        description="RAG 流水线：MinerU 解析 PDF + VLM 图片摘要 + MinIO 上传。",
    )
    p.add_argument(
        "--mode",
        choices=["live", "mock"],
        default="live",
        help="运行模式：live 走真实服务，mock 仅做占位验证。",
    )
    p.add_argument("--pdf", type=str, default=None, help="输入 PDF 路径。")
    p.add_argument("--md", type=str, default=None, help="已解析好的 Markdown 路径。")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="产物输出目录，默认 0914work/rag/result/。",
    )
    p.add_argument("--skip-mineru", action="store_true", help="跳过 MinerU 解析。")
    p.add_argument("--skip-vlm", action="store_true", help="跳过视觉模型摘要。")
    p.add_argument("--skip-upload", action="store_true", help="跳过 MinIO 上传。")
    return p.parse_args()


if __name__ == "__main__":
    _bootstrap_path()  # 先把 conda/venv bin 加进 PATH
    sys.exit(run_pipeline(parse_args()))
