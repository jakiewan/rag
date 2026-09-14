# RAG 流水线 —— MinerU + VLM + MinIO

> 整合来源：同级仓库 01-07 既有脚本（document_loader / minerU-base / mineru-pdf / image-handler / upload_image / md_im）

本目录封装一条端到端 RAG 预处理流水线，**单个 Python 脚本**即可走通：

```
PDF ─MinerU→ Markdown ─扫描→ 图片清单 ─VLM→ 中文摘要 ─MinIO→ 远端URL ─替换→ 新Markdown
```

---

## 一、RAG 关键技术总结

RAG（Retrieval-Augmented Generation，检索增强生成）= **"外挂知识库"** 的大模型问答范式。把"非参数记忆"（向量库 + 检索）接在"参数记忆"（LLM）之前，让模型在不重训的前提下回答私域 / 实时 / 长尾问题。

### 1. 经典三段式架构

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ 文档加载   │ →  │ 文档分割   │ →  │ 检索（向量 │ →  │ LLM 生成   │
│ Loader     │    │ Splitter   │    │  + 关键词）│    │ Generator  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
      │                  │                  │                │
      ▼                  ▼                  ▼                ▼
  多格式原文       语义级 chunks       相似度 / 混合       引用 + 回答
```

### 2. 关键模块拆解

| 模块 | 作用 | 本流水线对应 | 主流选型 |
|------|------|-------------|----------|
| **文档加载** | 多格式解析：txt / md / pdf / docx / pptx / html / 表格 | `01-document_loader.ipynb` | LangChain Loaders、Unstructured、MinerU |
| **文档清洗** | 修复 Unicode 引号 / 去除项目符号 / 合并空白 | `01` 中 `clean_text` | `unstructured.cleaners` |
| **文档分割** | 按 chunk_size + overlap 切分，保留语义边界 | 后续章节 | RecursiveCharacterTextSplitter、Markdown splitter |
| **Embedding** | 把文本 / 图片 / 表格压成稠密向量 | 后续章节 | BGE、text-embedding-3、Qwen3-Embedding |
| **向量检索** | Top-K 相似度召回（Cosine / IP / L2） | 后续章节 | FAISS、Chroma、Milvus、Qdrant、pgvector |
| **重排序** | Cross-Encoder 二次精排，弥补 Bi-Encoder 损失 | 后续章节 | BGE-Reranker、Cohere Rerank |
| **Prompt 工程** | 上下文拼接 + 角色约束 + 引用规范 | 后续章节 | LCEL、Template、Few-shot |
| **LLM 生成** | 带引用的回答、流式输出、工具调用 | 后续章节 | Qwen / GPT / Claude / DeepSeek |
| **多模态扩展** | 图片 / 表格 / 公式解析 + 视觉摘要 | `05-image-handler.py`、`07-md_im.py` | Qwen-VL、GPT-4o、Claude Vision |
| **对象存储** | 解析产出的图片、原始文件归档 | `06-upload_image.py`、`07-md_im.py` | MinIO / S3 / OSS / COS |
| **评估** | Recall@k / MRR / 人工 spot-check | 后续章节 | RAGAS、TruLens、LangSmith |

### 3. 进阶优化方向

- **混合检索**：BM25 关键词 + 向量召回，互补语义鸿沟
- **HyDE / Query Rewrite**：先生成假设性回答再检索，提升长尾 query 召回
- **多路召回 + RRF 融合**：用 Reciprocal Rank Fusion 合并多个召回源
- **元数据过滤**：按时间 / 作者 / 部门过滤，缩小搜索空间
- **Self-RAG / CRAG**：让 LLM 自评检索结果，不达标就改写问题或拒答
- **Agentic RAG**：让 LLM 决定何时检索 / 检索几次 / 检索什么
- **GraphRAG / LightRAG**：把图谱结构引入检索，擅长全局性问题

### 4. 常见坑点

1. **PDF 解析丢表格 / 公式** → 用 MinerU / PaddleOCR / Unstructured hi_res
2. **图片孤岛** → MD 切分后图片 URL 失效，必须先**归档到对象存储**再切
3. **Chunk 切碎** → 表格 / 代码块被切到两个 chunk 里 → 用专用 splitter
4. **Embedding 维度不匹配** → 换模型时务必同步重建索引
5. **Prompt 中塞不下 Top-K 全文** → 加 rerank / context compression

### 5. 本流水线的端到端流程图

```
[knowledge_base/sample.pdf]
            │
            │  mineru -p ... -o ... -b pipeline         (Step 1, 04)
            ▼
[result/sample/auto/sample.md + images/*.jpg]          (Step 2)
            │
            │  正则扫描 MD 找图片引用 + 上下文         (Step 3, 07)
            ▼
[有效图片清单 (filename, path, context)]                (Step 3)
            │
            │  qwen-vl-plus: data:image/jpeg;base64,…  (Step 4, 05/07)
            ▼
[图片中文摘要 dict]                                    (Step 4)
            │
            │  minio.fput_object → 远端 URL              (Step 5, 06/07)
            ▼
[新 MD sample_new.md：本地路径 → MinIO 远端 URL]       (Step 6)
            │
            ▼
       【下一步：切分 / Embedding / 检索 / 生成】
```

---

## 二、MinerU + VLM + MinIO 实现（一个 py 文件）

### 2.1 文件清单（目录自包含，零外部依赖）

```
0914work/rag/
├── rag_mineru_pipeline.py     # 整合主程序（≈570 行）
├── README.md                  # 本文件
├── requirements.txt           # 依赖
├── .env.example               # 环境变量模板（不含真实密钥）
├── .gitignore                 # 排除 .env / result / 临时文件
├── knowledge_base/            # 知识库样本（与 01 一致）
│   ├── sample.pdf             # 待解析 PDF
│   ├── sample.txt             # 文本样本
│   └── sample.docx            # Word 样本
└── result/                    # MinerU 解析产物 + 流水线输出
    └── sample/auto/
        ├── sample.md          # 解析后的 Markdown（含图片本地引用）
        ├── sample_new.md      # 流水线输出：链接已替换为 MinIO 远端 URL
        ├── sample_summary.json# 流水线输出：VLM 摘要 + 远端 URL 映射
        └── images/            # 21 张 PDF 抽取出的原始图片
```

### 2.2 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 复制环境变量模板，按需填写（不写也能跑，会用占位文本）
cp .env.example .env

# 3) 一键跑通（默认使用 ./result/sample/auto/sample.md，目录自包含）
python3 rag_mineru_pipeline.py

# 4) 显式指定 MD 路径
python3 rag_mineru_pipeline.py --md ./result/sample/auto/sample.md

# 5) 从 PDF 重新解析（需要先装 MinerU）
python3 rag_mineru_pipeline.py --pdf ./knowledge_base/sample.pdf

# 6) mock 模式：只跑占位逻辑，不连任何外部服务
python3 rag_mineru_pipeline.py --mode mock
```

### 2.3 关键开关

| 参数 | 作用 |
|------|------|
| `--mode live` / `--mode mock` | live 走真实服务；mock 仅做占位验证 |
| `--skip-mineru` | 跳过 PDF 解析（用已有 MD 时） |
| `--skip-vlm` | 跳过 VLM 摘要 |
| `--skip-upload` | 跳过 MinIO 上传（仅做链接替换） |

### 2.4 与 01-07 的差异（重要）

| 维度 | 01-07 原版 | 本整合版 |
|------|-----------|----------|
| 路径 | 硬编码 `E:\Xupeng\04-rag\...` | `Path(__file__).parent` 自适应 |
| API key | 硬编码在源码（**有泄漏风险**） | 全部从 `.env` / 环境变量读 |
| 异常处理 | 失败直接崩 | 三层降级：mineru 缺 / MinIO 缺 / VLM 缺 都能继续 |
| 入口 | 多个 ipynb + 多个 py | **一个 `rag_mineru_pipeline.py`** |
| 运行参数 | 改源码 | argparse 开关 |

---

## 三、运行结果与截图指引

### 3.1 验证脚本能否成功跑通

```bash
cd 个人/0914work/rag
python3 rag_mineru_pipeline.py
```

**预期日志关键节点**（用于截图）：

```
已加载 .env: .../0914work/rag/.env          ← 成功读配置
运行模式: live
[入口] 使用默认 MD: .../result/sample/auto/sample.md
[Step2] MD 长度=... 字符, images_dir=...
[Step3] 命中有效图片 N 张。
[Step4] xxx.jpg -> 中国科学院国家天文台...   ← VLM 摘要
[Step5] 上传成功: xxx.jpg → http://127.0.0.1:9000/...  ← MinIO 上传
[Step6] 已写入: .../sample_new.md
===== 运行结束 =====
```

### 3.2 截图清单

| 截图 | 截图内容 | 用途 |
|------|----------|------|
| 1 | 终端整段 INFO 日志 | 证明流水线跑通 |
| 2 | `sample_new.md` 中图片链接已变成 MinIO URL | 证明链接替换成功 |
| 3 | MinIO 控制台 (http://127.0.0.1:9090) bucket 内文件列表 | 证明图片归档 |
| 4 | `sample_summary.json` 中 summaries 字段 | 证明 VLM 摘要质量 |
| 5 | 本 README 第一节（RAG 总结） | RAG 关键技术参考 |

### 3.3 失败兜底

- **未装 mineru** → 日志提示"未在 PATH 中找到 mineru"，用 `--md` 跳过解析
- **MinIO 未启** → 日志提示"MinIO 初始化失败"，脚本仍会写出占位 URL
- **DASHSCOPE_API_KEY 为空** → 日志提示，使用占位标题 `图片摘要-xxxxxxxx`

---

## 四、推送至 GitHub 远程仓库

> ⚠️ **目标远端**：`https://github.com/liukelly1993-ship-it/rag.git`
> ⚠️ **重要约束**：**只推送 `个人/0914work/rag/` 这一个子目录**，**严禁**把父级 `04-rag` 整仓（含 PDF 资料、API key、MinIO 数据）推上去。

### 4.1 一次性准备

```bash
cd 个人/0914work/rag

# 1) 在本子目录内独立建仓（不影响父级 04-rag 仓库）
git init
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 邮箱"

# 2) 确认 .env 不在待提交列表
git status
# 应该看不到 .env，只能看到 .env.example、.gitignore、*.py、README.md

# 3) 逐个 add，不要 git add .
git add .gitignore .env.example
git add rag_mineru_pipeline.py
git add requirements.txt
git add README.md
```

### 4.2 首次提交

```bash
git commit -m "feat: 整合 MinerU+VLM+MinIO 流水线

整合 01-07 既有脚本为单文件 rag_mineru_pipeline.py；
路径自适配，敏感配置全部从 .env 读取；
提供 mock / live 双模式与 --skip-* 系列开关。"
```

### 4.3 关联远端并推送

```bash
# 用 GitHub Personal Access Token 推（不要在命令行写明文密码）
git remote add origin https://<TOKEN>@github.com/liukelly1993-ship-it/rag.git

# 首次推送并设 upstream
git push -u origin main
# 如果远端默认分支是 master，把 main 改成 master
```

### 4.4 推送前自检清单

- [ ] `git status` 无 `.env` / `result/` / 大文件
- [ ] `git log --stat` 看到只有上述 5 个文件被提交
- [ ] 父级 `04-rag` 仓库的 remote 仍是 Gitee，**没有被改动**
- [ ] GitHub 远端仓库 `rag` 是**新建的、空的**（避免与远端历史冲突）
