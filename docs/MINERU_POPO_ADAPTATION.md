# MinerU 批量解析结果与 MinerU-Popo 适配说明

**文档定位**：MinerU 生产端的数据契约与运行状态说明<br>
**主要读者**：MinerU-Popo 适配开发者、批处理运维人员<br>
**最后核对时间**：2026-09-16 UTC<br>
**权威实现**：`mineru-tianshu` 的输出归一化器、OA inventory 与批次运行脚本

> 本文中的任务数字是带时间戳的运行快照，不能作为程序常量。目录结构、文件语义和就绪检查才是后处理适配应依赖的契约。

## 1. 先看结论

- 本文同时描述单个 PDF 从上传到下载的服务内生命周期，以及 OA 结果从 `mineru_outputs` 归档到 `pdfs_parsed` 的批处理生命周期。
- 默认模式为 `hybrid-auto-engine + effort=high`；本地 vLLM 可用时实际转换为 `hybrid-http-client`。
- 上传 PDF 以 `UUID_原文件名` 落盘并计算 SHA256；MinerU 解析前还会用 PDFium 重建页面，因此 `*_origin.pdf` 通常不是上传文件的逐字节副本。
- 一次解析模式运行只生成一个 `*_origin.pdf`。同一结果中两个 origin 位于不同解析模式目录，表示两次或两种模式的历史产物被共同保留。
- 稳定消费文件是根目录的 `result.json`、`full.md`、`result.md`、`mineru_model.json` 和 `images/`；解析模式子目录属于诊断性原始产物。
- Popo 的首选结构化输入是每篇结果根目录下的 `result.json`，它是当前 MinerU `content_list` 的规范化副本。
- `mineru_model.json` 是低层模型输出，不应优先于 `result.json`。当前 Popo 的 MinerU Reader 正好相反，需要调整。
- Popo 推理必须能打开原始 PDF。应通过 SHA256 直接定位源 PDF，不要依赖结果目录中带随机前缀的 PDF 文件名。
- `_inventory/progress.json` 的全局结果分类可能滞后于同步守护进程；`completed_db_unsynced` 不能单独证明文件尚未同步。最终可消费性必须通过目标结果目录的文件门槛确认。
- Popo 可以改善标题层级、跨页文本/表格关系和图文关联，但不能把 MinerU 已经裁成多个文件的组合图自动重新拼成一张图。

## 2. 数据处理链路

```text
源 PDF（SHA256 分片）
  /share/.../pdfs/oa/by_sha256/{前2位}/{sha256}.pdf
                │
                ▼
OA inventory + 5000 文件批次
  oa_queue.sqlite3 / progress.json / batches.csv
                │
                ▼
Redis 队列 + MinerU 任务数据库
                │
                ▼
Worker 池 ─────────────────► 本地 vLLM / NPU
  8101–8108（按实例启停）       30025–30032（按实例启停）
                │
                ▼
临时任务结果 mineru_outputs
                │
                ▼
sync_results_to_parsed.py
                │
                ▼
稳定结果（SHA256 分片）
  /share/.../pdfs_parsed/{前2位}/{sha256}/
                │
                ▼
MinerU-Popo 归一化、推理与文档树构建
```

单个 PDF 在天枢服务内的生命周期：

```text
上传 → UUID 文件落盘、SHA256、数据库与 Redis
  → 可选拆分/去水印
  → PDFium 重建选定且可读的页面
  → Hybrid layout/OCR/VLM/公式/表格处理
  → MinerU 原始产物
  → 天枢标准化
  → 数据库 completed、API/前端下载
  → 可选归档到 pdfs_parsed/{sha前2位}/{sha256}/
```

### 2.1 关键路径

| 用途 | 路径 |
|---|---|
| OA 源 PDF | `/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256` |
| MinerU 稳定结果 | `/share/wangjiong/databases/escorpus-assets/pdfs_parsed` |
| OA inventory | `/share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/oa_queue.sqlite3` |
| 全局进度 | `/share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/progress.json` |
| 批次明细 | `/share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/batches.csv` |
| Runner checkpoint | `/share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/oa_batch_runner.checkpoint.json` |
| MinerU 任务数据库 | `/share/wangjiong/databases/mineru_database/mineru-runner-worker-0/mineru_tianshu.db` |
| MinerU 日志 | `/share/wangjiong/databases/mineru_database/mineru-runner-worker-0/mineru_logs` |
| Popo 代码 | `/data/projects/mineru/MinerU-Popo` |

### 2.2 上传、排队和预处理

1. `/api/v1/tasks/submit` 将内容写入 `mineru_uploads/UUID_原文件名`，以 8 MiB 分块计算 SHA256，并把任务参数写入 SQLite、任务消息写入 Redis。
2. 去重键为 `(file_hash, backend, lang, method)`；相同组合复用已有逻辑任务，失败或超时任务可重新入队。
3. Worker 可选执行 Office 转 PDF 和去水印。当前超过 50 页的 PDF 按 50 页拆成子任务，分页 PDF 暂存在 `mineru_outputs/splits/{parent_task_id}/`。
4. 子任务完成后，父任务按页序合并 Markdown 和 content-list。分页输入会删除，但子任务结果目录不会删除，其中的 JSON、图片和 origin PDF 可能继续保留。

### 2.3 PDFium 重建与双 origin PDF

进入 MinerU 前，`do_parse` 会用 PDFium 新建文档、导入指定或确认可读的页面，再重新保存。`*_origin.pdf` 保存的是重建后的字节，不是上传源文件的直接复制品。

即使页面渲染完全一致，重建也可能改变 PDF Info、Producer/Creator、对象编号、xref、trailer、stream 压缩和文件大小。因此比较两个 origin 时，应同时检查文件 SHA256、页数、逐页渲染哈希和元信息。

MinerU 每个解析模式目录只调用一次输出函数、生成一个 origin。两个 origin 位于不同模式目录，说明任务目录中同时保留了不同运行的产物。常见形成过程是：

- 同一文档曾以不同 backend 或 parse method 处理，产生 `hybrid_auto/`、`hybrid_ocr/`、`vlm/` 或 `pipeline/` 等不同目录；
- 重试不会统一清除其他模式目录；
- `sync_results_to_parsed.py` 在目标已存在时采用 no-clobber 合并：旧条目优先，只补充缺失的顶层条目，因此两个模式目录会共同保留。

### 2.4 默认 Hybrid 解析

默认参数为 `backend=hybrid-auto-engine`、`method=auto`、`effort=high`、`lang=auto`，公式和表格识别开启。本地 vLLM 可用时后端转为 `hybrid-http-client`；当前引擎会把 `lang=auto` 转成 `en`。

Hybrid 执行布局检测、OCR/VLM 块识别、公式与表格识别、段落重建，并形成模型输出和 middle JSON。API 虽默认 `draw_span_bbox=true`，Hybrid 内部会强制关闭 Span 绘制，因此默认 Hybrid 任务通常没有 `*_span.pdf`。

### 2.5 MinerU 原始产物

天枢把内部安全文件名固定为 `result.pdf`。典型原始目录如下：

```text
result.pdf/
└── hybrid_auto/
    ├── images/
    ├── result.pdf.md
    ├── result.pdf_content_list.json
    ├── result.pdf_content_list_v2.json
    ├── result.pdf_middle.json
    ├── result.pdf_model.json
    ├── result.pdf_origin.pdf
    └── result.pdf_layout.pdf
```

- `*.md`：由最终 block 顺序生成的 Markdown。
- `*_content_list.json`：扁平 block 列表，是 `result.json` 的首选来源。
- `*_content_list_v2.json`：按页组织、语义更丰富的第二版内容列表。
- `*_middle.json`：页面尺寸、嵌套 block、bbox、丢弃块等处理中间状态。
- `*_model.json`：布局/VLM 等模型的低层输出。
- `*_origin.pdf`：PDFium 重建后的实际解析输入。
- `*_layout.pdf`：带版面框的诊断 PDF。
- `images/`：MinerU 生成的全部图片裁剪；其中只有一部分会被 Markdown 或 content-list 引用。

### 2.6 天枢标准化和保留规则

| 标准产物 | 处理 |
|---|---|
| `full.md` | 从主 Markdown 复制，图片保持 `images/...` 相对路径 |
| `result.md` | 同一 Markdown 的展示版；图片改成本地 API 或 RustFS URL |
| `result.json` | 选择首个文件名含 `content_list`/`result` 的 JSON；当前生成顺序下通常是 v1 content-list |
| `mineru_model.json` | 复制首个 `*_model.json` |
| `images/` | 汇总 MinerU 生成的全部图片到根目录，不按 Markdown/content-list 引用过滤 |
| 源 PDF 副本 | 标准化后由 Worker 从上传目录复制，供前端预览 |

产物保留由任务参数 `preserve_all_artifacts` 控制。默认 `false` 为精简模式：不生成或清理 `*_layout.pdf`、`*_span.pdf`、`*_origin.pdf`、`*_middle.json` 和 `*_content_list_v2.json`，保留 Markdown、content-list、model、标准结果、根 `images/` 与源 PDF 副本。设为 `true` 时打开全部 MinerU dump/draw 开关，标准化器只复制标准文件，不移动或删除解析模式目录中的原始产物。两种策略属于不同的任务去重键。

当前精简模式仍会保留根 `images/` 中未被 Markdown/content-list 引用的中间裁剪图；按引用过滤属于后续优化，不能把根 `images/` 文件数解释为最终正文图片数。

### 2.7 下载边界

前端以 `format=both` 获取结果：Markdown 和 JSON 下载由浏览器根据 API 内容重新生成；PDF 选择顺序为 layout、span、其他 PDF，默认 Hybrid 标准化后通常回落到根目录源 PDF 副本；图片通过 `/api/v1/files/output/.../images/...` 加载。

已完成任务可通过带用户鉴权和任务归属校验的 `/api/v1/tasks/{task_id}/download` 下载服务端实际保留的完整结果 ZIP；批处理脚本默认使用该接口。

通用 `files/output` 与 `files/upload` 接口能按已知相对路径读取目录内任意文件，因此原始 middle/model/content-list/origin 也可直接下载。接口有目录越界防护，但当前没有用户鉴权和任务归属校验。

## 3. 文档寻址规则

OA 文献的稳定主键是源 PDF 内容的64位小写 SHA256。目录采用前两个十六进制字符分片。

设：

```text
sha256 = 0101c4344b51c435f1bbeaf6c55a70b2cbb3fa658a7e4d4436bdeb6e18cb127c
shard  = 01
```

则：

```text
源 PDF：
/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256/01/
  0101c4344b51c435f1bbeaf6c55a70b2cbb3fa658a7e4d4436bdeb6e18cb127c.pdf

解析结果：
/share/wangjiong/databases/escorpus-assets/pdfs_parsed/01/
  0101c4344b51c435f1bbeaf6c55a70b2cbb3fa658a7e4d4436bdeb6e18cb127c/
```

结果目录内可能另有一份原始 PDF，其名称通常为 `<任务或随机前缀>_<sha256>.pdf`。该名称不是稳定接口，只能作为源 PDF 不可用时的 glob 回退。

## 4. 单篇解析结果的数据契约

### 4.1 目录结构

```text
{sha256}/
├── result.json                 # 规范化 content_list；Popo 首选输入
├── full.md                     # 图片保持 images/... 相对路径
├── result.md                   # 图片可能改写为本地 API/RustFS URL
├── mineru_model.json           # 低层模型输出；诊断/回退用途
├── images/                     # 图片、图表和表格裁剪
├── <prefix>_{sha256}.pdf       # 可选的原始 PDF 副本，文件名不稳定
└── hybrid_auto/                # 可选的 MinerU 原始详细产物
    ├── *_content_list.json
    ├── *_content_list_v2.json
    ├── *_middle.json
    ├── *_model.json
    ├── *.md
    └── *_origin.pdf
```

`hybrid_auto/` 是实现细节，旧结果或清理后的结果中不保证所有文件都存在。后处理的基础能力必须只依赖根目录标准文件和源 PDF。

### 4.2 文件语义与优先级

| 文件 | 稳定性 | 语义 | Popo 用法 |
|---|---:|---|---|
| `result.json` | 高 | 扁平 `content_list`；当前 MinerU 结果中 bbox 通常为 0–1000、`page_idx` 从0开始 | **主输入** |
| `full.md` | 高 | 原始 Markdown，图片引用保持 `images/...` | 人工复核、文本回退 |
| `result.md` | 高 | 展示版 Markdown；本机部署会把图片改写为 `/api/v1/files/output/...` | 不作为结构适配输入 |
| `mineru_model.json` | 中 | 页面级低层模型/Layout 输出 | 诊断用途；禁止作为第一优先级 |
| `images/` | 高 | 全部标准化图片裁剪；`result.json.img_path` 只引用其中一部分 | 按引用消费，未引用文件视为中间产物 |
| `*_content_list.json` | 中/可选 | MinerU 原始扁平内容列表；标准化后通常与 `result.json` 等价 | 诊断标准化来源 |
| `*_content_list_v2.json` | 低/可选 | 按页嵌套的较丰富语义结构 | 第二阶段增强，不作为首版强依赖 |
| `*_middle.json` | 低/可选 | 含 `page_size`、嵌套 block、丢弃块和 caption bbox 的中间表示 | 可增强图文关联 |
| `*_model.json` | 低/可选 | 当次解析模式的模型原始输出；标准化器复制为 `mineru_model.json` | 模型诊断 |
| `*.md` | 低/可选 | MinerU 原始 Markdown；标准化器据此生成根目录两个 Markdown | 追踪标准化前文本 |
| `*_origin.pdf` | 低/可选 | PDFium 重建后的实际解析输入 | 页面范围和坏页诊断，不当作权威源 PDF |

已核对的代表性结果中：

- 根目录 `result.json` 与 `hybrid_auto/*_content_list.json` 字节一致。
- 根目录 `mineru_model.json` 与 `hybrid_auto/*_model.json` 字节一致。
- `result.md` 与 `full.md` 不一定一致，因为只有 `result.md` 会经过图片 URL 改写。

### 4.3 `result.json` 基本形态

顶层是 block 数组，而不是按页对象：

```json
[
  {
    "type": "text",
    "text": "Section title or paragraph",
    "text_level": 1,
    "bbox": [80, 130, 794, 177],
    "page_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/<sha256>.jpg",
    "image_caption": ["Figure caption"],
    "image_footnote": [],
    "content": "",
    "bbox": [137, 68, 860, 445],
    "page_idx": 6
  },
  {
    "type": "table",
    "img_path": "images/<sha256>.jpg",
    "table_caption": ["Table caption"],
    "table_footnote": [],
    "table_body": "<table>...</table>",
    "bbox": [295, 70, 915, 137],
    "page_idx": 7
  }
]
```

实际结果还出现 `chart`、`equation`、`list`、`ref_text`、`header`、`footer`、`page_number` 和 `page_footnote` 等类型。适配器不能只处理 `text/image/table` 三类。

### 4.4 坐标和页码

- `result.json.page_idx`：从0开始。
- Popo 页字典键和内部 `page`：从1开始。
- `result.json.bbox`：`[x1, y1, x2, y2]`，当前样本位于约 0–1000 范围。
- Popo 推理输入 bbox：同样是 `xyxy`，但必须归一化到 0–1。

基础转换：

```text
popo_page = page_idx + 1
popo_bbox = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]
```

使用 `middle.json` 时不能套用 `/1000`。其中 bbox 通常是 PDF 页面坐标，应使用对应 `pdf_info[].page_size` 的宽高分别归一化。

### 4.5 两个 origin PDF 的核验方法

先保留相对目录信息，再比较文件本身：

```bash
find <result_dir> -type f -name '*_origin.pdf' -print
sha256sum <origin-a.pdf> <origin-b.pdf>
pdfinfo <origin-a.pdf>
pdfinfo <origin-b.pdf>
```

若 SHA256 或元信息不同，但需要确认页面视觉内容是否一致，可渲染后逐页比较：

```bash
mkdir -p /tmp/mineru-origin-a /tmp/mineru-origin-b
pdftoppm -png -r 144 <origin-a.pdf> /tmp/mineru-origin-a/page
pdftoppm -png -r 144 <origin-b.pdf> /tmp/mineru-origin-b/page
diff \\
  <(sha256sum /tmp/mineru-origin-a/page-*.png | awk '{print $1}') \\
  <(sha256sum /tmp/mineru-origin-b/page-*.png | awk '{print $1}')
```

判定规则：

- 相对目录不同：先按目录名识别 backend/parse method 或分片任务来源。
- 页数、页面渲染哈希相同：可认为页面视觉内容一致。
- PDF 文件 SHA256、元信息或内部对象结构不同：符合独立 PDFium 重建的预期。
- 页面渲染哈希不同：继续检查页码范围、坏页跳过、去水印和大 PDF 分片参数。

## 5. Popo 适配契约

### 5.1 为什么当前代码不能直接扫描 `pdfs_parsed`

当前 `MinerU-Popo/post_processing/label_normalization.py` 的 `MineruReader` 假定：

```text
<input_dir>/<doc_id>/vlm/<doc_id>_model.json
<input_dir>/<doc_id>/vlm/<doc_id>_middle.json
<input_dir>/<doc_id>/vlm/<doc_id>_content_list.json
```

并按 `model → middle → content_list` 的顺序选择输入。当前生产结果存在四个不匹配点：

1. `pdfs_parsed` 顶层是 `00`–`ff` 分片，不是文档目录；直接传给 `--input-dir` 会把分片名误认为 `doc_id`。
2. 实际文档路径是 `{shard}/{sha256}/`，没有固定的 `vlm/` 层。
3. 标准文件名是根目录 `result.json` 和 `mineru_model.json`。
4. 优先读 model 会漏掉标题、列表、表格正文等已经在 content-list 中整理好的语义内容。

另外，Popo 的默认 `--pdf-dir` 假定 PDF 平铺为 `<doc_id>.pdf`，而当前源 PDF 也是 SHA 分片目录。适配器必须直接构造源 PDF 路径，或者生成 `--pdf-map-json`。

### 5.2 推荐的归一化输出

每篇文档生成一个 `{sha256}.json`：

```json
{
  "model": "mineru",
  "doc_id": "<sha256>",
  "input_label": "/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256/<shard>/<sha256>.pdf",
  "pages": {
    "1": [
      {
        "type": "title",
        "content": "Document title",
        "bbox": [0.08, 0.13, 0.794, 0.177],
        "title_level": 1,
        "source_label": "text",
        "source_id": "<sha256>:0"
      }
    ]
  }
}
```

Popo 推理入口会校验：

- `pages` 必须是对象，page key 可转换为整数。
- 每页内容必须是 block 数组。
- bbox 必须是四个有限数，顺序为 `xyxy` 且默认范围为 0–1。
- block type 必须落在 Popo 当前允许集合中。
- `input_label` 必须是 Popo 进程能够打开的真实 PDF 路径。

### 5.3 推荐类型映射

| MinerU `result.json` | Popo type | 内容来源/处理 |
|---|---|---|
| `text` 且有 `text_level` | `title` | `content=text`，保留 `title_level` |
| `text`、`ref_text` | `text` | `content=text` |
| `list` | `list_item` | 将 `list_items` 按换行连接为一个布局 block，避免重复同一 bbox |
| `equation`、`interline_equation` | `equation` | `content=text` |
| `image` | `image` | 主视觉 block；保留 `img_path` 到 adapter metadata/manifest |
| `chart` | `image` | 当前 Popo 输入校验器不接受 `chart`；用 `source_label=chart` 保留来源类型 |
| `table` | `table` | `content=table_body`，必须保留 HTML 以支持跨页表格分析 |
| `header` | `header` | `content=text` |
| `footer` | `footer` | `content=text` |
| `page_number` | `page_number` | `content=text` |
| `page_footnote` | `page_footnote` | `content=text` |

`result.json` 中 image/table/chart 的 caption 和 footnote 是嵌套字符串，通常没有独立 bbox。不要简单复制父图 bbox 冒充 caption bbox。需要图文关联增强时，从 `hybrid_auto/*_middle.json` 的视觉 block 子项中提取 `image_caption`、`table_caption`、`image_footnote` 和 `table_footnote` 的真实 bbox，再合并进 Popo pages；找不到中间文件时保留视觉 block，但将 caption 降级为元数据或内容补充，并记录降级原因。

### 5.4 就绪判定

inventory 状态适合做候选筛选，不适合作为唯一文件门槛：

- `complete_valid`：inventory 已看到严格结果，数据库也为 completed。
- `legacy_result_only`：结果目录通过严格检查，但数据库没有对应 completed 状态。
- `completed_db_unsynced`：数据库完成，但 inventory 最近一次完整扫描没有看到严格结果。同步完成后，在下一次完整 inventory refresh 前仍可能暂时保持此状态。
- `active`、`queued`：任务仍在处理或排队。

推荐的 Popo 文件门槛：

1. `pdfs_parsed/{shard}/{sha256}/result.json` 存在，是非空、可解析的 JSON 数组。
2. 每个待转换 block 的 `page_idx` 和 bbox 合法。
3. 源 PDF `{source_root}/{shard}/{sha256}.pdf` 存在且可打开。
4. 对 image/table/chart 项，存在 `img_path` 时，对应本地文件可读；缺失时记录并按策略跳过或降级。
5. `result.md` 非空、`mineru_model.json` 可解析，作为结果完整性的附加检查。

因此：

- `complete_valid` 和 `legacy_result_only` 通常可以直接进入文件门槛。
- `completed_db_unsynced` 需要实际检查目标目录；文件门槛通过即可消费，不必等待分类刷新。
- 不读取 `mineru_outputs` 中的工作目录，不处理 `active/queued` 任务。

### 5.5 幂等与输出隔离

- 使用 SHA256 作为文档幂等键。
- Popo 输出必须写入独立、可配置的输出根目录，例如 `<popo_output_root>/{shard}/{sha256}/`。
- 不得覆盖 `pdfs_parsed` 中的 MinerU 原始结果。
- 为每篇文档保存输入 `result.json` 的大小、mtime 或内容哈希，以及 adapter 版本。
- 支持 `--resume`：完整输出存在且输入指纹一致时跳过。
- 单篇失败只记录错误并继续，不阻塞整个批次。
- 后处理批次应独立于 MinerU 的5000文件提交批次，按自己的并发和 checkpoint 推进。

## 6. 历史任务运行快照

### 6.1 2026-08-25 15:27 UTC

| 指标 | 数值 |
|---|---:|
| OA 源 PDF 总数 | 458,203 |
| `complete_valid` | 171,073 |
| `legacy_result_only` | 2,332 |
| inventory 严格结果合计 | 173,405（37.84458%） |
| `completed_db_unsynced` | 16,389 |
| `active` | 255 |
| `queued` | 942 |
| `unsubmitted` | 267,205 |
| `failed_permanent` | 7 |
| Redis processing | 244 |
| Redis queued | 1,439 |
| 批次总数 | 66 |
| 当前批次 | `oa-g001-b00012` |
| 当前批次完成/提交中/失败 | 3,825 / 1,175 / 0 |
| deferred backlog | 22 |

Runner checkpoint：

- `action=wait_batch`
- `stop_reason=null`
- `consecutive_health_failures=0`
- 第12批最后提交时间：2026-08-25 13:08:02 UTC

从第12批提交到快照的批次平均速度约为 27.6 份/分钟。同步日志最近五轮共搬运806份结果，约为 27.6 份/分钟，与计算吞吐基本一致。

### 6.2 历史服务拓扑

| NPU | Worker | VLLM |
|---:|---:|---:|
| 0 | 8101 | 30025 |
| 1 | 8102 | 30026 |
| 2 | 8103 | 30027 |
| 3 | 8104 | 30028 |
| 4 | 8105 | 30029 |
| 5 | 8106 | 30030 |
| 6 | 8107 | 30031 |
| 7 | 8108 | 30032 |

快照检查中，上述所有 Worker/VLLM `/health` 端口均返回 HTTP 200。批次 runner、同步 daemon 均存活；同步日志持续出现成功迁移，未出现 missing/conflict。

### 6.3 关于 `completed_db_unsynced`

该状态的字面含义是“任务数据库已 completed，但 inventory 最近一次结果扫描未把目标目录标成严格完成”。它不等于永久提交失败，也不一定表示同步 daemon 当前停止。

该历史快照中：

- `sync_results_to_parsed.py` 持续成功搬运结果并更新任务数据库路径。
- batch runner 的快速刷新主要更新当前批次数据库状态，不会每轮重新扫描全部458K结果目录。
- 因此 `progress.json.updated_at` 很新时，`strict_results_ready` 和 `completed_db_unsynced` 仍可能基于较旧的全目录扫描结果。

Popo 适配器应把 inventory 当作发现和观测来源，把实际文件门槛当作最终判断。

## 7. 实时状态查询

### 7.1 全局进度

```bash
jq '{
  updated_at,
  total_source,
  states,
  strict_results_ready,
  strict_results_ready_ratio,
  db_completed_or_ready,
  remaining_without_strict_result,
  batch_size,
  batch_count,
  queue
}' /share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/progress.json
```

### 7.2 当前批次和健康门禁

```bash
jq '{
  updated_at,
  pid,
  action,
  stop_reason,
  current_batch,
  last_refresh_at,
  last_submit_at,
  consecutive_health_failures,
  deferred_backlog
}' /share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/oa_batch_runner.checkpoint.json
```

### 7.3 最近批次

```bash
tail -n 12 /share/wangjiong/databases/escorpus-assets/pdfs_parsed/_inventory/batches.csv
```

### 7.4 Worker/VLLM 端口

```bash
for port in 8101 8102 8103 8104 8105 8106 8107 8108 \
            30025 30026 30027 30028 30029 30030 30031 30032; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' \
    --connect-timeout 1 --max-time 3 "http://127.0.0.1:${port}/health" || true)
  printf '%s %s\n' "$port" "$code"
done
```

## 8. 适配验收清单

- [ ] 能从 SHA256 唯一定位源 PDF 和 `pdfs_parsed` 结果目录。
- [ ] 不把 `00`–`ff` 分片名误认为文档 ID。
- [ ] 首选根目录 `result.json`，不会先读 `mineru_model.json`。
- [ ] `page_idx + 1`，bbox 正确归一化为 `xyxy_01`。
- [ ] 标题、列表、公式、图像、图表、表格和页面辅助类型均有明确映射。
- [ ] 表格保留 `table_body` HTML。
- [ ] 图注需要真实 bbox 时从 `middle.json` 增强；缺失时明确降级。
- [ ] `input_label` 指向真实、可打开的源 PDF。
- [ ] 文件门槛通过后才提交 Popo，不只依赖 inventory 状态。
- [ ] Popo 输出独立保存、支持断点续跑，单篇异常不会阻塞全局。
- [ ] 不宣称 Popo 会重新拼接 MinerU 已拆分的组合图。

## 9. 已知边界

1. `result.json` 是当前生产输出契约，但不同 MinerU 版本可能增加 block type；适配器应记录未知类型，不能静默丢弃。
2. `hybrid_auto/` 属于可选增强数据，不能作为最小可运行条件。
3. v1 content-list 中嵌套 caption 缺少独立 bbox；高质量图文关联需要从 middle/model 中补充几何信息或接受降级。
4. Popo 的结构后处理作用于语义 block 和文档树，不修改原始 PDF，也不自动合并已经分离的图片文件。
5. 本文的进度数字会持续变化；程序只依赖路径和文件契约。
6. API 保存分页参数为 `start_page/end_page`，当前 MinerU wrapper 读取的是 `start_page_id/end_page_id`；在映射修复前，不能假定 API 分页参数已经生效。
7. 标准化 JSON 选择条件同时匹配 `*_content_list.json` 和 `*_content_list_v2.json`，只取遍历到的第一个文件。当前 MinerU 先写 v1，代表性结果也为 v1，但代码尚未把这一点固化为显式优先级。
8. `*_origin.pdf` 的清理 glob 与实际文件名不匹配，因此它属于当前会保留、但不应作为稳定下游契约的诊断产物。
