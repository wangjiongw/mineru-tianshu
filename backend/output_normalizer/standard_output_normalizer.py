"""
标准输出规范化器
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger
import shutil
import re
from .base_output_normalizer import BaseOutputNormalizer


class StandardOutputNormalizer(BaseOutputNormalizer):
    """
    标准输出规范化器（MinerU 等）

    将不同引擎的输出统一为标准格式：
    - full.md:         原始 Markdown，图片路径保持 images/xxx.jpg
    - result.md:       处理版，图片路径替换为 RustFS URL 或本地 API 路径
    - images/:         图片目录（统一名称）
    - result.json:     结构化数据（content_list）
    - content_list.json: result.json 的兼容别名
    - mineru_model.json: MinerU 模型输出（如果存在）
    """

    def __init__(self, preserve_intermediate_files: bool = False, artifact_family: Optional[str] = None):
        super().__init__()
        self.preserve_intermediate_files = preserve_intermediate_files
        self.artifact_family = artifact_family

    def _preserve_intermediate_files(self) -> bool:
        return self.preserve_intermediate_files

    def _normalize_local_files(self, output_dir: Path) -> Dict[str, Any]:
        result = {
            "markdown_file": None,   # result.md，供 base 类做 URL 替换
            "full_md_file": None,    # full.md，保持 images/xxx.jpg 不变
            "json_file": None,
            "image_dir": None,
            "image_count": 0,
        }

        # 1. 规范化 Markdown — 同时生成 full.md 和 result.md
        result["full_md_file"], result["markdown_file"] = self._normalize_markdown(output_dir)

        # 2. 规范化图片目录
        result["image_dir"], result["image_count"] = self._normalize_images(output_dir)

        # 3. Preserve MinerU's public content-list artifacts before creating
        # the cross-backend result.json compatibility view.
        if self.artifact_family == "mineru":
            self._preserve_mineru_content_lists(output_dir)

        # 4. 规范化 JSON 文件
        result["json_file"] = self._normalize_json(output_dir)
        self._ensure_content_list_alias(output_dir)

        # 5. 复制 MinerU model JSON
        self._copy_model_json(output_dir)

        # 6. 两个 md 文件都先统一为 images/xxx.jpg（base 类后续只修改 result.md）
        if result["image_dir"]:
            for md in [result["full_md_file"], result["markdown_file"]]:
                if md:
                    self._update_markdown_image_refs(md)

        # 7. Compact mode only keeps images referenced by canonical outputs.
        if not self._preserve_intermediate_files():
            result["image_count"] = self._prune_unreferenced_images(output_dir)

        # 8. 清理 MinerU 原始输出中不再需要的文件
        self._cleanup_original_files(output_dir)

        return result

    def _preserve_mineru_content_lists(self, output_dir: Path) -> None:
        """Promote MinerU v1/v2 content lists without changing their names."""
        candidates = sorted({
            path
            for pattern in ("*_content_list.json", "*_content_list_v2.json")
            for path in output_dir.rglob(pattern)
            if path.is_file()
        })
        for source in candidates:
            destination = output_dir / source.name
            if source == destination:
                continue
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError(
                        f"Conflicting MinerU content-list artifacts: {source} and {destination}"
                    )
                continue
            shutil.copy2(source, destination)
            logger.info(f"📄 Preserved MinerU artifact: {destination.name}")
            if not self._preserve_intermediate_files():
                source.unlink()

    def _normalize_markdown(self, output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        规范化 Markdown 文件，同时生成 full.md 和 result.md。

        Returns:
            (full_md_path, result_md_path)
        """
        full_md = output_dir / "full.md"
        result_md = output_dir / "result.md"

        # 两个文件都已存在，直接返回
        if full_md.exists() and result_md.exists():
            logger.info("✅ full.md and result.md already exist")
            return full_md, result_md

        # 查找所有 .md 文件（递归），排除已存在的标准文件
        md_files = [
            f for f in output_dir.rglob("*.md")
            if f.name not in ("full.md", "result.md")
        ]

        if not md_files:
            # 如果只有其中一个存在，用它补另一个
            if full_md.exists() and not result_md.exists():
                shutil.copy2(full_md, result_md)
                logger.info("📄 Copied full.md -> result.md")
                return full_md, result_md
            if result_md.exists() and not full_md.exists():
                shutil.copy2(result_md, full_md)
                logger.info("📄 Copied result.md -> full.md")
                return full_md, result_md
            logger.warning("⚠️  No markdown files found")
            return None, None

        # 选择最大的 .md 文件（通常是主文件）
        main_md = max(md_files, key=lambda f: f.stat().st_size)
        logger.info(f"📄 Found main markdown: {main_md.relative_to(output_dir)}")

        # 同时 copy 为 full.md 和 result.md
        if not full_md.exists():
            shutil.copy2(main_md, full_md)
            logger.info(f"   Copied -> full.md")
        if not result_md.exists():
            shutil.copy2(main_md, result_md)
            logger.info(f"   Copied -> result.md")

        # 兼容旧的节省空间模式；默认保留原始 Markdown。
        if main_md.parent == output_dir and not self._preserve_intermediate_files():
            main_md.unlink()
            logger.info(f"   Removed original: {main_md.name}")

        return full_md, result_md

    def _normalize_images(self, output_dir: Path) -> tuple[Optional[Path], int]:
        """
        规范化图片目录

        将所有图片统一到 images/ 目录
        """
        standard_image_dir = output_dir / self.STANDARD_IMAGE_DIR

        # 查找可能的图片目录
        possible_dirs = ["imgs", "images", "img", "pictures", "pics"]
        found_dirs = []

        for dir_name in possible_dirs:
            img_dir = output_dir / dir_name
            if img_dir.exists() and img_dir.is_dir():
                found_dirs.append(img_dir)

        # 如果没有找到图片目录，查找散落的图片文件
        if not found_dirs:
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
            image_files = [f for f in output_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

            if not image_files:
                logger.info("ℹ️  No images found")
                return None, 0

            # 创建标准图片目录并移动图片
            logger.info(f"📁 Creating standard image directory: {self.STANDARD_IMAGE_DIR}/")
            standard_image_dir.mkdir(exist_ok=True)

            for img_file in image_files:
                if img_file.parent != standard_image_dir:
                    dest = standard_image_dir / img_file.name
                    if self._preserve_intermediate_files():
                        logger.debug(f"   Copying while preserving original: {img_file.name}")
                        shutil.copy2(img_file, dest)
                    else:
                        logger.debug(f"   Moving: {img_file.name}")
                        shutil.move(str(img_file), str(dest))

            return standard_image_dir, len(image_files)

        # 如果标准目录已存在，直接返回
        if standard_image_dir in found_dirs:
            image_count = len(list(standard_image_dir.iterdir()))
            logger.info(f"✅ Standard image directory already exists: {self.STANDARD_IMAGE_DIR}/")
            return standard_image_dir, image_count

        # 合并所有图片目录到标准目录
        logger.info(f"📁 Consolidating image directories to: {self.STANDARD_IMAGE_DIR}/")
        standard_image_dir.mkdir(exist_ok=True)

        total_images = 0
        for img_dir in found_dirs:
            logger.info(f"   Processing: {img_dir.name}/")
            for img_file in img_dir.iterdir():
                if img_file.is_file():
                    dest = standard_image_dir / img_file.name
                    # 处理文件名冲突
                    if dest.exists():
                        stem = img_file.stem
                        suffix = img_file.suffix
                        counter = 1
                        while dest.exists():
                            dest = standard_image_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    if self._preserve_intermediate_files():
                        shutil.copy2(img_file, dest)
                    else:
                        shutil.move(str(img_file), str(dest))
                    total_images += 1

            # 仅旧的节省空间模式删除已搬空目录。
            if not self._preserve_intermediate_files():
                try:
                    img_dir.rmdir()
                    logger.debug(f"   Removed empty directory: {img_dir.name}/")
                except OSError:
                    pass

        return standard_image_dir, total_images

    def _normalize_json(self, output_dir: Path) -> Optional[Path]:
        """
        规范化 JSON 文件

        查找并重命名为标准名称：result.json
        """
        # 查找所有 .json 文件（排除子目录中的临时文件）
        json_files = sorted(
            f
            for f in output_dir.rglob("*.json")
            if not f.parent.name.startswith("page_")  # 排除 PaddleOCR-VL 的分页文件
            and f.name != "mineru_model.json"          # 排除已生成的 model json
            and "_content_list_v2" not in f.name
        )

        if not json_files:
            logger.info("ℹ️  No JSON files found")
            return None

        # 如果已经有 result.json，直接返回
        standard_json = output_dir / self.STANDARD_JSON_NAME
        if standard_json.exists():
            logger.info(f"✅ Standard JSON file already exists: {standard_json.name}")
            return standard_json

        # 选择主 JSON 文件（优先选择 content_list.json 或最大的文件，排除 model json）
        main_json = None
        for f in json_files:
            if f.name == "mineru_model.json":
                continue
            if "content_list" in f.name or "result" in f.name:
                main_json = f
                break

        if not main_json:
            candidates = [f for f in json_files if f.name != "mineru_model.json"]
            if candidates:
                main_json = max(candidates, key=lambda f: f.stat().st_size)

        if not main_json:
            return None

        logger.info(f"📄 Found main JSON: {main_json.relative_to(output_dir)}")

        # 如果不在根目录，移动到根目录
        if main_json.parent != output_dir:
            logger.info("   Copying canonical JSON to root directory...")
            shutil.copy2(main_json, standard_json)
            if not self._preserve_intermediate_files():
                main_json.unlink()
        else:
            if self._preserve_intermediate_files() or "content_list" in main_json.name:
                logger.info(f"   Copying to {self.STANDARD_JSON_NAME} while preserving original...")
                shutil.copy2(main_json, standard_json)
            else:
                logger.info(f"   Renaming to {self.STANDARD_JSON_NAME}...")
                main_json.rename(standard_json)

        return standard_json

    def _ensure_content_list_alias(self, output_dir: Path) -> None:
        result_json = output_dir / self.STANDARD_JSON_NAME
        content_list = output_dir / "content_list.json"
        if not result_json.is_file() or content_list.exists():
            return
        try:
            content_list.hardlink_to(result_json)
        except OSError:
            shutil.copy2(result_json, content_list)
        logger.info("📄 Created content_list.json compatibility alias")

    def _copy_model_json(self, output_dir: Path):
        """
        复制 MinerU 的 *_model.json 到根目录，命名为 mineru_model.json。
        """
        dest = output_dir / "mineru_model.json"
        if dest.exists():
            logger.info("✅ mineru_model.json already exists")
            return

        model_jsons = sorted(
            f for f in output_dir.rglob("*_model.json")
            if f.name != "mineru_model.json"
        )
        if model_jsons:
            source = model_jsons[0]
            shutil.copy2(source, dest)
            if not self._preserve_intermediate_files():
                source.unlink()
            logger.info(f"📄 Copied model JSON: {source.name} -> mineru_model.json")
        else:
            logger.debug("ℹ️  No *_model.json found, skipping")

    def _prune_unreferenced_images(self, output_dir: Path) -> int:
        image_dir = output_dir / self.STANDARD_IMAGE_DIR
        if not image_dir.is_dir():
            return 0
        referenced = set()

        def add_reference(value):
            if not isinstance(value, str):
                return
            if "images/" in value or "/images/" in value:
                name = Path(value.split("?", 1)[0]).name
                if name:
                    referenced.add(name)

        for md_name in ("full.md", "result.md"):
            md = output_dir / md_name
            if not md.is_file():
                continue
            content = md.read_text(encoding="utf-8")
            for left, right in re.findall(
                r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)", content
            ):
                add_reference(left or right)

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                add_reference(value)

        json_paths = {
            output_dir / "result.json",
            output_dir / "mineru_model.json",
            *output_dir.glob("*_content_list.json"),
            *output_dir.glob("*_content_list_v2.json"),
        }
        for path in sorted(json_paths):
            if not path.is_file():
                continue
            try:
                import json
                walk(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning(f"Could not inspect {path.name} image references: {exc}")

        for image in list(image_dir.iterdir()):
            if image.is_file() and image.name not in referenced:
                image.unlink()
        return sum(1 for image in image_dir.iterdir() if image.is_file())

    def _cleanup_original_files(self, output_dir: Path):
        """
        清理 MinerU 原始输出中不再需要的文件：
        - *_layout.pdf
        - *.origin.pdf
        - 子目录中的 images/ 文件夹（根目录的 images/ 保留）
        """
        standard_image_dir = output_dir / self.STANDARD_IMAGE_DIR

        if self._preserve_intermediate_files():
            logger.info("📦 Preserving MinerU intermediate files and original directory tree")
            return

        # 兼容旧的节省空间模式：删除诊断 PDF 和子目录图片。
        for pattern in ("*_layout.pdf", "*_span.pdf", "*_origin.pdf", "*.origin.pdf", "*_middle.json"):
            for f in output_dir.rglob(pattern):
                try:
                    f.unlink()
                    logger.info(f"🗑️  Deleted: {f.relative_to(output_dir)}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete {f.name}: {e}")

        # 删除子目录中的 images/ 文件夹（不删除根目录的标准 images/）
        for img_dir in output_dir.rglob("images"):
            if img_dir.is_dir() and img_dir != standard_image_dir:
                try:
                    shutil.rmtree(img_dir)
                    logger.info(f"🗑️  Deleted directory: {img_dir.relative_to(output_dir)}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete directory {img_dir.name}: {e}")

    def _update_markdown_image_refs(self, markdown_file: Path):
        """
        更新 Markdown 文件中的图片引用

        将所有图片路径统一为 images/xxx.jpg 格式
        支持两种格式：
        1. Markdown 语法：![alt](path)
        2. HTML 标签：<img src="path" ...>
        """
        try:
            content = markdown_file.read_text(encoding="utf-8")

            # 1. 匹配 Markdown 图片语法：![alt](path)
            md_img_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

            def replace_md_path(match):
                alt_text = match.group(1)
                img_path = match.group(2)

                # 提取文件名
                img_filename = Path(img_path).name

                # 统一为 images/filename 格式
                new_path = f"{self.STANDARD_IMAGE_DIR}/{img_filename}"

                return f"![{alt_text}]({new_path})"

            # 2. 匹配 HTML img 标签：<img src="path" ...>
            html_img_pattern = r'<img\s+([^>]*\s+)?src="([^"]+)"([^>]*)>'

            def replace_html_path(match):
                before_src = match.group(1) or ""
                img_path = match.group(2)
                after_src = match.group(3) or ""

                # 提取文件名
                img_filename = Path(img_path).name

                # 统一为 images/filename 格式
                new_path = f"{self.STANDARD_IMAGE_DIR}/{img_filename}"

                return f'<img {before_src}src="{new_path}"{after_src}>'

            # 替换所有图片引用
            new_content = re.sub(md_img_pattern, replace_md_path, content)
            new_content = re.sub(html_img_pattern, replace_html_path, new_content)

            # 只有内容变化时才写入
            if new_content != content:
                markdown_file.write_text(new_content, encoding="utf-8")
                logger.info(f"✅ Updated image references in {markdown_file.name}")
            else:
                logger.debug(f"ℹ️  No image references to update in {markdown_file.name}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to update image references: {e}")
