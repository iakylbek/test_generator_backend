from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from .chunking_languages import LanguageProfile, LanguageRegistry
from .models import Chunk, ProjectFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkerConfig:
    max_chunk_chars: int = 4000
    overlap_lines: int = 3

    def __post_init__(self) -> None:
        if self.max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be positive")
        if self.overlap_lines < 0:
            raise ValueError("overlap_lines must be non-negative")


@dataclass(frozen=True)
class _Metadata:
    chunk_type: str = "code"
    symbol_name: str | None = None
    parent_symbol: str | None = None

    @property
    def qualified_symbol(self) -> str | None:
        return ".".join(filter(None, (self.parent_symbol, self.symbol_name))) or None


@dataclass(frozen=True)
class _Region:
    node: Node
    start: int
    end: int
    metadata: _Metadata


class ProjectChunker:
    def __init__(
        self,
        config: ChunkerConfig | None = None,
        registry: LanguageRegistry | None = None,
        parser_factory: Callable[[str], Parser] = get_parser,
    ) -> None:
        self.config = config or ChunkerConfig()
        self.registry = registry or LanguageRegistry()
        self.parser_factory = parser_factory

    def chunk_project(self, files: Iterable[ProjectFile]) -> list[Chunk]:
        return [chunk for file in files for chunk in self.chunk_file(file)]

    def chunk_file(self, file: ProjectFile) -> list[Chunk]:
        if not file.content.strip() or "\x00" in file.content:
            return []
        language, is_config = self.registry.detect(file.path, file.content)
        worker = _FileChunker(file, language, self.config, self.registry.profile_for(language))
        metadata = _Metadata("config" if is_config else "code")
        tree = None
        if language not in {"unknown", "text"}:
            try:
                tree = self.parser_factory(language).parse(worker.source)
            except Exception as error:
                logger.warning("Tree-sitter unavailable for %s (%s): %s", file.path, language, error)
        if tree is None:
            worker.split_lines(0, len(worker.source), metadata)
            return worker.chunks
        worker.parser = "tree-sitter"
        if tree.root_node.has_error:
            logger.debug("Partial syntax tree for %s; preserving unrecognized source", file.path)
        if is_config:
            worker.split_node(tree.root_node, 0, len(worker.source), metadata)
        else:
            regions = worker.find_regions(tree.root_node.named_children, metadata)
            worker.split_regions(regions, 0, len(worker.source), metadata)
        return worker.chunks


class _FileChunker:

    def __init__(self, file: ProjectFile, language: str, config: ChunkerConfig,
                 profile: LanguageProfile) -> None:
        self.file = file
        self.language = language
        self.config = config
        self.profile = profile
        self.source = file.content.encode("utf-8")
        self.line_starts = [0] + [i + 1 for i, byte in enumerate(self.source) if byte == 10]
        self.parser = "lines"
        self.chunks: list[Chunk] = []

    def text(self, start: int, end: int) -> str:
        return self.source[start:end].decode("utf-8")

    def fits(self, start: int, end: int) -> bool:
        return len(self.text(start, end)) <= self.config.max_chunk_chars

    def emit(self, start: int, end: int, metadata: _Metadata) -> None:
        content = self.text(start, end)
        if not content.strip():
            return
        self.chunks.append(Chunk(
            file_path=self.file.path, language=self.language,
            chunk_type=metadata.chunk_type, symbol_name=metadata.symbol_name,
            parent_symbol=metadata.parent_symbol,
            start_line=bisect_right(self.line_starts, start),
            end_line=bisect_right(self.line_starts, end - 1),
            content=content, start_byte=start, end_byte=end, parser=self.parser,
        ))

    def split_lines(self, start: int, end: int, metadata: _Metadata) -> None:
        if self.fits(start, end):
            self.emit(start, end, metadata)
            return
        units: list[tuple[int, int, int]] = []
        offset = start
        limit = self.config.max_chunk_chars
        for line in self.text(start, end).splitlines(keepends=True):
            for index in range(0, len(line), limit):
                fragment = line[index:index + limit]
                next_offset = offset + len(fragment.encode("utf-8"))
                units.append((offset, next_offset, len(fragment)))
                offset = next_offset
        first = 0
        while first < len(units):
            stop = first
            size = 0
            while stop < len(units) and size + units[stop][2] <= limit:
                size += units[stop][2]
                stop += 1
            self.emit(units[first][0], units[stop - 1][1], metadata)
            if stop == len(units):
                break
            following = stop
            size = units[stop][2]
            while (following > max(first + 1, stop - self.config.overlap_lines)
                   and size + units[following - 1][2] <= limit):
                following -= 1
                size += units[following][2]
            first = following

    def symbol_name(self, node: Node) -> str | None:
        for field in self.profile.node_name_fields.get(node.type, self.profile.name_fields):
            candidate = node.child_by_field_name(field)
            if candidate is None:
                continue
            while candidate.child_by_field_name("declarator") is not None:
                candidate = candidate.child_by_field_name("declarator")
            return self.text(candidate.start_byte, candidate.end_byte)
        return None

    def explicit_parent(self, node: Node) -> str | None:
        fields = self.profile.parent_fields.get(node.type, ())
        candidate = node
        for field in fields:
            stack = [candidate]
            found = None
            while stack:
                current = stack.pop()
                found = current.child_by_field_name(field)
                if found is not None:
                    break
                stack.extend(reversed(current.named_children))
            if found is None:
                return None
            candidate = found
        if not fields:
            return None
        return self.text(candidate.start_byte, candidate.end_byte).lstrip("*&")

    def describe(self, node: Node, parent: _Metadata) -> _Region | None:
        original = node
        while node.type in self.profile.wrappers:
            children = [child for child in node.named_children
                        if child.type not in {"comment", "decorator"}]
            if len(children) != 1:
                break
            node = children[0]
        kind = self.profile.node_types.get(node.type)
        if kind is None:
            return None
        name = self.symbol_name(node)
        value = node.child_by_field_name("value") or node.child_by_field_name("right")
        if value is not None and value.type in {
            "arrow_function", "function_expression", "lambda", "lambda_expression",
        }:
            kind = "function"
        type_node = node.child_by_field_name("type")
        if kind == "declaration" and type_node is not None:
            kind = self.profile.node_types.get(type_node.type, kind)
        if kind == "function" and parent.chunk_type in {
            "class", "struct", "interface", "implementation",
        }:
            kind = "method"
        if kind == "import":
            name = None
        end = original.end_byte
        following = original.next_sibling
        if following is not None and following.type == ";":
            end = following.end_byte
        return _Region(node, original.start_byte, end,
                       _Metadata(kind, name, self.explicit_parent(node) or parent.qualified_symbol))

    def find_regions(self, nodes: Iterable[Node], parent: _Metadata) -> list[_Region]:
        regions: list[_Region] = []
        stack = list(reversed(list(nodes)))
        while stack:
            node = stack.pop()
            region = self.describe(node, parent)
            if region is not None:
                regions.append(region)
            else:
                stack.extend(reversed(node.named_children))
        return regions

    def split_regions(self, regions: list[_Region], start: int, end: int,
                      metadata: _Metadata) -> None:
        cursor = start
        for region in regions:
            self.split_lines(cursor, region.start, metadata)
            self.split_node(region.node, region.start, region.end, region.metadata)
            cursor = region.end
        self.split_lines(cursor, end, metadata)

    def split_node(self, node: Node, start: int, end: int, metadata: _Metadata) -> None:
        if self.fits(start, end):
            self.emit(start, end, metadata)
            return
        if metadata.chunk_type != "config":
            regions = self.find_regions(node.named_children, metadata)
            if regions:
                self.split_regions(regions, start, end, metadata)
                return
        body = node.child_by_field_name("body")
        children = (body or node).named_children
        while (len(children) == 1 and children[0].start_byte == start
               and children[0].end_byte == end):
            children = children[0].named_children
        children = [child for child in children
                    if start <= child.start_byte < child.end_byte <= end
                    and (child.start_byte > start or child.end_byte < end)]
        if not children:
            self.split_lines(start, end, metadata)
            return
        pending = start
        cursor = start
        for child in children:
            if self.fits(pending, child.end_byte):
                cursor = child.end_byte
                continue
            self.split_lines(pending, cursor, metadata)
            self.split_lines(cursor, child.start_byte, metadata)
            if self.fits(child.start_byte, child.end_byte):
                pending = child.start_byte
            else:
                self.split_node(child, child.start_byte, child.end_byte, metadata)
                pending = child.end_byte
            cursor = child.end_byte
        if self.fits(pending, end):
            self.emit(pending, end, metadata)
        else:
            self.split_lines(pending, cursor, metadata)
            self.split_lines(cursor, end, metadata)
