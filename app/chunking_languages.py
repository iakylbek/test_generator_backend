from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Mapping

from tree_sitter_language_pack import detect_language, detect_language_from_content


COMMON_NODE_TYPES = {
    **dict.fromkeys(("function_definition", "function_declaration", "function_item",
                     "generator_function_declaration", "local_function_statement"), "function"),
    **dict.fromkeys(("method_definition", "method_declaration", "method",
                     "singleton_method", "constructor_declaration", "destructor_declaration",
                     "method_signature", "abstract_method_signature"), "method"),
    **dict.fromkeys(("class_definition", "class_declaration", "class_specifier", "class"), "class"),
    **dict.fromkeys(("interface_declaration", "interface_type", "trait_item"), "interface"),
    **dict.fromkeys(("struct_item", "struct_specifier", "struct_declaration", "struct_type",
                     "record_declaration"), "struct"),
    **dict.fromkeys(("enum_item", "enum_declaration", "enum_specifier"), "enum"),
    **dict.fromkeys(("import_statement", "import_from_statement", "import_declaration",
                     "preproc_include", "use_declaration", "using_directive",
                     "namespace_use_declaration", "extern_crate_declaration"), "import"),
    **dict.fromkeys(("namespace_definition", "namespace_declaration", "module_definition",
                     "module", "mod_item"), "namespace"),
    "impl_item": "implementation",
    **dict.fromkeys(("type_alias_declaration", "type_item", "type_definition", "type_spec",
                     "const_item", "static_item", "const_declaration", "var_declaration",
                     "variable_declarator", "assignment", "annotated_assignment",
                     "field_declaration", "property_declaration", "event_declaration",
                     "preproc_def", "preproc_function_def", "macro_definition",
                     "package_clause", "package_declaration"), "declaration"),
}


@dataclass(frozen=True)
class LanguageProfile:
    node_types: Mapping[str, str] = field(default_factory=lambda: dict(COMMON_NODE_TYPES))
    wrappers: frozenset[str] = frozenset({
        "decorated_definition", "export_statement", "lexical_declaration",
        "variable_declaration", "declaration", "type_declaration",
    })
    name_fields: tuple[str, ...] = ("name", "declarator", "left")
    node_name_fields: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "impl_item": ("type",),
    })
    parent_fields: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "method_declaration": ("receiver", "type"),
    })


CONFIG_EXTENSIONS = frozenset({
    ".json", ".jsonc", ".json5", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    ".conf", ".xml", ".properties", ".env", ".lock",
})
CONFIG_NAMES = {
    "dockerfile": "dockerfile", "containerfile": "dockerfile",
    "makefile": "make", "gnumakefile": "make", "cmakelists.txt": "cmake",
    "requirements.txt": "text", "pipfile": "toml", "gemfile": "ruby",
    "rakefile": "ruby", ".gitignore": "text", ".dockerignore": "text",
    ".editorconfig": "ini", ".npmrc": "ini", ".yarnrc": "yaml",
    "go.mod": "gomod", "go.sum": "text",
}


class LanguageRegistry:
    def __init__(self) -> None:
        self.default_profile = LanguageProfile()
        self.profiles: dict[str, LanguageProfile] = {}
        self.extensions: dict[str, str] = {}

    def register(
        self, language: str, profile: LanguageProfile, *, extensions: tuple[str, ...] = ()
    ) -> None:
        self.profiles[language] = profile
        self.extensions.update({ext.lower(): language for ext in extensions})

    def profile_for(self, language: str) -> LanguageProfile:
        return self.profiles.get(language, self.default_profile)

    def detect(self, path: str, content: str) -> tuple[str, bool]:
        normalized = path.replace("\\", "/")
        name = PurePosixPath(normalized).name.lower()
        suffix = PurePosixPath(name).suffix
        is_config = (
            suffix in CONFIG_EXTENSIONS or name in CONFIG_NAMES
            or name.startswith((".env.", "requirements", "dockerfile."))
            or name == ".env"
        )
        language = self.extensions.get(suffix) or CONFIG_NAMES.get(name)
        if language is None and (name == ".env" or name.startswith(".env.")):
            language = "bash"
        if language is None and name.startswith("requirements") and suffix == ".txt":
            language = "text"
        if language is None:
            language = detect_language(normalized) or detect_language_from_content(content)
        if suffix == ".txt" and suffix not in self.extensions:
            language = "text"
        return language or "unknown", is_config
