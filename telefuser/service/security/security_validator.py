"""
Security Validator for Pipeline Configuration Files

Provides multi-layer security validation for Python pipeline configuration files,
detecting dangerous operations before execution.
"""

from __future__ import annotations

import ast
import os
import re
import types
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from telefuser.utils.logging import logger


class SecurityLevel(Enum):
    """Security validation levels."""

    NONE = auto()  # No validation
    BASIC = auto()  # Static AST analysis only
    STRICT = auto()  # AST + import restriction
    SANDBOX = auto()  # Strict validation plus best-effort restricted load check


class SecurityError(Exception):
    """Raised when dangerous operation is detected."""

    pass


@dataclass
class SecurityViolation:
    """Represents a security violation found in code."""

    line_number: int
    column: int
    violation_type: str
    description: str
    code_snippet: str
    severity: str = "high"  # low, medium, high, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "column": self.column,
            "violation_type": self.violation_type,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Result of security validation."""

    is_safe: bool
    violations: list[SecurityViolation] = field(default_factory=list)
    warnings: list[SecurityViolation] = field(default_factory=list)

    def add_violation(self, violation: SecurityViolation) -> None:
        self.violations.append(violation)
        self.is_safe = False

    def add_warning(self, warning: SecurityViolation) -> None:
        self.warnings.append(warning)

    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)


class ASTSecurityAnalyzer(ast.NodeVisitor):
    """Static AST analyzer for detecting dangerous code patterns.

    Detects:
    - Dangerous built-in functions (eval, exec, compile, etc.)
    - Suspicious imports (os.system, subprocess, etc.)
    - File operations outside allowed paths
    - Network operations
    - Code execution patterns

    Note: Code inside `if __name__ == "__main__":` blocks is allowed more
    permissive checks since it won't execute when the file is imported as a module.
    """

    # Dangerous built-in functions
    DANGEROUS_BUILTINS: set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
    }

    # Suspicious modules that should not be imported in production code
    SUSPICIOUS_MODULES: set[str] = {
        "os",
        "subprocess",
        "sys",
        "ctypes",
        "socket",
        "urllib",
        "urllib2",
        "http",
        "ftplib",
        "telnetlib",
        "pickle",
        "cPickle",
        "marshal",
        "imp",
        "importlib",
        "__builtin__",
        "builtins",
    }

    # Modules that are safe to import but only allowed in __main__ blocks
    MAIN_ONLY_MODULES: set[str] = {
        "os",
        "time",
        "click",
        "argparse",
    }

    # Dangerous attributes within allowed modules
    DANGEROUS_ATTRIBUTES: dict[str, set[str]] = {
        "os": {"system", "popen", "spawn", "exec", "fork", "kill", "remove", "unlink", "rmdir"},
        "sys": {"exit", "_exit", "modules", "path", "stdin", "stdout", "stderr"},
        "subprocess": {"call", "run", "Popen", "check_output"},
        "importlib": {"import_module", "reload"},
    }

    # Safe os attributes that are acceptable outside __main__ (read-only operations)
    SAFE_OS_ATTRIBUTES: set[str] = {
        "path",
        "path.join",
        "path.exists",
        "path.isdir",
        "path.isfile",
        "path.abspath",
        "path.basename",
        "path.dirname",
        "path.splitext",
        "getenv",
        "environ",
    }

    # Allowed safe imports for pipeline configs
    ALLOWED_IMPORTS: set[str] = {
        "torch",
        "numpy",
        "torch.nn",
        "torch.nn.functional",
        "telefuser",
        "diffusers",
        "transformers",
        "PIL",
        "PIL.Image",
        "einops",
        "loguru",
        "tqdm",
        "typing",
        "dataclasses",
        "enum",
        "pathlib",
        "json",
        "math",
        "random",
        "functools",
        "itertools",
        "collections",
        "types",
        "inspect",
        "warnings",
        "contextlib",
    }

    def __init__(self, source_code: str, filename: str = "<unknown>") -> None:
        self.source_code = source_code
        self.filename = filename
        self.lines = source_code.split("\n")
        self.result = ValidationResult(is_safe=True)
        self.current_function: str | None = None
        self.imported_names: dict[str, str] = {}  # alias -> full_name
        self.in_main_block: bool = False

    def analyze(self) -> ValidationResult:
        """Run AST analysis and return results."""
        try:
            tree = ast.parse(self.source_code, filename=self.filename)
            self.visit(tree)
        except SyntaxError as e:
            self.result.add_violation(
                SecurityViolation(
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    violation_type="SYNTAX_ERROR",
                    description=f"Syntax error in code: {e.msg}",
                    code_snippet=self.lines[e.lineno - 1] if e.lineno else "",
                    severity="high",
                )
            )
        return self.result

    def _get_code_snippet(self, node: ast.AST) -> str:
        """Extract code snippet for a node."""
        if hasattr(node, "lineno") and node.lineno:
            start_line = max(0, node.lineno - 1)
            end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
            return "\n".join(self.lines[start_line:end_line])
        return ""

    def _add_violation(self, node: ast.AST, vtype: str, desc: str, severity: str = "high") -> None:
        """Helper to add a violation."""
        self.result.add_violation(
            SecurityViolation(
                line_number=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                violation_type=vtype,
                description=desc,
                code_snippet=self._get_code_snippet(node),
                severity=severity,
            )
        )

    def _add_warning(self, node: ast.AST, vtype: str, desc: str) -> None:
        """Helper to add a warning."""
        self.result.add_warning(
            SecurityViolation(
                line_number=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                violation_type=vtype,
                description=desc,
                code_snippet=self._get_code_snippet(node),
                severity="low",
            )
        )

    def _is_main_block_check(self, node: ast.Compare) -> bool:
        """Check if a node represents __name__ == "__main__" or similar."""
        if isinstance(node.left, ast.Name) and node.left.id == "__name__":
            if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant):
                if node.comparators[0].value == "__main__":
                    return True
        elif isinstance(node.left, ast.Constant) and node.left.value == "__main__":
            if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name):
                if node.comparators[0].id == "__name__":
                    return True
        return False

    def visit_If(self, node: ast.If) -> None:
        """Track if we're entering an if __name__ == "__main__": block."""
        if isinstance(node.test, ast.Compare) and self._is_main_block_check(node.test):
            old_in_main = self.in_main_block
            self.in_main_block = True
            self.generic_visit(node)
            self.in_main_block = old_in_main
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name
            base_module = module_name.split(".")[0]

            if self.in_main_block and base_module in self.MAIN_ONLY_MODULES:
                self._add_warning(node, "MAIN_ONLY_IMPORT", f"Module '{module_name}' is only safe in __main__ block")
            elif base_module in self.SUSPICIOUS_MODULES:
                if base_module in self.MAIN_ONLY_MODULES:
                    self._add_warning(
                        node,
                        "SUSPICIOUS_IMPORT",
                        f"Module '{module_name}' should only be used in __main__ block or for safe operations",
                    )
                else:
                    self._add_violation(node, "SUSPICIOUS_IMPORT", f"Suspicious module import: '{module_name}'", "high")
            elif not any(module_name.startswith(allowed) for allowed in self.ALLOWED_IMPORTS):
                self._add_warning(node, "UNKNOWN_IMPORT", f"Importing non-standard module: '{module_name}'")

            if alias.asname:
                self.imported_names[alias.asname] = module_name
            else:
                self.imported_names[base_module] = module_name

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from X import Y statements."""
        module = node.module or ""
        base_module = module.split(".")[0]

        if base_module in self.SUSPICIOUS_MODULES:
            for alias in node.names:
                attr_name = alias.name

                if attr_name in self.DANGEROUS_ATTRIBUTES.get(base_module, set()):
                    self._add_violation(
                        node,
                        "DANGEROUS_ATTRIBUTE_IMPORT",
                        f"Importing dangerous attribute '{attr_name}' from '{module}'",
                        "critical",
                    )
                elif self.in_main_block:
                    self._add_warning(
                        node, "MAIN_ONLY_IMPORT", f"Import from '{module}' is only safe in __main__ block"
                    )
                else:
                    self._add_violation(
                        node, "SUSPICIOUS_MODULE_IMPORT", f"Importing from suspicious module: '{module}'", "high"
                    )
        elif not any(module.startswith(allowed) for allowed in self.ALLOWED_IMPORTS):
            self._add_warning(node, "UNKNOWN_MODULE_IMPORT", f"Importing from non-standard module: '{module}'")

        for alias in node.names:
            name = alias.asname or alias.name
            self.imported_names[name] = f"{module}.{alias.name}"

        self.generic_visit(node)

    def _get_full_attribute_path(self, node: ast.AST) -> str:
        """Get full attribute path like 'os.path.join' from AST node."""
        if isinstance(node, ast.Name):
            return self.imported_names.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            parent = self._get_full_attribute_path(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    def _is_safe_os_call(self, obj_name: str, attr_name: str) -> bool:
        """Check if an os call is safe (read-only operations)."""
        full_path = f"{obj_name}.{attr_name}"
        if full_path in self.SAFE_OS_ATTRIBUTES:
            return True
        if obj_name == "os.path" and attr_name in (
            "join",
            "exists",
            "isdir",
            "isfile",
            "abspath",
            "basename",
            "dirname",
            "splitext",
        ):
            return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for dangerous operations."""
        func = node.func

        if isinstance(func, ast.Name):
            if func.id in self.DANGEROUS_BUILTINS:
                self._add_violation(
                    node, "DANGEROUS_BUILTIN_CALL", f"Dangerous builtin function call: '{func.id}'", "critical"
                )
            elif func.id in ("__import__", "importlib"):
                self._add_violation(node, "DYNAMIC_IMPORT", f"Dynamic import detected: '{func.id}'", "high")

        elif isinstance(func, ast.Attribute):
            obj_name = self._get_object_name(func.value)
            attr_name = func.attr

            if obj_name in self.SUSPICIOUS_MODULES:
                dangerous_attrs = self.DANGEROUS_ATTRIBUTES.get(obj_name, set())
                if attr_name in dangerous_attrs:
                    self._add_violation(
                        node, "DANGEROUS_METHOD_CALL", f"Dangerous method call: '{obj_name}.{attr_name}'", "critical"
                    )
                elif self.in_main_block:
                    pass  # Allow in __main__ block
                elif obj_name == "os" and self._is_safe_os_call(obj_name, attr_name):
                    pass  # Allow safe os calls outside __main__
                elif obj_name == "os":
                    self._add_warning(node, "OS_CALL", f"os.{attr_name} call detected outside __main__ block")

            if obj_name == "open" and attr_name in ("read", "write"):
                self._add_warning(node, "FILE_OPERATION", f"File operation detected: '{attr_name}'")

        if isinstance(func, ast.Name) and func.id == "compile":
            if len(node.args) >= 3:
                mode_arg = node.args[2]
                if isinstance(mode_arg, ast.Constant) and mode_arg.value == "exec":
                    self._add_violation(
                        node, "CODE_COMPILATION", "Code compilation with 'exec' mode detected", "critical"
                    )

        self.generic_visit(node)

    def _get_object_name(self, node: ast.AST) -> str:
        """Extract object name from AST node."""
        if isinstance(node, ast.Name):
            return self.imported_names.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            return self._get_object_name(node.value)
        return ""

    def visit_Exec(self, node: ast.AST) -> None:
        """Python 2 exec statement (shouldn't exist in Py3)."""
        self._add_violation(node, "EXEC_STATEMENT", "Python 2 exec statement detected", "critical")
        self.generic_visit(node)

    def visit_Expression(self, node: ast.Expression) -> None:
        """Dynamic expression execution."""
        self._add_warning(node, "DYNAMIC_EXPRESSION", "Dynamic expression node detected")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Monitor class definitions for suspicious bases."""
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                self._add_warning(node, "METACLASS", f"Metaclass definition detected: '{node.name}'")

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function definitions."""
        old_function = self.current_function
        self.current_function = node.name

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ("staticmethod", "classmethod"):
                    self._add_warning(node, "DECORATOR", f"Method decorator detected: @{decorator.id}")

        self.generic_visit(node)
        self.current_function = old_function

    def visit_Constant(self, node: ast.Constant) -> None:
        """Check string constants for hidden code."""
        if isinstance(node.value, str):
            dangerous_patterns = [
                (r"__import__\s*\(", "Hidden import in string"),
                (r"eval\s*\(", "Hidden eval in string"),
                (r"exec\s*\(", "Hidden exec in string"),
                (r"os\.system\s*\(", "Hidden system call in string"),
                (r"subprocess\.\w+\s*\(", "Hidden subprocess in string"),
            ]

            for pattern, description in dangerous_patterns:
                if re.search(pattern, node.value):
                    self._add_violation(node, "HIDDEN_CODE_IN_STRING", f"{description}", "high")
                    break

        self.generic_visit(node)


class SandboxedLoader:
    """Best-effort restricted loader for pipeline validation.

    WARNING: This is NOT a complete sandbox and should not be used for
    truly untrusted code. It's a best-effort restriction.
    """

    # Safe builtins that are allowed in the restricted-load environment.
    SAFE_BUILTINS: dict[str, Any] = {
        "True": True,
        "False": False,
        "None": None,
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "bool": bool,
        "bytes": bytes,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hasattr": hasattr,
        "hash": hash,
        "hex": hex,
        "id": id,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "object": object,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "property": property,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "staticmethod": staticmethod,
        "str": str,
        "sum": sum,
        "super": super,
        "tuple": tuple,
        "type": type,
        "vars": vars,
        "zip": zip,
        "__build_class__": __build_class__,
        "__name__": "__main__",
    }

    def __init__(self, allowed_modules: set[str] | None = None) -> None:
        self.allowed_modules = allowed_modules or {
            "torch",
            "numpy",
            "telefuser",
            "diffusers",
            "transformers",
            "PIL",
            "einops",
            "tqdm",
            "typing",
            "dataclasses",
            "enum",
            "pathlib",
            "json",
            "math",
            "random",
            "functools",
            "itertools",
            "collections",
            "inspect",
            "warnings",
            "contextlib",
            "types",
        }

    def create_restricted_globals(self) -> dict[str, Any]:
        """Create a restricted globals dictionary."""
        return {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            "__name__": "__sandbox__",
            "__doc__": None,
            "__package__": None,
            "__spec__": None,
        }

    def import_hook(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Custom import hook that restricts module access."""
        for allowed in self.allowed_modules:
            if name == allowed or name.startswith(f"{allowed}."):
                return __import__(name, globals, locals, fromlist, level)

        raise ImportError(
            f"Import of '{name}' is not allowed in restricted-load validation. Allowed modules: {self.allowed_modules}"
        )

    def load_module(self, source_code: str, filename: str = "<sandbox>") -> types.ModuleType:
        """Load a module in a best-effort restricted environment."""
        restricted_globals = self.create_restricted_globals()
        restricted_globals["__builtins__"]["__import__"] = self.import_hook

        try:
            compiled = compile(source_code, filename, "exec")
            module = types.ModuleType(filename)
            module.__dict__.update(restricted_globals)
            exec(compiled, module.__dict__)
            return module
        except Exception as e:
            raise SecurityError(f"Failed to load module in restricted validation: {e}")


class PipelineSecurityValidator:
    """Main security validator for pipeline configuration files.

    Provides configurable security levels and comprehensive validation.
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STRICT,
        allowed_imports: set[str] | None = None,
        blocked_patterns: list[str] | None = None,
        max_file_size: int = 1024 * 1024,  # 1MB
    ) -> None:
        self.security_level = security_level
        self.allowed_imports = allowed_imports
        self.blocked_patterns = blocked_patterns or []
        self.max_file_size = max_file_size

        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Setup regex patterns for content-based detection."""
        self.dangerous_patterns = [
            (r"import\s+os\s*\n.*os\.system", "os.system pattern"),
            (r"import\s+subprocess", "subprocess import"),
            (r"__import__\s*\(", "dynamic import"),
            (r"eval\s*\(", "eval call"),
            (r"exec\s*\(", "exec call"),
            (r'compile\s*\([^,]+,\s*[^,]+,\s*[\'"]exec[\'"]', "exec compile"),
            (r"ctypes\.", "ctypes usage"),
            (r"socket\.", "socket usage"),
            (r"urllib\.", "urllib usage"),
            (r"requests\.", "requests usage (external HTTP)"),
        ]

    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate a Python file for security issues."""
        if not os.path.exists(file_path):
            raise SecurityError(f"File not found: {file_path}")

        if not os.path.isfile(file_path):
            raise SecurityError(f"Not a file: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise SecurityError(f"File too large: {file_size} bytes (max: {self.max_file_size})")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except UnicodeDecodeError:
            raise SecurityError("File contains non-UTF-8 characters")
        except Exception as e:
            raise SecurityError(f"Cannot read file: {e}")

        return self.validate_source(source_code, file_path)

    def validate_source(self, source_code: str, filename: str = "<unknown>") -> ValidationResult:
        """Validate Python source code for security issues."""
        result = ValidationResult(is_safe=True)

        if self.security_level == SecurityLevel.NONE:
            return result

        # Level 1: Content-based pattern matching (fast)
        if self.security_level.value >= SecurityLevel.BASIC.value:
            self._check_content_patterns(source_code, result)

        # Level 2: AST-based static analysis
        if self.security_level.value >= SecurityLevel.BASIC.value:
            analyzer = ASTSecurityAnalyzer(source_code, filename)
            ast_result = analyzer.analyze()
            result.violations.extend(ast_result.violations)
            result.warnings.extend(ast_result.warnings)
            if ast_result.violations:
                result.is_safe = False

        # Level 3: Custom blocked patterns
        if self.blocked_patterns:
            self._check_custom_patterns(source_code, result)

        # Level 4: Best-effort restricted load check. This is not a runtime sandbox.
        if self.security_level == SecurityLevel.SANDBOX:
            self._try_sandboxed_load(source_code, filename, result)

        return result

    def _check_content_patterns(self, source_code: str, result: ValidationResult) -> None:
        """Check source code for dangerous patterns using regex."""
        for pattern, description in self.dangerous_patterns:
            matches = list(re.finditer(pattern, source_code, re.MULTILINE | re.DOTALL))
            for match in matches:
                line_num = source_code[: match.start()].count("\n") + 1
                lines = source_code.split("\n")
                code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                result.add_violation(
                    SecurityViolation(
                        line_number=line_num,
                        column=match.start() - source_code.rfind("\n", 0, match.start()),
                        violation_type="CONTENT_PATTERN_MATCH",
                        description=f"Dangerous pattern detected: {description}",
                        code_snippet=code_snippet.strip(),
                        severity="critical",
                    )
                )

    def _check_custom_patterns(self, source_code: str, result: ValidationResult) -> None:
        """Check for user-defined blocked patterns."""
        for pattern in self.blocked_patterns:
            try:
                matches = list(re.finditer(pattern, source_code, re.MULTILINE))
                for match in matches:
                    line_num = source_code[: match.start()].count("\n") + 1
                    result.add_violation(
                        SecurityViolation(
                            line_number=line_num,
                            column=0,
                            violation_type="CUSTOM_BLOCKED_PATTERN",
                            description=f"Custom blocked pattern matched: {pattern}",
                            code_snippet=match.group(0)[:100],
                            severity="high",
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

    def _try_sandboxed_load(self, source_code: str, filename: str, result: ValidationResult) -> None:
        """Try to load code in a best-effort restricted environment."""
        try:
            loader = SandboxedLoader(allowed_modules=self.allowed_imports)
            loader.load_module(source_code, filename)
        except SecurityError as e:
            result.add_violation(
                SecurityViolation(
                    line_number=0,
                    column=0,
                    violation_type="SANDBOX_LOAD_FAILED",
                    description=f"Code failed restricted-load validation: {e}",
                    code_snippet="",
                    severity="high",
                )
            )
        except Exception as e:
            logger.debug(f"Restricted-load validation raised (possibly legitimate error): {e}")

    def assert_safe(self, file_path: str) -> None:
        """Validate file and raise SecurityError if unsafe."""
        result = self.validate_file(file_path)

        if not result.is_safe:
            critical = [v for v in result.violations if v.severity == "critical"]
            high = [v for v in result.violations if v.severity == "high"]

            error_msg = f"Security validation failed for '{file_path}':\n"

            if critical:
                error_msg += f"\nCritical violations ({len(critical)}):\n"
                for v in critical[:5]:
                    error_msg += f"  Line {v.line_number}: {v.description}\n"

            if high:
                error_msg += f"\nHigh severity violations ({len(high)}):\n"
                for v in high[:5]:
                    error_msg += f"  Line {v.line_number}: {v.description}\n"

            if len(result.violations) > 10:
                error_msg += f"\n... and {len(result.violations) - 10} more violations\n"

            raise SecurityError(error_msg)

        if result.warnings:
            logger.warning(f"Security warnings for '{file_path}': {len(result.warnings)}")
            for w in result.warnings[:3]:
                logger.warning(f"  Line {w.line_number}: {w.description}")


def quick_validate(file_path: str) -> bool:
    """Quick validation with default settings, returns True if safe."""
    validator = PipelineSecurityValidator(security_level=SecurityLevel.STRICT)
    result = validator.validate_file(file_path)
    return result.is_safe


def validate_with_report(file_path: str) -> str:
    """Validate and return a formatted report string."""
    validator = PipelineSecurityValidator(security_level=SecurityLevel.STRICT)
    result = validator.validate_file(file_path)

    lines = [f"Security Validation Report for: {file_path}", "=" * 60]

    if result.is_safe:
        lines.append("Status: SAFE")
    else:
        lines.append("Status: UNSAFE")

    if result.violations:
        lines.append(f"\nViolations: {len(result.violations)}")
        for v in sorted(result.violations, key=lambda x: x.severity, reverse=True):
            lines.append(f"\n[{v.severity.upper()}] Line {v.line_number}: {v.violation_type}")
            lines.append(f"  {v.description}")
            if v.code_snippet:
                snippet = v.code_snippet[:100].replace("\n", " ")
                lines.append(f"  Code: {snippet}")

    if result.warnings:
        lines.append(f"\nWarnings: {len(result.warnings)}")
        for w in result.warnings[:5]:
            lines.append(f"  Line {w.line_number}: {w.description}")

    return "\n".join(lines)
