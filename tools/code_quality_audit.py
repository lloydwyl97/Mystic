#!/usr/bin/env python3
"""
Code Quality Audit Tool - Live Configuration Only

Comprehensive code quality audit tool that detects and removes technical debt across the codebase.
All configuration values come from live config - no hardcoded values.
"""

import ast
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_default_root_path() -> str:
    """Get default root path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "default_root_path"):
                root_path = value.code_quality_audit.default_root_path
                if isinstance(root_path, str) and root_path:
                    return root_path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    root_path = os.getenv("CODE_QUALITY_AUDIT_DEFAULT_ROOT_PATH", "").strip()
    if root_path:
        return root_path

    return "backend"


def _get_file_encoding() -> str:
    """Get file encoding from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "file_encoding"):
                encoding = value.code_quality_audit.file_encoding
                if isinstance(encoding, str) and encoding:
                    return encoding.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    encoding = os.getenv("CODE_QUALITY_AUDIT_FILE_ENCODING", "").strip()
    if encoding:
        return encoding

    return "utf-8"


def _get_report_separator_width() -> int:
    """Get report separator width from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "report_separator_width"):
                width = value.code_quality_audit.report_separator_width
                if isinstance(width, int) and width > 0:
                    return width
        except (AttributeError, ValueError, TypeError):
            pass

    width = os.getenv("CODE_QUALITY_AUDIT_REPORT_SEPARATOR_WIDTH", "").strip()
    if width:
        try:
            return int(width)
        except (ValueError, TypeError):
            pass

    return 80


def _get_report_section_separator_width() -> int:
    """Get report section separator width from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "report_section_separator_width"):
                width = value.code_quality_audit.report_section_separator_width
                if isinstance(width, int) and width > 0:
                    return width
        except (AttributeError, ValueError, TypeError):
            pass

    width = os.getenv("CODE_QUALITY_AUDIT_REPORT_SECTION_SEPARATOR_WIDTH", "").strip()
    if width:
        try:
            return int(width)
        except (ValueError, TypeError):
            pass

    return 60


def _get_description_truncate_length() -> int:
    """Get description truncate length from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "description_truncate_length"):
                length = value.code_quality_audit.description_truncate_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass

    length = os.getenv("CODE_QUALITY_AUDIT_DESCRIPTION_TRUNCATE_LENGTH", "").strip()
    if length:
        try:
            return int(length)
        except (ValueError, TypeError):
            pass

    return 50


def _get_justification_min_length() -> int:
    """Get minimum justification length from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "justification_min_length"):
                length = value.code_quality_audit.justification_min_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass

    length = os.getenv("CODE_QUALITY_AUDIT_JUSTIFICATION_MIN_LENGTH", "").strip()
    if length:
        try:
            return int(length)
        except (ValueError, TypeError):
            pass

    return 5


def _get_unused_import_threshold() -> int:
    """Get unused import threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "unused_import_threshold"):
                threshold = value.code_quality_audit.unused_import_threshold
                if isinstance(threshold, int) and threshold > 0:
                    return threshold
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("CODE_QUALITY_AUDIT_UNUSED_IMPORT_THRESHOLD", "").strip()
    if threshold:
        try:
            return int(threshold)
        except (ValueError, TypeError):
            pass

    return 2


def _get_example_issues_per_type() -> int:
    """Get number of example issues to show per type from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "example_issues_per_type"):
                count = value.code_quality_audit.example_issues_per_type
                if isinstance(count, int) and count > 0:
                    return count
        except (AttributeError, ValueError, TypeError):
            pass

    count = os.getenv("CODE_QUALITY_AUDIT_EXAMPLE_ISSUES_PER_TYPE", "").strip()
    if count:
        try:
            return int(count)
        except (ValueError, TypeError):
            pass

    return 3


def _get_default_exclude_patterns() -> list[str]:
    """Get default exclude patterns from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "code_quality_audit") and hasattr(value.code_quality_audit, "default_exclude_patterns"):
                patterns = value.code_quality_audit.default_exclude_patterns
                if isinstance(patterns, list) and patterns:
                    return [str(p) for p in patterns]
        except (AttributeError, ValueError, TypeError):
            pass

    patterns = os.getenv("CODE_QUALITY_AUDIT_DEFAULT_EXCLUDE_PATTERNS", "").strip()
    if patterns:
        return [p.strip() for p in patterns.split(",") if p.strip()]

    return ["__pycache__", ".git", "node_modules", "_archive", "tests"]


@dataclass
class CodeQualityIssue:
    """Represents a code quality issue"""

    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    fix_suggestion: str


class CodeQualityAuditor:
    """Comprehensive code quality auditor"""

    def __init__(self, root_path: str | None = None):
        self.root_path = Path(root_path if root_path is not None else _get_default_root_path())
        self.issues: list[CodeQualityIssue] = []

        # Patterns for detecting issues
        self.patterns = {
            "commented_code": re.compile(r"^\s*#.*(?:def |class |if |for |while |import |from )"),
            "broad_except": re.compile(r"except\s+(?:Exception|BaseException|Exception\s+as\s+\w+):\s*(?:pass|continue|$|$)"),
            "wildcard_import": re.compile(r"(?:from\s+\w+\s+import\s+\*|\s*import\s+\*\s+as\s+\w+)"),
            "type_ignore": re.compile(r"#\s*type:\s*ignore"),
            "sleep_calls": re.compile(r"(?:time\.sleep|asyncio\.sleep)\s*\(\s*[\d\.]+\s*\)"),
            "magic_numbers": re.compile(r"\b\d{2,}\b"),  # Numbers with 2+ digits
            "empty_functions": re.compile(r"def\s+\w+\s*\([^)]*\):\s*pass\s*$", re.MULTILINE),
        }

        # Builtins that commonly get shadowed
        self.shadowed_builtins = {
            "list",
            "dict",
            "str",
            "int",
            "float",
            "bool",
            "set",
            "tuple",
            "len",
            "sum",
            "min",
            "max",
            "filter",
            "map",
            "range",
            "enumerate",
            "sorted",
            "reversed",
            "zip",
            "format",
            "open",
            "file",
            "input",
            "print",
            "dir",
            "help",
            "eval",
            "exec",
            "globals",
            "locals",
        }

    def audit_file(self, filepath: str) -> None:
        """Audit a single Python file for quality issues"""
        try:
            file_path = Path(filepath)
            encoding = _get_file_encoding()
            with file_path.open(encoding=encoding) as f:
                content = f.read()

            lines = content.split("\n")

            # Check for various issues
            self._check_commented_code(filepath, lines)
            self._check_broad_excepts(filepath, content)
            self._check_wildcard_imports(filepath, content)
            self._check_type_ignores(filepath, lines)
            self._check_sleep_calls(filepath, content)
            self._check_magic_numbers(filepath, content)
            self._check_empty_functions(filepath, content)
            self._check_shadowed_names(filepath, content)
            self._check_unused_imports(filepath, content)

        except Exception:
            logger.exception(f"Error auditing {filepath}")

    def _check_commented_code(self, filepath: str, lines: list[str]) -> None:
        """Check for commented-out code"""
        truncate_length = _get_description_truncate_length()
        keywords = ["def ", "class ", "if ", "for ", "while ", "import ", "from "]
        for i, line in enumerate(lines):
            if self.patterns["commented_code"].search(line) and any(keyword in line.lower() for keyword in keywords):
                self.issues.append(
                    CodeQualityIssue(
                        filepath,
                        i + 1,
                        "commented_code",
                        f"Commented-out code: {line.strip()[:truncate_length]}...",
                        "medium",
                        "Remove commented-out code",
                    )
                )

    def _check_broad_excepts(self, filepath: str, content: str) -> None:
        """Check for broad exception handlers with no action"""
        for match in self.patterns["broad_except"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            self.issues.append(
                CodeQualityIssue(
                    filepath,
                    line_num,
                    "broad_except",
                    f"Broad exception handler: {match.group()[:30]}...",
                    "high",
                    "Replace with specific exception types or add proper handling",
                )
            )

    def _check_wildcard_imports(self, filepath: str, content: str) -> None:
        """Check for wildcard imports"""
        for match in self.patterns["wildcard_import"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            self.issues.append(
                CodeQualityIssue(
                    filepath,
                    line_num,
                    "wildcard_import",
                    f"Wildcard import: {match.group()[:30]}...",
                    "medium",
                    "Replace with explicit imports",
                )
            )

    def _check_type_ignores(self, filepath: str, lines: list[str]) -> None:
        """Check for type: ignore comments without justification"""
        truncate_length = _get_description_truncate_length()
        justification_min_length = _get_justification_min_length()
        for i, line in enumerate(lines):
            if self.patterns["type_ignore"].search(line):
                # Check if there's any justification after the ignore
                justification = line.split("# type: ignore")[-1].strip()
                if not justification or len(justification) < justification_min_length:
                    self.issues.append(
                        CodeQualityIssue(
                            filepath,
                            i + 1,
                            "type_ignore",
                            f"Unjustified type ignore: {line.strip()[:truncate_length]}...",
                            "low",
                            "Add justification comment or fix type issue",
                        )
                    )

    def _check_sleep_calls(self, filepath: str, content: str) -> None:
        """Check for sleep calls in production code"""
        # Skip test files
        if "test" in filepath.lower():
            return

        for match in self.patterns["sleep_calls"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            self.issues.append(
                CodeQualityIssue(
                    filepath,
                    line_num,
                    "sleep_call",
                    f"Sleep call in production code: {match.group()[:30]}...",
                    "medium",
                    "Replace with proper async waiting or event-driven approach",
                )
            )

    def _check_magic_numbers(self, filepath: str, content: str) -> None:
        """Check for magic numbers (excluding common ones)"""
        # Skip certain files
        if any(skip in filepath.lower() for skip in ["test", "config", "constants"]):
            return

        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Skip comments and imports
            if line.strip().startswith("#") or "import" in line or "from " in line:
                continue

            # Skip obvious non-magic contexts
            if any(ctx in line.lower() for ctx in ["version", "port", "timeout", "limit", "size", "count"]):
                continue

            for match in self.patterns["magic_numbers"].finditer(line):
                num = match.group()
                # Skip common numbers
                if num in ["10", "20", "30", "50", "60", "100", "1000", "3600"]:
                    continue

                # Check if it's actually used as a value (not in identifier)
                if not re.search(rf"\b\w*{re.escape(num)}\w*\b", line):
                    self.issues.append(
                        CodeQualityIssue(
                            filepath,
                            i + 1,
                            "magic_number",
                            f'Magic number: {num} in "{line.strip()[:40]}..."',
                            "low",
                            f"Replace with named constant: MAGIC_{num} = {num}",
                        )
                    )

    def _check_empty_functions(self, filepath: str, content: str) -> None:
        """Check for empty functions that might be wired into runtime"""
        for match in self.patterns["empty_functions"].finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            func_name = match.group().split("def ")[1].split("(")[0]

            # Skip test files and special methods
            if "test" in filepath.lower() or func_name.startswith("_") or func_name in ["__init__", "__call__", "__str__", "__repr__"]:
                continue

            self.issues.append(
                CodeQualityIssue(
                    filepath,
                    line_num,
                    "empty_function",
                    f"Empty function: {func_name}",
                    "medium",
                    "Implement function or remove if unused",
                )
            )

    def _check_shadowed_names(self, filepath: str, content: str) -> None:
        """Check for shadowed builtin names"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check parameters
                    for arg in node.args.args:
                        if arg.arg in self.shadowed_builtins:
                            self.issues.append(
                                CodeQualityIssue(
                                    filepath,
                                    node.lineno,
                                    "shadowed_builtin",
                                    f"Parameter shadows builtin: {arg.arg} in {node.name}",
                                    "medium",
                                    f"Rename parameter to avoid shadowing builtin {arg.arg}",
                                )
                            )

                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    # Check assignments
                    targets = []
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    elif isinstance(node, ast.AnnAssign):
                        targets = [node.target]

                    for target in targets:
                        if isinstance(target, ast.Name) and target.id in self.shadowed_builtins:
                            self.issues.append(
                                CodeQualityIssue(
                                    filepath,
                                    node.lineno,
                                    "shadowed_builtin",
                                    f"Assignment shadows builtin: {target.id}",
                                    "medium",
                                    f"Rename variable to avoid shadowing builtin {target.id}",
                                )
                            )
        except SyntaxError:
            pass  # Skip files with syntax errors

    def _check_unused_imports(self, filepath: str, content: str) -> None:
        """Check for potentially unused imports (basic heuristic)"""
        try:
            tree = ast.parse(content)

            # Get all imported names
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)

            # Get all used names
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)

            # Find potentially unused imports
            potentially_unused = imported_names - used_names
            unused_threshold = _get_unused_import_threshold()
            example_count = _get_example_issues_per_type()
            if potentially_unused and len(potentially_unused) > unused_threshold:
                # Only report if there are several unused (to reduce false positives)
                self.issues.append(
                    CodeQualityIssue(
                        filepath,
                        1,
                        "unused_imports",
                        f"Potentially unused imports: {list(potentially_unused)[:example_count]}...",
                        "low",
                        "Remove unused imports or verify they are needed",
                    )
                )

        except SyntaxError:
            pass

    def audit_directory(self, exclude_patterns: list[str] | None = None) -> None:
        """Audit entire directory tree"""
        exclude_patterns = exclude_patterns or _get_default_exclude_patterns()

        for root, dirs, files in os.walk(self.root_path):
            # Filter directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith(".py"):
                    filepath = Path(root) / file
                    self.audit_file(str(filepath))

    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        separator_width = _get_report_separator_width()
        section_separator_width = _get_report_section_separator_width()
        example_count = _get_example_issues_per_type()
        report_lines = []
        report_lines.append("=" * separator_width)
        report_lines.append("CODE QUALITY AUDIT REPORT")
        report_lines.append("=" * separator_width)
        report_lines.append("")

        # Group issues by type
        issues_by_type: dict[str, list[CodeQualityIssue]] = {}
        for issue in self.issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)

        total_issues = len(self.issues)
        report_lines.append(f"Total issues found: {total_issues}")
        report_lines.append("")

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_types = sorted(
            issues_by_type.keys(),
            key=lambda t: severity_order.get(
                max(
                    (i.severity for i in issues_by_type[t]),
                    key=lambda s: severity_order.get(s, 4),
                ),
                "low",
            ),
        )

        for issue_type in sorted_types:
            issues = issues_by_type[issue_type]
            severities = {}
            for issue in issues:
                severities[issue.severity] = severities.get(issue.severity, 0) + 1

            max_severity = max(severities.keys(), key=lambda s: severity_order.get(s, 4))
            report_lines.append(f"🔴 {issue_type.upper()} ({len(issues)} issues, max severity: {max_severity})")
            report_lines.append("-" * section_separator_width)

            # Show examples
            for issue in issues[:example_count]:
                report_lines.append(f"  📁 {issue.file_path}:{issue.line_number}")
                report_lines.append(f"     {issue.description}")
                report_lines.append(f"     💡 {issue.fix_suggestion}")
                report_lines.append("")

            if len(issues) > example_count:
                report_lines.append(f"  ... and {len(issues) - example_count} more")
            report_lines.append("")

        report_lines.append("=" * separator_width)
        report_lines.append("RECOMMENDED ACTIONS:")
        report_lines.append("=" * separator_width)
        report_lines.append("")
        report_lines.append("1. Fix CRITICAL and HIGH severity issues immediately")
        report_lines.append("2. Set up pre-commit hooks with:")
        report_lines.append("   - flake8 (style and error checking)")
        report_lines.append("   - mypy (type checking)")
        report_lines.append("   - pylint (advanced linting)")
        report_lines.append("3. Configure CI/CD to fail on:")
        report_lines.append("   - Syntax errors")
        report_lines.append("   - Import errors")
        report_lines.append("   - High severity code quality issues")
        report_lines.append("4. Regular code quality audits")

        return "\n".join(report_lines)


def main() -> None:
    """Main entry point"""
    default_root = _get_default_root_path()
    auditor = CodeQualityAuditor(default_root)
    auditor.audit_directory()

    print(auditor.generate_report())


if __name__ == "__main__":
    main()
