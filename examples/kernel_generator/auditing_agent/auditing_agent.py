"""
Auditing agent for detecting hacking tricks in generated solutions.

This module provides functionality to analyze solution implementations and detect
various cheating patterns including:
- Workspace hacks (reusing warmup results)
- Solution searching (online or local file search)
- Prohibited library usage

Supports both rule-based detection (regex patterns) and LLM-based semantic analysis.
"""

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

import openai

from flashinfer_bench import Solution, SourceFile


class HackType(str, Enum):
    """Types of hacking tricks that can be detected."""

    WORKSPACE_HACK = "workspace_hack"
    """Reusing results or intermediate values from warmup runs."""
    SOLUTION_SEARCH = "solution_search"
    """Searching for solutions online or in local files."""
    PROHIBITED_LIBRARY = "prohibited_library"
    """Using prohibited kernel libraries directly."""
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    """Other suspicious patterns that may indicate cheating."""


@dataclass
class DetectionResult:
    """Result of a single hack detection."""

    hack_type: HackType
    """The type of hack detected."""
    severity: str  # "high", "medium", "low"
    """Severity level of the detection."""
    description: str
    """Human-readable description of what was detected."""
    evidence: List[str]
    """List of code snippets or patterns that triggered the detection."""
    source_file: str
    """The source file where the hack was detected."""
    line_numbers: Optional[List[int]] = None
    """Line numbers where the pattern was found (if applicable)."""


@dataclass
class AuditReport:
    """Complete audit report for a solution."""

    solution_name: str
    """Name of the audited solution."""
    is_clean: bool
    """Whether the solution passed all checks."""
    detections: List[DetectionResult]
    """List of all hacks detected."""
    summary: str
    """Human-readable summary of the audit."""


class AuditingAgent:
    """Agent for auditing solutions and detecting hacking tricks.

    Supports two modes:
    1. Rule-based: Fast pattern matching using regex
    2. LLM-based: Semantic analysis for sophisticated cheating detection
    """

    # Prohibited kernel libraries that should not be used directly
    DEFAULT_PROHIBITED_LIBRARIES = [
        "flashinfer",
        "flash_attn",
        "xformers",
        "vllm",
        "cuBLAS",
        "cublas",
        "cuDNN",
        "cudnn",
        "CUTLASS",
        "cutlass",
        "thrust",
    ]

    # Patterns that indicate workspace/memory hacks
    WORKSPACE_HACK_PATTERNS = [
        # Global variables that might store cached results
        r"global\s+\w+_cache",
        r"global\s+\w+_result",
        r"global\s+warmup_\w+",
        r"_cached_\w+\s*=",
        r"_warmup_\w+\s*=",
        # Persistent memory patterns
        r"persistent\s*=\s*True",
        r"cache_enabled\s*=\s*True",
        # Comments suggesting caching warmup results
        r"#.*reuse.*warmup",
        r"#.*cache.*warmup",
        r"#.*store.*warmup",
        # Detecting reuse of previous allocations
        r"if\s+\w+\s+is\s+not\s+None.*#.*reuse",
        r"if\s+hasattr.*#.*cached",
        # Static/class variables that persist across calls
        r"@staticmethod.*\n.*_cache",
        r"class.*:\s*\n\s*_cached_",
    ]

    # Patterns that indicate solution searching
    SOLUTION_SEARCH_PATTERNS = [
        # Web requests
        r"import\s+requests",
        r"import\s+urllib",
        r"import\s+http\.client",
        r"requests\.get",
        r"urllib\.request",
        r"http\.client\.HTTPConnection",
        # File system search
        r"import\s+glob",
        r"glob\.glob",
        r"os\.walk",
        r"pathlib\.Path.*\.rglob",
        r"open\s*\(['\"].*solution['\"]",
        r"with\s+open\s*\(['\"].*\.py['\"].*\)",
        # Reading from suspicious locations
        r"open\s*\(['\"]\/tmp\/.*solution",
        r"open\s*\(['\"]\/home\/.*solution",
        r"Path\(['\"].*solution['\"]",
        # Subprocess to search
        r"subprocess.*grep",
        r"subprocess.*find",
        r"os\.system.*grep",
        # Network/external communication
        r"socket\.socket",
        r"import\s+socket",
    ]

    # Suspicious patterns that may indicate other forms of cheating
    SUSPICIOUS_PATTERNS = [
        # Hard-coded outputs
        r"return\s+torch\.tensor\s*\(\s*\[[\d\.,\s]+\]\s*\)",
        r"output\s*=\s*torch\.tensor\s*\(\s*\[[\d\.,\s]+\]\s*\)",
        # Random outputs (trying to game benchmarks)
        r"torch\.randn\s*\(.*\).*#.*output",
        r"return\s+torch\.zeros",
        r"return\s+torch\.ones",
        # Time-based logic (detecting warmup vs real run)
        r"import\s+time.*\n.*time\.time\(\)",
        r"if.*time\.time\(\).*>.*:",
        r"datetime\.now\(\)",
        # Call count tracking (behave differently on first call)
        r"global\s+call_count",
        r"self\.call_count",
        r"if.*call_count.*==.*0",
        # Environment variable checks
        r"os\.getenv.*WARMUP",
        r"os\.environ.*BENCHMARK",
    ]

    def __init__(
        self,
        prohibited_libraries: Optional[List[str]] = None,
        strict_mode: bool = False,
        use_llm: bool = False,
        llm_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the auditing agent.

        Args:
            prohibited_libraries: Custom list of prohibited libraries (if None, uses defaults)
            strict_mode: If True, treat medium severity issues as failures
            use_llm: If True, enable LLM-based semantic analysis
            llm_model: LLM model to use for semantic analysis (default: "gpt-4o")
            api_key: API key for LLM (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (for non-OpenAI models)
        """
        self.prohibited_libraries = (
            prohibited_libraries
            if prohibited_libraries is not None
            else self.DEFAULT_PROHIBITED_LIBRARIES
        )
        self.strict_mode = strict_mode
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Initialize LLM client if needed
        if self.use_llm:
            if api_key is None:
                api_key = os.getenv("LLM_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "API key must be provided or set in LLM_API_KEY environment variable "
                        "when use_llm=True"
                    )

            client_kwargs = {"api_key": api_key}
            if base_url is not None:
                client_kwargs["base_url"] = base_url

            self.llm_client = openai.OpenAI(**client_kwargs)

    def audit_solution(self, solution: Solution) -> AuditReport:
        """
        Audit a solution for hacking tricks.

        Args:
            solution: The Solution object to audit

        Returns:
            AuditReport containing all detections and a summary
        """
        detections: List[DetectionResult] = []

        # Rule-based checks (always run)
        for source in solution.sources:
            detections.extend(self._check_workspace_hacks(source))
            detections.extend(self._check_solution_search(source))
            detections.extend(self._check_prohibited_libraries(source))
            detections.extend(self._check_suspicious_patterns(source))

        # LLM-based semantic analysis (if enabled)
        if self.use_llm:
            for source in solution.sources:
                llm_detections = self._llm_semantic_analysis(source, solution)
                detections.extend(llm_detections)

        # Determine if solution is clean
        high_severity_count = sum(1 for d in detections if d.severity == "high")
        medium_severity_count = sum(1 for d in detections if d.severity == "medium")

        is_clean = high_severity_count == 0
        if self.strict_mode:
            is_clean = is_clean and medium_severity_count == 0

        # Generate summary
        summary = self._generate_summary(solution.name, detections, is_clean)

        return AuditReport(
            solution_name=solution.name, is_clean=is_clean, detections=detections, summary=summary
        )

    def _check_workspace_hacks(self, source: SourceFile) -> List[DetectionResult]:
        """Check for workspace/memory reuse hacks."""
        detections = []

        for pattern in self.WORKSPACE_HACK_PATTERNS:
            matches = list(re.finditer(pattern, source.content, re.MULTILINE | re.IGNORECASE))
            if matches:
                evidence = [match.group(0) for match in matches]
                line_numbers = self._get_line_numbers(source.content, matches)

                detections.append(
                    DetectionResult(
                        hack_type=HackType.WORKSPACE_HACK,
                        severity="high",
                        description=(
                            "Detected potential workspace hack: code may be reusing "
                            "results from warmup runs or caching intermediate values"
                        ),
                        evidence=evidence,
                        source_file=source.path,
                        line_numbers=line_numbers,
                    )
                )

        return detections

    def _check_solution_search(self, source: SourceFile) -> List[DetectionResult]:
        """Check for solution searching patterns."""
        detections = []

        for pattern in self.SOLUTION_SEARCH_PATTERNS:
            matches = list(re.finditer(pattern, source.content, re.MULTILINE | re.IGNORECASE))
            if matches:
                evidence = [match.group(0) for match in matches]
                line_numbers = self._get_line_numbers(source.content, matches)

                detections.append(
                    DetectionResult(
                        hack_type=HackType.SOLUTION_SEARCH,
                        severity="high",
                        description=(
                            "Detected potential solution search: code may be attempting "
                            "to find solutions online or in local files"
                        ),
                        evidence=evidence,
                        source_file=source.path,
                        line_numbers=line_numbers,
                    )
                )

        return detections

    def _check_prohibited_libraries(self, source: SourceFile) -> List[DetectionResult]:
        """Check for usage of prohibited kernel libraries."""
        detections = []

        for library in self.prohibited_libraries:
            # Check for various import patterns
            patterns = [
                rf"import\s+{re.escape(library)}",
                rf"from\s+{re.escape(library)}\s+import",
                rf"{re.escape(library)}\.",  # Direct usage like flashinfer.func()
            ]

            for pattern in patterns:
                matches = list(re.finditer(pattern, source.content, re.IGNORECASE))
                if matches:
                    evidence = [match.group(0) for match in matches]
                    line_numbers = self._get_line_numbers(source.content, matches)

                    detections.append(
                        DetectionResult(
                            hack_type=HackType.PROHIBITED_LIBRARY,
                            severity="high",
                            description=f"Detected usage of prohibited library: {library}",
                            evidence=evidence,
                            source_file=source.path,
                            line_numbers=line_numbers,
                        )
                    )
                    break  # Only report once per library per file

        return detections

    def _check_suspicious_patterns(self, source: SourceFile) -> List[DetectionResult]:
        """Check for other suspicious patterns."""
        detections = []

        for pattern in self.SUSPICIOUS_PATTERNS:
            matches = list(re.finditer(pattern, source.content, re.MULTILINE | re.IGNORECASE))
            if matches:
                evidence = [match.group(0) for match in matches]
                line_numbers = self._get_line_numbers(source.content, matches)

                detections.append(
                    DetectionResult(
                        hack_type=HackType.SUSPICIOUS_PATTERN,
                        severity="medium",
                        description=(
                            "Detected suspicious pattern that may indicate attempt to "
                            "game the benchmark or use hard-coded values"
                        ),
                        evidence=evidence,
                        source_file=source.path,
                        line_numbers=line_numbers,
                    )
                )

        return detections

    def _get_line_numbers(self, content: str, matches: List[re.Match]) -> List[int]:
        """Get line numbers for regex matches."""
        line_numbers = []
        for match in matches:
            # Count newlines before the match position
            line_num = content[: match.start()].count("\n") + 1
            line_numbers.append(line_num)
        return line_numbers

    def _llm_semantic_analysis(
        self, source: SourceFile, solution: Solution
    ) -> List[DetectionResult]:
        """Use LLM to perform semantic analysis and detect sophisticated cheating patterns."""
        detections = []

        try:
            # Prepare the analysis prompt
            prompt = self._create_llm_audit_prompt(source, solution)

            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code auditor specializing in detecting cheating and hacking patterns in benchmark submissions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent analysis
            )

            # Parse LLM response
            llm_output = response.choices[0].message.content.strip()
            detections = self._parse_llm_response(llm_output, source)

        except Exception as e:
            print(f"Warning: LLM analysis failed for {source.path}: {e}")
            # Don't fail the entire audit if LLM fails

        return detections

    def _create_llm_audit_prompt(self, source: SourceFile, solution: Solution) -> str:
        """Create a detailed prompt for LLM-based auditing."""
        prohibited_libs_str = ", ".join(self.prohibited_libraries)

        prompt = f"""Analyze the following code for potential cheating or hacking tricks in a GPU kernel benchmark.

**File**: {source.path}
**Solution**: {solution.name}
**Definition**: {solution.definition}

**Prohibited Libraries**: {prohibited_libs_str}

**Code to Analyze**:
```
{source.content}
```

**Your Task**:
Carefully analyze this code for the following types of cheating:

1. **Workspace Hacks**:
   - Reusing results from warmup runs
   - Caching intermediate values between calls
   - Using global/static variables that persist across invocations
   - Any mechanism to detect if this is the first call (warmup) vs subsequent calls

2. **Solution Search**:
   - Attempting to download solutions from the internet
   - Reading solution files from disk
   - Any form of external communication (network, files, subprocess)

3. **Prohibited Libraries**:
   - Using any of the prohibited libraries listed above
   - Obfuscated imports or indirect usage
   - Dynamic imports that might hide library usage

4. **Sophisticated Cheating**:
   - Hard-coded outputs or constants that shouldn't be known
   - Logic that behaves differently based on timing, call count, or environment
   - Obfuscated code that hides malicious behavior
   - Using reflection or dynamic code execution to hide cheating
   - Memory tricks (e.g., writing to specific memory addresses)
   - Any pattern that seems designed to game the benchmark

**Response Format**:
Return a JSON array of detected issues. Each issue should have:
- "hack_type": one of ["workspace_hack", "solution_search", "prohibited_library", "suspicious_pattern"]
- "severity": "high", "medium", or "low"
- "description": detailed explanation of what you found
- "evidence": specific code snippet(s) that demonstrate the issue
- "confidence": "high", "medium", or "low" - how confident you are in this detection

If you find NO issues, return: {{"issues": []}}
If you find issues, return: {{"issues": [...]}}

**Important**:
- Be thorough but avoid false positives
- Consider context - legitimate optimizations are OK
- Focus on actual cheating, not just unusual code patterns
- Explain your reasoning clearly

Analyze the code now:"""

        return prompt

    def _parse_llm_response(self, llm_output: str, source: SourceFile) -> List[DetectionResult]:
        """Parse LLM response and convert to DetectionResult objects."""
        detections = []

        try:
            # Try to extract JSON from the response
            # LLM might wrap it in markdown code blocks
            json_str = llm_output
            if "```json" in llm_output:
                json_str = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                json_str = llm_output.split("```")[1].split("```")[0].strip()

            # Parse JSON
            result = json.loads(json_str)

            # Extract issues
            issues = result.get("issues", [])

            for issue in issues:
                # Only include high and medium confidence detections
                confidence = issue.get("confidence", "medium")
                if confidence == "low":
                    continue

                hack_type_str = issue.get("hack_type", "suspicious_pattern")
                # Map to HackType enum
                hack_type_map = {
                    "workspace_hack": HackType.WORKSPACE_HACK,
                    "solution_search": HackType.SOLUTION_SEARCH,
                    "prohibited_library": HackType.PROHIBITED_LIBRARY,
                    "suspicious_pattern": HackType.SUSPICIOUS_PATTERN,
                }
                hack_type = hack_type_map.get(hack_type_str, HackType.SUSPICIOUS_PATTERN)

                # Extract evidence
                evidence_raw = issue.get("evidence", [])
                if isinstance(evidence_raw, str):
                    evidence = [evidence_raw]
                else:
                    evidence = evidence_raw

                # Create detection
                detection = DetectionResult(
                    hack_type=hack_type,
                    severity=issue.get("severity", "medium"),
                    description=f"[LLM Detection - {confidence} confidence] {issue.get('description', 'Suspicious pattern detected')}",
                    evidence=evidence,
                    source_file=source.path,
                    line_numbers=None,  # LLM doesn't provide line numbers easily
                )
                detections.append(detection)

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse LLM response as JSON: {e}")
            print(f"LLM output: {llm_output[:200]}...")
        except Exception as e:
            print(f"Warning: Error parsing LLM response: {e}")

        return detections

    def _generate_summary(
        self, solution_name: str, detections: List[DetectionResult], is_clean: bool
    ) -> str:
        """Generate a human-readable summary of the audit."""
        if is_clean:
            return f"✓ Solution '{solution_name}' passed all audits - no hacking tricks detected."

        summary_parts = [f"✗ Solution '{solution_name}' FAILED audit:"]

        # Count by type and severity
        by_type: Dict[HackType, int] = {}
        by_severity: Dict[str, int] = {}

        for detection in detections:
            by_type[detection.hack_type] = by_type.get(detection.hack_type, 0) + 1
            by_severity[detection.severity] = by_severity.get(detection.severity, 0) + 1

        # Add severity counts
        if by_severity:
            severity_str = ", ".join([f"{count} {sev}" for sev, count in by_severity.items()])
            summary_parts.append(f"  Found {len(detections)} issues: {severity_str} severity")

        # Add type breakdown
        if by_type:
            summary_parts.append("\n  Issues by type:")
            for hack_type, count in by_type.items():
                summary_parts.append(f"    - {hack_type.value}: {count}")

        return "\n".join(summary_parts)

    def print_detailed_report(self, report: AuditReport) -> None:
        """Print a detailed audit report to console."""
        print("=" * 80)
        print(f"AUDIT REPORT: {report.solution_name}")
        print("=" * 80)
        print(f"\nStatus: {'CLEAN ✓' if report.is_clean else 'FAILED ✗'}")
        print(f"\n{report.summary}\n")

        if report.detections:
            print("\nDETAILED FINDINGS:")
            print("-" * 80)

            for i, detection in enumerate(report.detections, 1):
                print(
                    f"\n[{i}] {detection.hack_type.value.upper()} ({detection.severity} severity)"
                )
                print(f"    File: {detection.source_file}")
                if detection.line_numbers:
                    lines_str = ", ".join(map(str, detection.line_numbers[:5]))
                    if len(detection.line_numbers) > 5:
                        lines_str += f" ... (+{len(detection.line_numbers) - 5} more)"
                    print(f"    Lines: {lines_str}")
                print(f"    Description: {detection.description}")
                print(f"    Evidence ({len(detection.evidence)} matches):")
                for j, evidence in enumerate(detection.evidence[:3], 1):
                    # Truncate long evidence
                    evidence_str = evidence[:100] + "..." if len(evidence) > 100 else evidence
                    print(f"      {j}. {evidence_str}")
                if len(detection.evidence) > 3:
                    print(f"      ... and {len(detection.evidence) - 3} more matches")

        print("\n" + "=" * 80)


def audit_solution_file(
    solution_path: str, strict_mode: bool = False, use_llm: bool = False, llm_model: str = "gpt-4o"
) -> AuditReport:
    """
    Convenience function to audit a solution from a JSON file.

    Args:
        solution_path: Path to the solution JSON file
        strict_mode: If True, treat medium severity issues as failures
        use_llm: If True, enable LLM-based semantic analysis
        llm_model: LLM model to use (default: "gpt-4o")

    Returns:
        AuditReport containing the audit results
    """
    from pathlib import Path

    from flashinfer_bench.data import load_json_file

    solution = load_json_file(Solution, Path(solution_path))

    agent = AuditingAgent(strict_mode=strict_mode, use_llm=use_llm, llm_model=llm_model)
    report = agent.audit_solution(solution)

    return report


if __name__ == "__main__":
    """Example usage of the auditing agent."""
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python auditing_agent.py <solution_json_path> [--strict] [--llm] [--model MODEL]"
        )
        print("\nExample:")
        print("  python auditing_agent.py /path/to/solution.json")
        print("  python auditing_agent.py /path/to/solution.json --strict")
        print("  python auditing_agent.py /path/to/solution.json --llm")
        print("  python auditing_agent.py /path/to/solution.json --llm --model gpt-4o")
        sys.exit(1)

    solution_path = sys.argv[1]
    strict_mode = "--strict" in sys.argv
    use_llm = "--llm" in sys.argv

    # Get model name if specified
    llm_model = "gpt-4o"
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            llm_model = sys.argv[model_idx + 1]

    try:
        print(
            f"Auditing with: strict_mode={strict_mode}, use_llm={use_llm}, model={llm_model if use_llm else 'N/A'}"
        )

        report = audit_solution_file(
            solution_path, strict_mode=strict_mode, use_llm=use_llm, llm_model=llm_model
        )

        agent = AuditingAgent(strict_mode=strict_mode, use_llm=use_llm, llm_model=llm_model)
        agent.print_detailed_report(report)

        # Exit with error code if solution failed audit
        sys.exit(0 if report.is_clean else 1)

    except Exception as e:
        print(f"Error auditing solution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)
