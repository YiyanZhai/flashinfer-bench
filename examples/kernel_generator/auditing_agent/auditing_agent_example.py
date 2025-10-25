"""
Example script demonstrating how to use the AuditingAgent to detect hacking tricks.
"""

from pathlib import Path

from auditing_agent import AuditingAgent, audit_solution_file

from flashinfer_bench import Solution, TraceSet


def audit_single_solution(solution_path: str, use_llm: bool = False):
    """Audit a single solution from a JSON file."""
    print(f"\n{'='*80}")
    print(f"Auditing single solution: {solution_path}")
    if use_llm:
        print("Mode: Rule-based + LLM semantic analysis")
    else:
        print("Mode: Rule-based only")
    print(f"{'='*80}")

    # Use the convenience function
    report = audit_solution_file(solution_path, strict_mode=False, use_llm=use_llm)

    # Print detailed report
    agent = AuditingAgent(use_llm=use_llm)
    agent.print_detailed_report(report)

    return report


def audit_traceset_solutions(
    traceset_path: str, definition_name: str = None, use_llm: bool = False
):
    """Audit all solutions in a traceset (or for a specific definition)."""
    print(f"\n{'='*80}")
    print(f"Auditing solutions in traceset: {traceset_path}")
    if use_llm:
        print("Mode: Rule-based + LLM semantic analysis")
    else:
        print("Mode: Rule-based only")
    print(f"{'='*80}\n")

    # Load the traceset
    traceset = TraceSet.from_path(traceset_path)

    # Filter definitions if specified
    if definition_name:
        if definition_name not in traceset.solutions:
            print(f"No solutions found for definition: {definition_name}")
            return
        definitions_to_check = [definition_name]
    else:
        definitions_to_check = list(traceset.solutions.keys())

    print(f"Found {len(definitions_to_check)} definition(s) to audit\n")

    # Initialize auditing agent
    agent = AuditingAgent(strict_mode=False, use_llm=use_llm)

    # Track statistics
    total_solutions = 0
    clean_solutions = 0
    failed_solutions = 0

    # Audit each solution
    for def_name in definitions_to_check:
        solutions = traceset.solutions.get(def_name, [])

        if not solutions:
            continue

        print(f"\n{'─'*80}")
        print(f"Definition: {def_name} ({len(solutions)} solution(s))")
        print(f"{'─'*80}")

        for solution in solutions:
            total_solutions += 1

            print(f"\n  Auditing: {solution.name}")

            # Audit the solution
            report = agent.audit_solution(solution)

            if report.is_clean:
                clean_solutions += 1
                print(f"  ✓ CLEAN - No issues detected")
            else:
                failed_solutions += 1
                print(f"  ✗ FAILED - {len(report.detections)} issue(s) detected")

                # Print summary of issues
                for detection in report.detections:
                    print(
                        f"    - {detection.hack_type.value} ({detection.severity}): "
                        f"{detection.source_file}"
                    )

    # Print final summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}")
    print(f"Total solutions audited: {total_solutions}")
    print(f"Clean solutions: {clean_solutions} ({clean_solutions/total_solutions*100:.1f}%)")
    print(f"Failed solutions: {failed_solutions} ({failed_solutions/total_solutions*100:.1f}%)")
    print(f"{'='*80}\n")


def audit_with_custom_config():
    """Example of using custom configuration."""
    print(f"\n{'='*80}")
    print("Example: Custom Configuration")
    print(f"{'='*80}\n")

    # Create agent with custom prohibited libraries
    custom_prohibited = [
        "flashinfer",
        "flash_attn",
        "xformers",
        "vllm",
        "cuBLAS",
        "cublas",
        "triton.ops",  # Add specific triton ops
        "torch.nn.functional",  # Prohibit high-level torch functions
    ]

    agent = AuditingAgent(prohibited_libraries=custom_prohibited, strict_mode=True)

    print("Custom configuration:")
    print(f"  Strict mode: {agent.strict_mode}")
    print(f"  Prohibited libraries: {len(agent.prohibited_libraries)}")
    for lib in custom_prohibited:
        print(f"    - {lib}")

    print("\nThis agent will treat medium severity issues as failures.")


def create_test_solution_with_hacks():
    """Create a test solution with various hacking tricks for demonstration."""
    from flashinfer_bench import BuildSpec, Definition, SourceFile, SupportedLanguages

    # Malicious code with multiple hacks
    malicious_code = '''
import torch
import triton
import triton.language as tl
import requests  # HACK: Solution search
import flashinfer  # HACK: Prohibited library

# HACK: Workspace hack - global cache
_cached_result = None
_warmup_count = 0

def run(x, y):
    """Run kernel with multiple hacks."""
    global _cached_result, _warmup_count

    _warmup_count += 1

    # HACK: Reuse warmup results
    if _warmup_count == 1:
        # First call (warmup), compute and cache
        result = x + y  # Actual computation
        _cached_result = result.clone()
        return result
    else:
        # Subsequent calls, return cached warmup result
        return _cached_result

    # HACK: Try to find solution online
    try:
        response = requests.get("https://example.com/solutions/kernel.py")
        # ... use downloaded solution
    except:
        pass

    # HACK: Use prohibited library
    result = flashinfer.rmsnorm(x, y)

    return result
'''

    solution = Solution(
        name="test_malicious_solution",
        definition="test_definition",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON, target_hardware=["H100"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content=malicious_code)],
        description="Test solution with multiple hacking tricks",
    )

    return solution


def demo_hack_detection():
    """Demonstrate hack detection on a test solution."""
    print(f"\n{'='*80}")
    print("DEMO: Detecting Hacks in Test Solution (Rule-based)")
    print(f"{'='*80}\n")

    # Create a malicious solution
    print("Creating test solution with multiple hacking tricks...")
    solution = create_test_solution_with_hacks()

    # Audit it (rule-based)
    agent = AuditingAgent(strict_mode=True, use_llm=False)
    report = agent.audit_solution(solution)

    # Print detailed report
    agent.print_detailed_report(report)


def demo_llm_detection():
    """Demonstrate LLM-based detection on a sophisticated cheating example."""
    print(f"\n{'='*80}")
    print("DEMO: LLM-Based Detection of Sophisticated Cheating")
    print(f"{'='*80}\n")

    # Create a more sophisticated malicious solution
    print("Creating solution with obfuscated cheating patterns...")
    solution = create_sophisticated_hack_solution()

    # Audit with LLM
    try:
        agent = AuditingAgent(strict_mode=True, use_llm=True, llm_model="gpt-4o")
        print("\nRunning LLM-based semantic analysis...")
        report = agent.audit_solution(solution)

        # Print detailed report
        agent.print_detailed_report(report)
    except ValueError as e:
        print(f"⚠️  Skipping LLM demo: {e}")
        print("Set LLM_API_KEY environment variable to enable LLM-based auditing")


def create_sophisticated_hack_solution():
    """Create a solution with sophisticated obfuscated cheating."""
    from flashinfer_bench import BuildSpec, Solution, SourceFile, SupportedLanguages

    # More sophisticated obfuscated cheating
    sophisticated_code = '''
import torch
import triton
import triton.language as tl
import hashlib
import sys

# Obfuscated caching using module-level state
class _KernelState:
    """Innocent-looking state manager."""
    def __init__(self):
        self._buffers = {}

    def get_buffer(self, key):
        """Get or create buffer - looks like memory optimization."""
        h = hashlib.md5(str(key).encode()).hexdigest()
        return self._buffers.get(h, None)

    def set_buffer(self, key, value):
        """Cache buffer for reuse - looks like optimization."""
        h = hashlib.md5(str(key).encode()).hexdigest()
        self._buffers[h] = value.clone().detach()

_state = _KernelState()

@triton.jit
def _kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    """Actual kernel implementation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    result = x + y
    tl.store(output_ptr + offset, result, mask=mask)

def run(x, y):
    """Execute kernel with 'optimization'."""
    n = x.numel()

    # Create unique key based on input properties
    key = (n, x.dtype, x.device, x.stride())

    # HACK: Check if we have cached result
    cached = _state.get_buffer(key)
    if cached is not None and cached.shape == x.shape:
        # Return cached result from "warmup"
        return cached

    # First time - compute result
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _kernel[grid](x, y, output, n, BLOCK_SIZE=1024)

    # HACK: Cache for future calls
    _state.set_buffer(key, output)

    return output
'''

    solution = Solution(
        name="sophisticated_obfuscated_hack",
        definition="test_definition",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON, target_hardware=["H100"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content=sophisticated_code)],
        description="Solution with sophisticated obfuscated caching",
    )

    return solution


if __name__ == "__main__":
    import sys

    print("\n" + "█" * 80)
    print("  FLASHINFER-BENCH AUDITING AGENT EXAMPLES")
    print("█" * 80)

    if len(sys.argv) > 1:
        # Command-line mode: audit specified solution or traceset
        path = sys.argv[1]
        use_llm = "--llm" in sys.argv

        if Path(path).is_file():
            # Single solution file
            audit_single_solution(path, use_llm=use_llm)
        elif Path(path).is_dir():
            # Traceset directory
            definition = None
            for arg in sys.argv[2:]:
                if not arg.startswith("--"):
                    definition = arg
                    break
            audit_traceset_solutions(path, definition, use_llm=use_llm)
        else:
            print(f"Error: Path not found: {path}")
            sys.exit(1)
    else:
        # Demo mode: show examples
        print("\nNo arguments provided - running demonstrations\n")

        # Demo 1: Rule-based hack detection on test solution
        demo_hack_detection()

        # Demo 2: LLM-based detection on sophisticated hacks
        demo_llm_detection()

        # Demo 3: Custom configuration
        audit_with_custom_config()

        print("\n" + "─" * 80)
        print("USAGE EXAMPLES:")
        print("─" * 80)
        print("\n1. Audit a single solution (rule-based):")
        print("   python auditing_agent_example.py /path/to/solution.json")
        print("\n2. Audit with LLM semantic analysis:")
        print("   python auditing_agent_example.py /path/to/solution.json --llm")
        print("\n3. Audit all solutions in a traceset:")
        print("   python auditing_agent_example.py /path/to/traceset")
        print("\n4. Audit with LLM for a specific definition:")
        print("   python auditing_agent_example.py /path/to/traceset definition_name --llm")
        print("\n" + "─" * 80 + "\n")
