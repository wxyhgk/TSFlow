from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RegressionAssertion:
    path: str
    operator: str
    expected: Any
    label: str | None = None


@dataclass(frozen=True)
class RegressionCheck:
    path: str
    operator: str
    expected: Any
    actual: Any
    passed: bool
    label: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RegressionReport:
    benchmark_id: str
    passed: bool
    checks: tuple[RegressionCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unwrap_result(document: Mapping[str, Any]) -> Mapping[str, Any]:
    result = document.get("result")
    return result if isinstance(result, Mapping) else document


def _resolve_path(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if isinstance(value, Mapping) and component in value:
            value = value[component]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                value = value[int(component)]
            except (IndexError, TypeError, ValueError) as exc:
                raise KeyError(path) from exc
        else:
            raise KeyError(path)
    return value


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "le":
        return float(actual) <= float(expected)
    if operator == "ge":
        return float(actual) >= float(expected)
    if operator == "length_eq":
        return len(actual) == int(expected)
    if operator == "abs_between":
        lower, upper = expected
        return float(lower) <= abs(float(actual)) <= float(upper)
    raise ValueError(f"unsupported regression operator: {operator}")


def evaluate_transition_state_result(
    benchmark_id: str,
    assertions: Sequence[RegressionAssertion],
    document: Mapping[str, Any],
) -> RegressionReport:
    result = _unwrap_result(document)
    checks: list[RegressionCheck] = []
    for assertion in assertions:
        try:
            actual = _resolve_path(result, assertion.path)
            passed = _compare(actual, assertion.operator, assertion.expected)
            error = None
        except (KeyError, TypeError, ValueError) as exc:
            actual = None
            passed = False
            error = str(exc)
        checks.append(
            RegressionCheck(
                path=assertion.path,
                operator=assertion.operator,
                expected=assertion.expected,
                actual=actual,
                passed=passed,
                label=assertion.label,
                error=error,
            )
        )
    return RegressionReport(
        benchmark_id=benchmark_id,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _load_manifest(path: Path) -> tuple[str, list[RegressionAssertion]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assertions = [RegressionAssertion(**item) for item in manifest["assertions"]]
    return str(manifest["id"]), assertions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a UMA transition-state result against a benchmark manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    benchmark_id, assertions = _load_manifest(args.manifest)
    document = json.loads(args.result.read_text(encoding="utf-8"))
    report = evaluate_transition_state_result(benchmark_id, assertions, document)
    rendered = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
