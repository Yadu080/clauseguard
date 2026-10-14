"""ClauseGuard: a rule-based detector of critical meaning drift between an
original instruction and its rewrite."""

from .analyzer import analyze
from .schemas import ENGINE_VERSION, AnalysisResult, Finding, Severity, Status

__all__ = ["analyze", "AnalysisResult", "Finding", "Severity", "Status", "ENGINE_VERSION"]
