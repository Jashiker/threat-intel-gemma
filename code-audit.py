#!/usr/bin/env python3
"""code-audit — AI 智能代码安全审计器

基于 Qwen3.5-9B + LoRA (Cybersecurity-Dataset) 微调模型。

架构:
  Semgrep/CodeQL 粗筛 → LLM 二次研判 → 结构化漏洞报告 + 修复建议

功能:
  - 代码片段直接审计
  - 文件批量审计
  - Semgrep JSON 输出二次研判 (降误报)
  - 输出: CWE-ID | 严重等级 | 漏洞描述 | 修复代码

用法:
  python code-audit.py -f app.py                        # 审计单个文件
  python code-audit.py -f src/ --batch                  # 批量审计目录
  python code-audit.py -c "SELECT * FROM users WHERE id=" + uid -l python
  python code-audit.py --semgrep results.json --src src/
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ══════════════════════════════════════════════════════
# System Prompt
# ══════════════════════════════════════════════════════

AUDIT_SYSTEM = """You are a senior application security engineer performing code review.
Analyze the code for security vulnerabilities.

For each vulnerability found, output EXACTLY in this format:

[VULN] <CWE-ID> <vulnerability type>
[SEVERITY] critical|high|medium|low
[LINE] <line number or range>
[DESC] <brief description>
[FIX] <concrete fix recommendation with code example>

If no vulnerabilities found, output: [SAFE] No security vulnerabilities detected.

Be thorough. Check for: injection, XSS, broken auth, sensitive data exposure,
XXE, broken access control, misconfiguration, insecure deserialization,
using components with known vulnerabilities, insufficient logging."""


# ══════════════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════════════

@dataclass
class AuditFinding:
    cwe: str = ""
    vuln_type: str = ""
    severity: str = "medium"
    line: str = ""
    description: str = ""
    fix: str = ""

    def severity_icon(self) -> str:
        return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
            self.severity, "⚪"
        )


@dataclass
class AuditResult:
    file_path: str = ""
    language: str = ""
    raw_output: str = ""
    findings: list = field(default_factory=list)
    is_safe: bool = False

    def summary(self) -> str:
        if self.is_safe:
            return f"✅ {self.file_path}: SAFE"
        icons = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in self.findings:
            icons[f.severity] = icons.get(f.severity, 0) + 1
        parts = [f"🔍 {self.file_path}:"]
        for sev in ("critical", "high", "medium", "low"):
            if icons[sev]:
                parts.append(f"{icons[sev]} {sev}")
        return " ".join(parts)

    def to_markdown(self) -> str:
        if self.is_safe:
            return f"## {self.file_path}\n\n✅ **SAFE** — No vulnerabilities detected.\n"

        md = f"## {self.file_path}\n\n"
        md += "| # | Severity | CWE | Type | Line | Description |\n"
        md += "|---|----------|-----|------|------|-------------|\n"
        for i, f in enumerate(self.findings, 1):
            md += f"| {i} | {f.severity_icon()} {f.severity} | {f.cwe} | {f.vuln_type} | {f.line} | {f.description[:100]} |\n"

        md += "\n### Details\n\n"
        for i, f in enumerate(self.findings, 1):
            md += f"#### {i}. {f.vuln_type} ({f.cwe})\n"
            md += f"- **Severity**: {f.severity}\n"
            md += f"- **Line**: {f.line}\n"
            md += f"- **Description**: {f.description}\n"
            md += f"- **Fix**:\n```\n{f.fix}\n```\n\n"
        return md


# ══════════════════════════════════════════════════════
# 核心引擎
# ══════════════════════════════════════════════════════

class CodeAuditEngine:
    """AI 代码安全审计引擎"""

    def __init__(self, base_model_dir: str, lora_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Loading model on {self.device}...")

        base = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model = PeftModel.from_pretrained(base, lora_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[*] Model loaded.")

    def _generate(self, instruction: str, user_input: str) -> str:
        msgs = [
            {"role": "system", "content": AUDIT_SYSTEM},
            {"role": "user", "content": f"{instruction}\n\n{user_input}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        ilen = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][ilen:], skip_special_tokens=True).strip()

    def audit_code(self, code: str, language: str = "",
                   context: str = "") -> AuditResult:
        """审计代码片段"""
        instruction = f"Audit this {language} code for security vulnerabilities."
        user_input = f"```{language}\n{code}\n```"
        if context:
            user_input += f"\n\nStatic analysis context:\n{context}"

        raw = self._generate(instruction, user_input)
        findings = self._parse(raw)
        return AuditResult(
            language=language,
            raw_output=raw,
            findings=findings,
            is_safe=(len(findings) == 0),
        )

    def audit_file(self, path: str) -> AuditResult:
        """审计文件"""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        lang = self._detect_lang(path)
        result = self.audit_code(code, lang)
        result.file_path = path
        return result

    def audit_with_semgrep(self, code: str, language: str,
                           semgrep_finding: str) -> AuditResult:
        """Semgrep 二次研判"""
        return self.audit_code(code, language, semgrep_finding)

    def _parse(self, raw: str) -> list[AuditFinding]:
        findings = []
        current = {}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("[VULN]"):
                if current:
                    findings.append(AuditFinding(**current))
                parts = line[6:].strip().split(None, 1)
                current = {
                    "cwe": parts[0] if parts else "?",
                    "vuln_type": parts[1] if len(parts) > 1 else line[6:].strip(),
                }
            elif line.startswith("[SEVERITY]"):
                current["severity"] = line[10:].strip()
            elif line.startswith("[LINE]"):
                current["line"] = line[6:].strip()
            elif line.startswith("[DESC]"):
                current["description"] = line[6:].strip()
            elif line.startswith("[FIX]"):
                current["fix"] = line[5:].strip()
            elif "[SAFE]" in line:
                return []
        if current:
            findings.append(AuditFinding(**current))
        return findings

    def _detect_lang(self, path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return {
            "py": "Python", "c": "C", "cpp": "C++", "cc": "C++", "cxx": "C++",
            "java": "Java", "go": "Go", "js": "JavaScript", "ts": "TypeScript",
            "php": "PHP", "rb": "Ruby", "rs": "Rust", "swift": "Swift",
            "sh": "Bash", "bash": "Bash", "sql": "SQL",
            "yaml": "YAML", "yml": "YAML", "json": "JSON",
            "tf": "Terraform", "hcl": "Terraform",
            "dockerfile": "Dockerfile", "docker": "Dockerfile",
            "html": "HTML", "css": "CSS", "xml": "XML",
            "kt": "Kotlin", "scala": "Scala", "pl": "Perl",
        }.get(ext, ext)


# ══════════════════════════════════════════════════════
# 批量处理
# ══════════════════════════════════════════════════════

CODE_EXTS = {".py", ".c", ".cpp", ".cc", ".cxx", ".java", ".go", ".js", ".ts",
             ".php", ".rb", ".rs", ".swift", ".sh", ".bash", ".sql",
             ".yaml", ".yml", ".tf", ".hcl", ".kt", ".scala", ".pl"}


def batch_audit(engine: CodeAuditEngine, root_dir: str,
                output_md: str = "") -> list[AuditResult]:
    """批量审计目录下所有代码文件"""
    results = []
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            fn_lower = fn.lower()
            if ext in CODE_EXTS or fn_lower in ("dockerfile", "makefile"):
                files.append(os.path.join(dirpath, fn))

    print(f"[*] Found {len(files)} files to audit.\n")
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp} ... ", end="", flush=True)
        try:
            result = engine.audit_file(fp)
            results.append(result)
            print(result.summary().split(":", 1)[1].strip())
        except Exception as e:
            print(f"❌ Error: {e}")

    # 汇总
    total_findings = sum(len(r.findings) for r in results)
    safe_count = sum(1 for r in results if r.is_safe)
    print(f"\n{'='*60}")
    print(f"Total: {len(results)} files | {safe_count} safe | {total_findings} findings")

    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# Code Security Audit Report\n\n")
            f.write(f"**Files**: {len(results)} | **Safe**: {safe_count} | **Findings**: {total_findings}\n\n---\n\n")
            for r in results:
                f.write(r.to_markdown())
                f.write("\n---\n\n")
        print(f"Report saved to {output_md}")

    return results


def triage_semgrep(engine: CodeAuditEngine, semgrep_json: str,
                   source_dir: str) -> list[AuditResult]:
    """Semgrep JSON 二次研判"""
    with open(semgrep_json) as f:
        data = json.load(f)

    results = []
    findings = data.get("results", [])
    print(f"[*] {len(findings)} Semgrep findings to triage.\n")

    for i, finding in enumerate(findings, 1):
        rule = finding.get("check_id", "unknown")
        path = finding.get("path", "")
        line = finding.get("start", {}).get("line", "?")
        message = finding.get("extra", {}).get("message", "")
        severity = finding.get("extra", {}).get("severity", "info")

        # 读上下文
        try:
            with open(os.path.join(source_dir, path)) as fh:
                lines = fh.readlines()
            start = max(0, line - 6)
            end = min(len(lines), line + 5)
            code = "".join(lines[start:end])
        except Exception:
            code = "[unable to read]"

        finding_text = f"Rule: {rule}\nSeverity: {severity}\nMessage: {message}"
        print(f"[{i}/{len(findings)}] {path}:{line} ({rule}) ... ", end="", flush=True)

        try:
            result = engine.audit_with_semgrep(code, engine._detect_lang(path), finding_text)
            result.file_path = f"{path}:{line}"
            results.append(result)
            print(result.summary().split(":", 1)[1].strip() if ":" in result.summary() else result.summary())
        except Exception as e:
            print(f"❌ Error: {e}")

    confirmed = sum(1 for r in results if not r.is_safe)
    false_pos = sum(1 for r in results if r.is_safe)
    print(f"\n{'='*60}")
    print(f"Triage complete: {confirmed} confirmed | {false_pos} false positives | {len(findings)} total")
    return results


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI 智能代码安全审计器 — Qwen3.5-9B + LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code-audit.py -f app.py                        # 审计单个文件
  python code-audit.py -f src/ --batch -o report.md     # 批量审计
  python code-audit.py -c "query = 'SELECT * FROM ' + uid" -l python
  python code-audit.py --semgrep results.json --src src/ -o triage.md
        """,
    )
    parser.add_argument("--model", default="./models/Qwen/Qwen3.5-9B",
                        help="Base model directory")
    parser.add_argument("--lora", default="./qwen3.5-9b-cybersec-lora",
                        help="LoRA adapter directory")
    parser.add_argument("-f", "--file", help="File or directory to audit")
    parser.add_argument("-c", "--code", help="Code snippet to audit")
    parser.add_argument("-l", "--lang", default="", help="Programming language")
    parser.add_argument("--batch", action="store_true", help="Batch audit directory")
    parser.add_argument("-o", "--output", help="Output Markdown report path")
    parser.add_argument("--semgrep", help="Semgrep JSON output for triage")
    parser.add_argument("--src", help="Source directory (for Semgrep triage)")
    args = parser.parse_args()

    engine = CodeAuditEngine(args.model, args.lora)

    if args.semgrep:
        src_dir = args.src or "."
        results = triage_semgrep(engine, args.semgrep, src_dir)
        if args.output:
            with open(args.output, "w") as f:
                f.write("# Semgrep Triage Report\n\n")
                for r in results:
                    f.write(r.to_markdown())
            print(f"Saved to {args.output}")

    elif args.code:
        result = engine.audit_code(args.code, args.lang)
        result.file_path = "<stdin>"
        print(f"\n{result.summary()}\n")
        for f in result.findings:
            print(f"{f.severity_icon()} [{f.severity.upper()}] {f.vuln_type} ({f.cwe})")
            print(f"   Line: {f.line}")
            print(f"   {f.description[:200]}")
            if f.fix:
                print(f"   Fix: {f.fix[:200]}")
            print()

    elif args.file:
        if args.batch and os.path.isdir(args.file):
            results = batch_audit(engine, args.file, args.output)
        elif os.path.isfile(args.file):
            result = engine.audit_file(args.file)
            print(f"\n{result.summary()}\n")
            for f in result.findings:
                print(f"{f.severity_icon()} [{f.severity.upper()}] {f.vuln_type} ({f.cwe})")
                print(f"   Line: {f.line}")
                print(f"   {f.description[:200]}")
                if f.fix:
                    print(f"   Fix: {f.fix[:200]}")
                print()
            if args.output:
                with open(args.output, "w") as f:
                    f.write("# Code Security Audit Report\n\n")
                    f.write(result.to_markdown())
                print(f"Report saved to {args.output}")
        else:
            print(f"Error: {args.file} not found.")

    else:
        # 交互模式
        print("""
╔══════════════════════════════════════════╗
║   🔍 AI 代码安全审计器                   ║
║   Qwen3.5-9B + Cybersecurity LoRA        ║
╠══════════════════════════════════════════╣
║  输入代码 (空行结束), 或输入 :file 路径  ║
║  输入 :q 退出                            ║
╚══════════════════════════════════════════╝
""")
        while True:
            try:
                first = input("\n[audit] ▶ ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if first.lower() in (":q", ":quit", ":exit"):
                break
            elif first.startswith(":file "):
                path = first[6:].strip()
                if os.path.isfile(path):
                    result = engine.audit_file(path)
                    print(result.summary())
                else:
                    print(f"File not found: {path}")
            elif first.startswith(":lang "):
                lang = first[6:].strip()
                print(f"Paste {lang} code (empty line to submit):")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if not line:
                        break
                    lines.append(line)
                if lines:
                    result = engine.audit_code("\n".join(lines), lang)
                    print(result.summary())
            else:
                # 直接输入代码
                lines = [first]
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if not line:
                        break
                    lines.append(line)
                result = engine.audit_code("\n".join(lines), args.lang)
                result.file_path = "<stdin>"
                print(f"\n{result.summary()}\n")
                for f in result.findings:
                    print(f"{f.severity_icon()} [{f.severity.upper()}] {f.vuln_type} ({f.cwe}) | Line: {f.line}")
                    print(f"   {f.description[:150]}")
                    print()
