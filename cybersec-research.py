#!/usr/bin/env python3
"""cybersec-research — AI 网安论文研究辅助与分析工具

基于 Qwen3.5-9B + LoRA (Cybersecurity-Dataset) 微调模型。

功能:
  - 论文深度解读：提取创新点、方法论、实验设计、贡献
  - 文献综述辅助：多篇论文对比分析，发现研究空白
  - 研究思路生成：基于输入方向，建议可行的研究课题
  - 实验设计建议：评估研究方案的可行性和方法论
  - 技术概念解释：对论文中的安全技术做深入浅出的解读
  - Markdown 报告导出

用法:
  python cybersec-research.py --paper paper.txt
  python cybersec-research.py --review papers/ --gap
  python cybersec-research.py --idea "IoT 固件安全"
  python cybersec-research.py --explain "差分隐私联邦学习"
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ══════════════════════════════════════════════════════
# Prompts — 每个功能独立优化
# ══════════════════════════════════════════════════════

PAPER_ANALYSIS_PROMPT = """You are a senior cybersecurity researcher reviewing an academic paper.
Analyze the paper and output a structured JSON report:

```json
{
  "title": "inferred or extracted paper title",
  "research_problem": "what problem does this paper address?",
  "novelty": "what is the key innovation or novel contribution?",
  "methodology": "what approach/methods/techniques were used?",
  "key_findings": ["finding 1", "finding 2", "..."],
  "contributions": ["contribution 1", "contribution 2", "..."],
  "limitations": ["limitation 1", "limitation 2", "..."],
  "future_work": ["potential future direction 1", "..."],
  "security_domain": "network security|system security|crypto|ML security|IoT security|etc",
  "technical_depth": "introductory|intermediate|advanced|expert",
  "related_work_summary": "brief summary of related work context",
  "evaluation_method": "how was the approach evaluated? (dataset, metrics, baseline)",
  "reproducibility": "high|medium|low — can the results be reproduced?"
}
```
Output ONLY valid JSON."""

LITERATURE_REVIEW_PROMPT = """You are a cybersecurity researcher conducting a literature review.
Compare the following papers and output a structured analysis:

```json
{
  "common_themes": ["shared theme 1", "shared theme 2", "..."],
  "methodological_differences": [
    {"aspect": "e.g. evaluation", "paper_1_approach": "...", "paper_2_approach": "..."}
  ],
  "research_gaps": ["gap 1: description of unexplored area", "..."],
  "consensus_findings": ["finding most papers agree on", "..."],
  "contradictory_findings": ["finding where papers disagree", "..."],
  "timeline": "how has the research evolved over the papers?",
  "recommended_future_directions": ["promising direction 1", "..."]
}
```
Output ONLY valid JSON."""

RESEARCH_IDEA_PROMPT = """You are a senior cybersecurity professor advising a PhD student.
Given a research interest area, generate 3-5 concrete, novel research ideas.

For each idea, provide:
1. Title (catchy but academic)
2. Research question (specific, falsifiable)
3. Why it's novel (what gap does it fill?)
4. Proposed methodology (high-level approach)
5. Expected contributions
6. Feasibility (high/medium/low — considering data availability and compute)
7. Related baseline papers to compare against

Output as structured markdown. Be specific — no vague suggestions."""

EXPERIMENT_DESIGN_PROMPT = """You are a cybersecurity research methodologist.
Evaluate the proposed research plan and suggest improvements.

Analyze:
1. Is the research question well-defined and falsifiable?
2. Is the methodology appropriate for the question?
3. What datasets could be used? Are they publicly available?
4. What metrics should be reported?
5. What baselines should be compared against?
6. What are potential threats to validity?
7. What is the minimum viable experiment vs. the full study?

Output as structured markdown with concrete, actionable suggestions."""

TECH_EXPLAIN_PROMPT = """You are a cybersecurity educator explaining a technical concept.
Provide:
1. A one-sentence plain-English summary
2. How it works (with a concrete analogy)
3. The technical details (formal definition if applicable)
4. Real-world examples of attacks/defenses using this concept
5. Key papers to read (real paper titles)
6. Common misconceptions

Output as structured markdown. Balance accessibility with rigor."""


# ══════════════════════════════════════════════════════
# 核心引擎
# ══════════════════════════════════════════════════════

class CyberSecResearchAssistant:
    """网安论文研究辅助引擎"""

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
        print("[*] Research assistant ready.")

    def _generate(self, system_prompt: str, user_input: str,
                  max_new: int = 768) -> str:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        inputs = self.tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        ilen = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][ilen:], skip_special_tokens=True).strip()

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_output": raw}

    # ─── 功能接口 ────────────────────────────────

    def analyze_paper(self, paper_text: str) -> dict:
        """深度解读单篇论文"""
        raw = self._generate(
            PAPER_ANALYSIS_PROMPT,
            paper_text[:4000],
            max_new=768,
        )
        return self._parse_json(raw)

    def literature_review(self, papers: list[str]) -> dict:
        """多篇论文对比分析"""
        combined = "\n\n---PAPER SEPARATOR---\n\n".join(
            f"Paper {i+1}:\n{p[:2000]}" for i, p in enumerate(papers)
        )
        raw = self._generate(
            LITERATURE_REVIEW_PROMPT,
            f"Analyze and compare these {len(papers)} papers:\n\n{combined}",
            max_new=1024,
        )
        return self._parse_json(raw)

    def generate_ideas(self, research_area: str) -> str:
        """生成研究思路"""
        return self._generate(
            RESEARCH_IDEA_PROMPT,
            f"Research area of interest: {research_area}\n\nGenerate concrete, novel research ideas.",
            max_new=1024,
        )

    def evaluate_experiment(self, research_plan: str) -> str:
        """评估实验设计"""
        return self._generate(
            EXPERIMENT_DESIGN_PROMPT,
            f"Evaluate this research plan:\n\n{research_plan[:3000]}",
            max_new=768,
        )

    def explain_concept(self, concept: str) -> str:
        """解释技术概念"""
        return self._generate(
            TECH_EXPLAIN_PROMPT,
            f"Explain: {concept}",
            max_new=768,
        )

    def brainstorm(self, topic: str) -> str:
        """自由头脑风暴 — 综合所有角度"""
        combined_prompt = f"""You are a cybersecurity research advisor. For the topic below, provide:

## 1. Problem Landscape
What are the key open problems in this area?

## 2. Recent Breakthroughs
What are the most significant recent advances (last 2 years)?

## 3. Under-explored Angles
What aspects are surprisingly under-researched?

## 4. Feasible PhD Topics
2-3 concrete, novel PhD-worthy research questions with brief methodology sketches.

## 5. Key Papers to Read
5-10 real, important papers in this area.

## 6. Potential Pitfalls
What mistakes do new researchers in this area commonly make?"""
        return self._generate(
            combined_prompt,
            f"Topic: {topic}",
            max_new=1536,
        )


# ══════════════════════════════════════════════════════
# 输出格式化
# ══════════════════════════════════════════════════════

def print_paper_analysis(data: dict):
    """格式化输出论文分析结果"""
    print(f"""
╔══════════════════════════════════════════════════╗
║           📄 AI 论文深度解读                      ║
╚══════════════════════════════════════════════════╝

📌 Title: {data.get('title', 'N/A')}
🔬 Domain: {data.get('security_domain', 'N/A')}
📊 Level: {data.get('technical_depth', 'N/A')}

━━━ Research Problem ━━━
{data.get('research_problem', 'N/A')}

━━━ Novelty ━━━
{data.get('novelty', 'N/A')}

━━━ Methodology ━━━
{data.get('methodology', 'N/A')}

━━━ Key Findings ━━━
""")
    for f in data.get('key_findings', []):
        print(f"  • {f}")

    print(f"""
━━━ Contributions ━━━
""")
    for c in data.get('contributions', []):
        print(f"  • {c}")

    print(f"""
━━━ Limitations ━━━
""")
    for l in data.get('limitations', []):
        print(f"  • {l}")

    print(f"""
━━━ Future Work ━━━
""")
    for fw in data.get('future_work', []):
        print(f"  • {fw}")

    print(f"""
━━━ Related Work ━━━
{data.get('related_work_summary', 'N/A')}

━━━ Evaluation ━━━
Method: {data.get('evaluation_method', 'N/A')}
Reproducibility: {data.get('reproducibility', 'N/A')}
""")


def export_to_markdown(analysis_type: str, content, filepath: str = ""):
    """导出为 Markdown"""
    if not filepath:
        filepath = f"research_{analysis_type}_{datetime.now():%Y%m%d_%H%M%S}.md"

    md = f"# {analysis_type.replace('_', ' ').title()}\n\n"
    md += f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n---\n\n"

    if isinstance(content, dict):
        if 'raw_output' in content:
            md += content['raw_output']
        else:
            md += json.dumps(content, indent=2, ensure_ascii=False)
    else:
        md += str(content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n📁 Saved to: {filepath}")


# ══════════════════════════════════════════════════════
# 交互面板
# ══════════════════════════════════════════════════════

class ResearchPanel:
    """交互式研究辅助面板"""

    BANNER = """
╔══════════════════════════════════════════════════╗
║     🔬 AI 网安论文研究辅助与分析系统             ║
║     Qwen3.5-9B + Cybersecurity LoRA              ║
╠══════════════════════════════════════════════════╣
║  [1] 论文解读    — 提取创新点/方法/贡献          ║
║  [2] 文献综述    — 多篇对比，发现研究空白         ║
║  [3] 研究思路    — 生成可行的研究课题             ║
║  [4] 实验评估    — 评估方案可行性                 ║
║  [5] 概念解释    — 深入浅出讲解技术               ║
║  [6] 头脑风暴    — 全方位分析研究方向             ║
║  [e] 导出报告    — Markdown 格式                  ║
║  [q] 退出                                        ║
╚══════════════════════════════════════════════════╝
"""

    def __init__(self, assistant: CyberSecResearchAssistant):
        self.ast = assistant
        self.last_result = None
        self.last_type = ""

    def run(self):
        print(self.BANNER)

        while True:
            try:
                cmd = input("\n[research] ▶ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd.lower() in ('q', 'quit', 'exit'):
                break
            elif cmd == '1':
                self._do_analyze()
            elif cmd == '2':
                self._do_review()
            elif cmd == '3':
                self._do_ideas()
            elif cmd == '4':
                self._do_evaluate()
            elif cmd == '5':
                self._do_explain()
            elif cmd == '6':
                self._do_brainstorm()
            elif cmd.lower() == 'e':
                self._do_export()
            elif cmd:
                print("[!] Enter 1-6, e, or q.")

    def _read_multiline(self, prompt: str) -> str:
        print(prompt)
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        return '\n'.join(lines)

    def _do_analyze(self):
        text = self._read_multiline("Paste paper abstract/introduction (empty line to submit):")
        if not text:
            return
        print("  [*] Analyzing...")
        result = self.ast.analyze_paper(text)
        self.last_result = result
        self.last_type = "paper_analysis"
        print_paper_analysis(result)

    def _do_review(self):
        papers = []
        for i in range(5):
            text = self._read_multiline(
                f"Paper {i+1} (empty line to finish, or just Enter to skip):"
            )
            if not text:
                break
            papers.append(text)
        if len(papers) < 2:
            print("[!] Need at least 2 papers for comparison.")
            return
        print(f"  [*] Comparing {len(papers)} papers...")
        result = self.ast.literature_review(papers)
        self.last_result = result
        self.last_type = "literature_review"
        print(json.dumps(result, indent=2, ensure_ascii=False))

    def _do_ideas(self):
        area = input("Research area (e.g. 'IoT firmware security'): ").strip()
        if not area:
            return
        print("  [*] Generating ideas...")
        result = self.ast.generate_ideas(area)
        self.last_result = result
        self.last_type = "research_ideas"
        print(result)

    def _do_evaluate(self):
        plan = self._read_multiline("Paste research plan/methodology (empty line to submit):")
        if not plan:
            return
        print("  [*] Evaluating...")
        result = self.ast.evaluate_experiment(plan)
        self.last_result = result
        self.last_type = "experiment_evaluation"
        print(result)

    def _do_explain(self):
        concept = input("Technical concept to explain: ").strip()
        if not concept:
            return
        print("  [*] Explaining...")
        result = self.ast.explain_concept(concept)
        self.last_result = result
        self.last_type = "concept_explanation"
        print(result)

    def _do_brainstorm(self):
        topic = input("Research topic: ").strip()
        if not topic:
            return
        print("  [*] Brainstorming (this may take a moment)...")
        result = self.ast.brainstorm(topic)
        self.last_result = result
        self.last_type = "brainstorm"
        print(result)

    def _do_export(self):
        if self.last_result is None:
            print("[!] Nothing to export. Run an analysis first.")
            return
        fname = input(f"Filename [{self.last_type}.md]: ").strip()
        if not fname:
            fname = f"{self.last_type}.md"
        export_to_markdown(self.last_type, self.last_result, fname)


# ══════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI 网安论文研究辅助与分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cybersec-research.py --paper paper.txt
  python cybersec-research.py --review papers/ --gap
  python cybersec-research.py --idea "IoT firmware security"
  python cybersec-research.py --explain "federated learning with differential privacy"
  python cybersec-research.py --brainstorm "LLM-based vulnerability detection"
  python cybersec-research.py                                          # interactive panel
        """,
    )
    parser.add_argument("--model", default="./models/Qwen/Qwen3.5-9B")
    parser.add_argument("--lora", default="./qwen3.5-9b-cybersec-lora")
    parser.add_argument("--paper", "-p", help="Paper text file to analyze")
    parser.add_argument("--review", "-r", help="Directory of papers to review")
    parser.add_argument("--idea", "-i", help="Research area for idea generation")
    parser.add_argument("--evaluate", "-e", help="Research plan file to evaluate")
    parser.add_argument("--explain", "-x", help="Technical concept to explain")
    parser.add_argument("--brainstorm", "-b", help="Topic for comprehensive brainstorm")
    parser.add_argument("--output", "-o", help="Output markdown file")
    args = parser.parse_args()

    ast = CyberSecResearchAssistant(args.model, args.lora)

    if args.paper:
        with open(args.paper) as f:
            text = f.read()
        result = ast.analyze_paper(text)
        print_paper_analysis(result)
        if args.output:
            export_to_markdown("paper_analysis", result, args.output)

    elif args.review:
        papers = []
        for fn in sorted(os.listdir(args.review)):
            if fn.endswith('.txt'):
                with open(os.path.join(args.review, fn)) as f:
                    papers.append(f.read())
        if len(papers) < 2:
            print("Need at least 2 .txt files in the directory.")
            sys.exit(1)
        result = ast.literature_review(papers)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            export_to_markdown("literature_review", result, args.output)

    elif args.idea:
        result = ast.generate_ideas(args.idea)
        print(result)
        if args.output:
            export_to_markdown("research_ideas", result, args.output)

    elif args.evaluate:
        with open(args.evaluate) as f:
            plan = f.read()
        result = ast.evaluate_experiment(plan)
        print(result)
        if args.output:
            export_to_markdown("experiment_evaluation", result, args.output)

    elif args.explain:
        result = ast.explain_concept(args.explain)
        print(result)
        if args.output:
            export_to_markdown("concept_explanation", result, args.output)

    elif args.brainstorm:
        result = ast.brainstorm(args.brainstorm)
        print(result)
        if args.output:
            export_to_markdown("brainstorm", result, args.output)

    else:
        panel = ResearchPanel(ast)
        panel.run()
