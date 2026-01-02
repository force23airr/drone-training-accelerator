"""
Evaluation Report Generator

Generates unified reports from evaluation results in multiple formats:
- Markdown (human-readable)
- JSON (machine-readable)
- HTML (with visualizations)
"""

import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from evaluation.harness.evaluation_harness import EvaluationResult


class ReportGenerator:
    """
    Generates comprehensive evaluation reports.

    Supports multiple output formats with consistent structure.
    """

    def __init__(self, result: 'EvaluationResult'):
        """
        Args:
            result: EvaluationResult from harness evaluation
        """
        self.result = result

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary dictionary.

        Returns:
            Dict with structured summary data
        """
        # Gate summary
        gates_passed = sum(1 for r in self.result.gate_results.values() if r.passed)
        gates_total = len(self.result.gate_results)

        # Metric highlights
        highlights = self._extract_highlights()

        return {
            'overview': {
                'num_episodes': self.result.num_episodes,
                'success_rate': f"{self.result.success_rate * 100:.1f}%",
                'mean_reward': f"{self.result.mean_reward:.2f}",
                'std_reward': f"{self.result.std_reward:.2f}",
                'timestamp': self.result.timestamp,
                'evaluation_time': f"{self.result.evaluation_time_seconds:.1f}s",
            },
            'promotion_status': {
                'all_passed': self.result.all_gates_passed,
                'gates_passed': gates_passed,
                'gates_total': gates_total,
                'gates': {
                    name: {
                        'passed': r.passed,
                        'message': r.message,
                    }
                    for name, r in self.result.gate_results.items()
                },
            },
            'highlights': highlights,
            'metrics': self.result.metrics,
            'config': self.result.config,
        }

    def _extract_highlights(self) -> Dict[str, Any]:
        """Extract key metric highlights."""
        metrics = self.result.metrics
        highlights = {}

        # Safety highlights
        if 'crash_count' in metrics:
            crash_sum = metrics['crash_count'].get('mean', 0) * self.result.num_episodes
            highlights['total_crashes'] = int(crash_sum)

        if 'safety_score' in metrics:
            highlights['safety_score'] = f"{metrics['safety_score']['mean']:.2f}"

        # Smoothness highlights
        if 'smoothness_score' in metrics:
            highlights['smoothness_score'] = f"{metrics['smoothness_score']['mean']:.2f}"

        if 'mean_jerk' in metrics:
            highlights['mean_jerk'] = f"{metrics['mean_jerk']['mean']:.2f} m/s³"

        # Efficiency highlights
        if 'efficiency_score' in metrics:
            highlights['efficiency_score'] = f"{metrics['efficiency_score']['mean']:.2f}"

        # Performance highlights
        if 'mean_position_error' in metrics:
            highlights['position_error'] = f"{metrics['mean_position_error']['mean']:.3f} m"

        if 'time_at_target' in metrics:
            highlights['time_at_target'] = f"{metrics['time_at_target']['mean']:.1f}s"

        return highlights

    def to_markdown(self) -> str:
        """
        Generate Markdown report.

        Returns:
            Markdown-formatted report string
        """
        summary = self.generate_summary()
        lines = []

        # Header
        lines.append("# UAV Policy Evaluation Report")
        lines.append(f"\nGenerated: {summary['overview']['timestamp']}")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Episodes | {summary['overview']['num_episodes']} |")
        lines.append(f"| Success Rate | {summary['overview']['success_rate']} |")
        lines.append(f"| Mean Reward | {summary['overview']['mean_reward']} ± {summary['overview']['std_reward']} |")
        lines.append(f"| Evaluation Time | {summary['overview']['evaluation_time']} |")
        lines.append("")

        # Promotion Status
        lines.append("## Promotion Status")
        lines.append("")

        status = summary['promotion_status']
        status_emoji = "✅" if status['all_passed'] else "❌"
        lines.append(f"**Overall: {status_emoji} {'PASSED' if status['all_passed'] else 'FAILED'}**")
        lines.append(f"\nGates: {status['gates_passed']}/{status['gates_total']} passed")
        lines.append("")

        lines.append("| Gate | Status | Details |")
        lines.append("|------|--------|---------|")
        for name, gate in status['gates'].items():
            emoji = "✅" if gate['passed'] else "❌"
            lines.append(f"| {name} | {emoji} | {gate['message']} |")
        lines.append("")

        # Key Highlights
        if summary['highlights']:
            lines.append("## Key Highlights")
            lines.append("")
            for key, value in summary['highlights'].items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

        # Detailed Metrics
        lines.append("## Detailed Metrics")
        lines.append("")

        # Group metrics by category
        categories = self._categorize_metrics(summary['metrics'])

        for category, metrics in categories.items():
            lines.append(f"### {category}")
            lines.append("")
            lines.append("| Metric | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")

            for name, stats in metrics.items():
                mean = f"{stats['mean']:.4f}" if abs(stats['mean']) < 1000 else f"{stats['mean']:.2e}"
                std = f"{stats['std']:.4f}" if abs(stats['std']) < 1000 else f"{stats['std']:.2e}"
                min_val = f"{stats['min']:.4f}" if abs(stats['min']) < 1000 else f"{stats['min']:.2e}"
                max_val = f"{stats['max']:.4f}" if abs(stats['max']) < 1000 else f"{stats['max']:.2e}"
                lines.append(f"| {name} | {mean} | {std} | {min_val} | {max_val} |")

            lines.append("")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by UAV Evaluation Harness*")

        return "\n".join(lines)

    def _categorize_metrics(self, metrics: Dict) -> Dict[str, Dict]:
        """Categorize metrics by type."""
        categories = {
            'Flight Performance': {},
            'Safety': {},
            'Smoothness': {},
            'Efficiency': {},
            'Behavioral': {},
            'Other': {},
        }

        category_keywords = {
            'Flight Performance': ['reward', 'success', 'episode', 'position', 'velocity', 'target', 'waypoint'],
            'Safety': ['crash', 'violation', 'safety', 'altitude', 'tilt', 'obstacle', 'near_miss'],
            'Smoothness': ['jerk', 'smooth', 'variance', 'action_rate', 'oscillation', 'angular'],
            'Efficiency': ['thrust', 'power', 'energy', 'efficiency', 'distance', 'path'],
            'Behavioral': ['expert', 'similarity', 'distribution', 'wasserstein', 'coverage'],
        }

        for metric_name, stats in metrics.items():
            categorized = False
            for category, keywords in category_keywords.items():
                if any(kw in metric_name.lower() for kw in keywords):
                    categories[category][metric_name] = stats
                    categorized = True
                    break

            if not categorized:
                categories['Other'][metric_name] = stats

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        metrics = self.result.metrics

        # Check for issues
        if not self.result.all_gates_passed:
            for name, gate in self.result.gate_results.items():
                if not gate.passed:
                    if 'safety' in name.lower():
                        recommendations.append(
                            f"Safety gate failed: Review crash causes and add safety constraints"
                        )
                    elif 'performance' in name.lower():
                        recommendations.append(
                            f"Performance gate failed: Consider additional training or hyperparameter tuning"
                        )

        # Safety recommendations
        if 'crash_count' in metrics and metrics['crash_count']['mean'] > 0:
            recommendations.append(
                "Crashes detected: Add collision avoidance training or reduce action magnitude"
            )

        if 'max_tilt_observed' in metrics and metrics['max_tilt_observed']['mean'] > 50:
            recommendations.append(
                "High tilt angles observed: Consider tilt limit constraints or smoother control"
            )

        # Smoothness recommendations
        if 'mean_jerk' in metrics and metrics['mean_jerk']['mean'] > 50:
            recommendations.append(
                "High jerk detected: Add action smoothness penalty or increase action filtering"
            )

        if 'saturation_ratio' in metrics and metrics['saturation_ratio']['mean'] > 0.2:
            recommendations.append(
                "Frequent action saturation: Consider scaling action limits or improving control"
            )

        # Efficiency recommendations
        if 'efficiency_score' in metrics and metrics['efficiency_score']['mean'] < 0.5:
            recommendations.append(
                "Low efficiency: Consider energy penalty in reward function"
            )

        # Success rate recommendations
        if self.result.success_rate < 0.8:
            recommendations.append(
                f"Success rate ({self.result.success_rate:.1%}) below 80%: Additional training recommended"
            )

        return recommendations

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Generate JSON report.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string
        """
        summary = self.generate_summary()

        # Add episode details
        summary['episodes'] = [
            {
                'id': ep.episode_id,
                'success': ep.success,
                'reward': ep.total_reward,
                'length': ep.episode_length,
                'termination': ep.termination_reason,
            }
            for ep in self.result.episodes
        ]

        json_str = json.dumps(summary, indent=2, default=str)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def to_html(self, filepath: Optional[str] = None) -> str:
        """
        Generate HTML report with basic styling.

        Args:
            filepath: Optional path to save HTML file

        Returns:
            HTML string
        """
        # Convert markdown to HTML (basic conversion)
        markdown_content = self.to_markdown()

        # Basic HTML wrapper
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>UAV Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
{self._markdown_to_html(markdown_content)}
</body>
</html>"""

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(html)

        return html

    def _markdown_to_html(self, markdown: str) -> str:
        """Basic markdown to HTML conversion."""
        import re

        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Tables (simplified)
        lines = html.split('\n')
        in_table = False
        new_lines = []

        for line in lines:
            if '|' in line and '---' not in line:
                if not in_table:
                    new_lines.append('<table>')
                    in_table = True

                cells = [c.strip() for c in line.split('|')[1:-1]]
                if cells:
                    row = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
                    new_lines.append(row)
            elif '---' in line and '|' in line:
                continue  # Skip table separator
            else:
                if in_table:
                    new_lines.append('</table>')
                    in_table = False
                new_lines.append(line)

        if in_table:
            new_lines.append('</table>')

        html = '\n'.join(new_lines)

        # Paragraphs
        html = re.sub(r'\n\n', '</p><p>', html)
        html = f'<p>{html}</p>'

        # Emojis
        html = html.replace('✅', '<span class="passed">✓</span>')
        html = html.replace('❌', '<span class="failed">✗</span>')

        return html


def generate_markdown_report(result: 'EvaluationResult') -> str:
    """
    Convenience function to generate markdown report.

    Args:
        result: EvaluationResult from evaluation

    Returns:
        Markdown string
    """
    generator = ReportGenerator(result)
    return generator.to_markdown()


def generate_json_report(
    result: 'EvaluationResult',
    filepath: Optional[str] = None,
) -> str:
    """
    Convenience function to generate JSON report.

    Args:
        result: EvaluationResult from evaluation
        filepath: Optional path to save

    Returns:
        JSON string
    """
    generator = ReportGenerator(result)
    return generator.to_json(filepath)
