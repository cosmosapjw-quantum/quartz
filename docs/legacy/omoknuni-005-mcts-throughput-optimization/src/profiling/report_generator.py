"""
Report Generator
================

Generates profiling reports in multiple formats:
- HTML: Interactive dashboards with charts
- JSON: Machine-readable structured data
- Flamegraph: SVG flame graphs for call stack visualization
- Markdown: Summary reports

Integrates data from all profilers for comprehensive analysis.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def generate_json_report(metrics: Dict[str, Any], output_path: Path):
    """
    Generate JSON format report.

    Args:
        metrics: Profiling metrics dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def generate_html_report(metrics: Dict[str, Any], output_path: Path):
    """
    Generate interactive HTML report with charts.

    Args:
        metrics: Profiling metrics dictionary
        output_path: Output file path
    """
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCTS Profiling Report - {session_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #999;
        }}
        .chart {{
            margin: 20px 0;
            min-height: 400px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .error {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .success {{
            background-color: #d4edda;
            border-left-color: #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MCTS Profiling Report</h1>
        <p><strong>Session ID:</strong> {session_id}</p>
        <p><strong>Duration:</strong> {duration:.2f} seconds</p>

        {summary_section}
        {gil_section}
        {inference_section}
        {thread_section}
        {memory_section}
        {cpp_section}

    </div>

    <script>
        {plot_scripts}
    </script>
</body>
</html>
    """

    # Extract session info
    session_id = metrics.get('session_id', 'unknown')
    duration = metrics.get('summary', {}).get('session_duration_seconds', 0.0)

    # Generate sections
    summary_section = _generate_summary_section(metrics)
    gil_section = _generate_gil_section(metrics.get('gil_metrics', {}))
    inference_section = _generate_inference_section(metrics.get('inference_metrics', {}))
    thread_section = _generate_thread_section(metrics.get('thread_metrics', {}))
    memory_section = _generate_memory_section(metrics.get('memory_metrics', {}))
    cpp_section = _generate_cpp_section(metrics.get('cpp_instrumentation', {}))

    # Generate plot scripts
    plot_scripts = _generate_plot_scripts(metrics)

    # Fill template
    html_content = html_template.format(
        session_id=session_id,
        duration=duration,
        summary_section=summary_section,
        gil_section=gil_section,
        inference_section=inference_section,
        thread_section=thread_section,
        memory_section=memory_section,
        cpp_section=cpp_section,
        plot_scripts=plot_scripts
    )

    with open(output_path, 'w') as f:
        f.write(html_content)


def _generate_summary_section(metrics: Dict[str, Any]) -> str:
    """Generate overall summary section."""
    html = "<h2>Overall Summary</h2><div class='metric-grid'>"

    # Extract key metrics
    gil_efficiency = metrics.get('gil_metrics', {}).get('summary', {}).get('gil_efficiency', 0.0)
    avg_latency = metrics.get('inference_metrics', {}).get('summary', {}).get('avg_latency_us', 0.0)
    thread_utilization = metrics.get('thread_metrics', {}).get('pool_summary', {}).get('avg_thread_utilization', 0.0)
    memory_growth = metrics.get('memory_metrics', {}).get('summary', {}).get('memory_growth_mb', 0.0)

    # Determine status classes
    gil_class = 'success' if gil_efficiency > 70 else ('warning' if gil_efficiency > 50 else 'error')
    latency_class = 'success' if avg_latency < 1000 else ('warning' if avg_latency < 5000 else 'error')
    thread_class = 'success' if thread_utilization > 70 else ('warning' if thread_utilization > 50 else 'error')
    memory_class = 'success' if abs(memory_growth) < 100 else ('warning' if abs(memory_growth) < 500 else 'error')

    html += f"""
    <div class='metric-card {gil_class}'>
        <div class='metric-label'>GIL Efficiency</div>
        <div class='metric-value'>{gil_efficiency:.1f}<span class='metric-unit'>%</span></div>
    </div>
    <div class='metric-card {latency_class}'>
        <div class='metric-label'>Avg Inference Latency</div>
        <div class='metric-value'>{avg_latency/1000:.2f}<span class='metric-unit'>ms</span></div>
    </div>
    <div class='metric-card {thread_class}'>
        <div class='metric-label'>Thread Utilization</div>
        <div class='metric-value'>{thread_utilization:.1f}<span class='metric-unit'>%</span></div>
    </div>
    <div class='metric-card {memory_class}'>
        <div class='metric-label'>Memory Growth</div>
        <div class='metric-value'>{memory_growth:.1f}<span class='metric-unit'>MB</span></div>
    </div>
    """

    html += "</div>"
    return html


def _generate_gil_section(gil_metrics: Dict[str, Any]) -> str:
    """Generate GIL profiling section."""
    if not gil_metrics:
        return ""

    summary = gil_metrics.get('summary', {})

    html = "<h2>GIL Profiling</h2><div class='metric-grid'>"

    html += f"""
    <div class='metric-card'>
        <div class='metric-label'>GIL Utilization</div>
        <div class='metric-value'>{summary.get('gil_utilization', 0.0):.1f}<span class='metric-unit'>%</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Avg Wait Time/Thread</div>
        <div class='metric-value'>{summary.get('avg_wait_time_per_thread', 0.0)*1000:.2f}<span class='metric-unit'>ms</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Contention Events</div>
        <div class='metric-value'>{summary.get('total_contention_events', 0)}</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>GIL Efficiency</div>
        <div class='metric-value'>{summary.get('gil_efficiency', 0.0):.1f}<span class='metric-unit'>%</span></div>
    </div>
    """

    html += "</div>"

    # Top wait hotspots
    hotspots = gil_metrics.get('top_wait_hotspots', [])
    if hotspots:
        html += "<h3>Top GIL Wait Hotspots</h3><table>"
        html += "<tr><th>Location</th><th>Wait Time (ms)</th></tr>"
        for hotspot in hotspots[:10]:
            html += f"<tr><td>{hotspot['location']}</td><td>{hotspot['total_wait_time_ms']:.2f}</td></tr>"
        html += "</table>"

    html += "<div id='gil-chart' class='chart'></div>"

    return html


def _generate_inference_section(inference_metrics: Dict[str, Any]) -> str:
    """Generate inference profiling section."""
    if not inference_metrics:
        return ""

    summary = inference_metrics.get('summary', {})

    html = "<h2>Inference Pipeline</h2><div class='metric-grid'>"

    html += f"""
    <div class='metric-card'>
        <div class='metric-label'>Avg Latency</div>
        <div class='metric-value'>{summary.get('avg_latency_us', 0.0)/1000:.2f}<span class='metric-unit'>ms</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>P99 Latency</div>
        <div class='metric-value'>{summary.get('p99_latency_us', 0.0)/1000:.2f}<span class='metric-unit'>ms</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Avg Batch Size</div>
        <div class='metric-value'>{summary.get('avg_batch_size', 0.0):.1f}</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>DLPack Usage</div>
        <div class='metric-value'>{summary.get('dlpack_usage_rate', 0.0)*100:.1f}<span class='metric-unit'>%</span></div>
    </div>
    """

    html += "</div>"

    # Stage breakdown
    stage_breakdown = inference_metrics.get('stage_breakdown', {})
    if stage_breakdown:
        html += "<h3>Pipeline Stage Breakdown</h3><table>"
        html += "<tr><th>Stage</th><th>Avg (μs)</th><th>P90 (μs)</th><th>P99 (μs)</th><th>% of Total</th></tr>"
        for stage, stats in stage_breakdown.items():
            html += f"""<tr>
                <td>{stage.replace('_', ' ').title()}</td>
                <td>{stats['avg_us']:.2f}</td>
                <td>{stats['p90_us']:.2f}</td>
                <td>{stats['p99_us']:.2f}</td>
                <td>{stats['percentage']:.1f}%</td>
            </tr>"""
        html += "</table>"

    html += "<div id='inference-chart' class='chart'></div>"

    return html


def _generate_thread_section(thread_metrics: Dict[str, Any]) -> str:
    """Generate thread coordination section."""
    if not thread_metrics:
        return ""

    summary = thread_metrics.get('summary', {})
    pool_summary = thread_metrics.get('pool_summary', {})

    html = "<h2>Thread Coordination</h2><div class='metric-grid'>"

    html += f"""
    <div class='metric-card'>
        <div class='metric-label'>Thread Utilization</div>
        <div class='metric-value'>{pool_summary.get('avg_thread_utilization', 0.0):.1f}<span class='metric-unit'>%</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Avg Future Latency</div>
        <div class='metric-value'>{summary.get('avg_future_latency_us', 0.0)/1000:.2f}<span class='metric-unit'>ms</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Success Rate</div>
        <div class='metric-value'>{summary.get('success_rate', 0.0):.1f}<span class='metric-unit'>%</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Futures/sec</div>
        <div class='metric-value'>{summary.get('futures_per_second', 0.0):.1f}</div>
    </div>
    """

    html += "</div>"
    html += "<div id='thread-chart' class='chart'></div>"

    return html


def _generate_memory_section(memory_metrics: Dict[str, Any]) -> str:
    """Generate memory profiling section."""
    if not memory_metrics:
        return ""

    summary = memory_metrics.get('summary', {})

    html = "<h2>Memory Profiling</h2><div class='metric-grid'>"

    html += f"""
    <div class='metric-card'>
        <div class='metric-label'>Current Memory</div>
        <div class='metric-value'>{summary.get('current_memory_mb', 0.0):.1f}<span class='metric-unit'>MB</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Peak Memory</div>
        <div class='metric-value'>{summary.get('peak_memory_mb', 0.0):.1f}<span class='metric-unit'>MB</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Memory Growth</div>
        <div class='metric-value'>{summary.get('memory_growth_mb', 0.0):.1f}<span class='metric-unit'>MB</span></div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>GC Events/sec</div>
        <div class='metric-value'>{summary.get('gc_events_per_second', 0.0):.2f}</div>
    </div>
    """

    html += "</div>"
    html += "<div id='memory-chart' class='chart'></div>"

    return html


def _generate_cpp_section(cpp_metrics: Dict[str, Any]) -> str:
    """Generate C++ instrumentation section."""
    if not cpp_metrics:
        return ""

    html = "<h2>C++ Instrumentation</h2>"

    html += "<table>"
    html += "<tr><th>Metric</th><th>Call Count</th><th>Total Time (ms)</th><th>Avg Time (μs)</th></tr>"

    for metric_name, data in cpp_metrics.items():
        call_count = data.get('call_count', 0)
        total_ns = data.get('total_elapsed_ns', 0)
        total_ms = total_ns / 1e6
        avg_us = (total_ns / call_count / 1000) if call_count > 0 else 0

        html += f"""<tr>
            <td>{metric_name}</td>
            <td>{call_count:,}</td>
            <td>{total_ms:.2f}</td>
            <td>{avg_us:.2f}</td>
        </tr>"""

    html += "</table>"

    return html


def _generate_plot_scripts(metrics: Dict[str, Any]) -> str:
    """Generate Plotly chart scripts."""
    scripts = []

    # GIL chart (thread breakdown)
    gil_metrics = metrics.get('gil_metrics', {})
    if gil_metrics:
        thread_metrics_data = gil_metrics.get('thread_metrics', {})
        if thread_metrics_data:
            scripts.append("""
            var gil_data = [{
                x: %s,
                y: %s,
                name: 'With GIL',
                type: 'bar'
            }, {
                x: %s,
                y: %s,
                name: 'Without GIL',
                type: 'bar'
            }];
            var gil_layout = {
                title: 'GIL Time per Thread',
                xaxis: { title: 'Thread' },
                yaxis: { title: 'Time (seconds)' },
                barmode: 'stack'
            };
            Plotly.newPlot('gil-chart', gil_data, gil_layout);
            """ % (
                str([f"Thread {m['thread_id']}" for m in thread_metrics_data.values()]),
                str([m['time_with_gil_seconds'] for m in thread_metrics_data.values()]),
                str([f"Thread {m['thread_id']}" for m in thread_metrics_data.values()]),
                str([m['time_without_gil_seconds'] for m in thread_metrics_data.values()])
            ))

    return '\n'.join(scripts)


def generate_flamegraph(metrics: Dict[str, Any], output_path: Path):
    """
    Generate flamegraph SVG from profiling data.

    Args:
        metrics: Profiling metrics dictionary
        output_path: Output SVG file path
    """
    # Flamegraph generation requires external tools (flamegraph.pl or speedscope)
    # For now, we'll generate a simple text-based call tree that can be
    # visualized with external tools

    logger.info(f"Flamegraph generation placeholder - would save to {output_path}")

    # Create simple SVG placeholder
    svg_content = f"""<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
    <text x="600" y="300" text-anchor="middle" font-size="24">
        Flamegraph generation requires py-spy or similar profiling tools.
        Install py-spy and use: py-spy record --format speedscope -o output.json -- python your_script.py
    </text>
</svg>
"""

    with open(output_path, 'w') as f:
        f.write(svg_content)


def generate_markdown_report(metrics: Dict[str, Any], output_path: Path):
    """
    Generate Markdown summary report.

    Args:
        metrics: Profiling metrics dictionary
        output_path: Output markdown file path
    """
    session_id = metrics.get('session_id', 'unknown')
    duration = metrics.get('summary', {}).get('session_duration_seconds', 0.0)

    md_content = f"""# MCTS Profiling Report

**Session ID:** {session_id}
**Duration:** {duration:.2f} seconds

## Overall Summary

"""

    # GIL summary
    gil_summary = metrics.get('gil_metrics', {}).get('summary', {})
    if gil_summary:
        md_content += f"""### GIL Profiling
- GIL Efficiency: {gil_summary.get('gil_efficiency', 0.0):.1f}%
- Average Wait Time per Thread: {gil_summary.get('avg_wait_time_per_thread', 0.0)*1000:.2f}ms
- Contention Events: {gil_summary.get('total_contention_events', 0)}

"""

    # Inference summary
    inference_summary = metrics.get('inference_metrics', {}).get('summary', {})
    if inference_summary:
        md_content += f"""### Inference Pipeline
- Average Latency: {inference_summary.get('avg_latency_us', 0.0)/1000:.2f}ms
- P99 Latency: {inference_summary.get('p99_latency_us', 0.0)/1000:.2f}ms
- Average Batch Size: {inference_summary.get('avg_batch_size', 0.0):.1f}
- DLPack Usage Rate: {inference_summary.get('dlpack_usage_rate', 0.0)*100:.1f}%

"""

    # Thread summary
    thread_summary = metrics.get('thread_metrics', {}).get('summary', {})
    if thread_summary:
        md_content += f"""### Thread Coordination
- Thread Utilization: {metrics.get('thread_metrics', {}).get('pool_summary', {}).get('avg_thread_utilization', 0.0):.1f}%
- Average Future Latency: {thread_summary.get('avg_future_latency_us', 0.0)/1000:.2f}ms
- Success Rate: {thread_summary.get('success_rate', 0.0):.1f}%

"""

    # Memory summary
    memory_summary = metrics.get('memory_metrics', {}).get('summary', {})
    if memory_summary:
        md_content += f"""### Memory Profiling
- Current Memory: {memory_summary.get('current_memory_mb', 0.0):.1f}MB
- Peak Memory: {memory_summary.get('peak_memory_mb', 0.0):.1f}MB
- Memory Growth: {memory_summary.get('memory_growth_mb', 0.0):.1f}MB
- GC Events per Second: {memory_summary.get('gc_events_per_second', 0.0):.2f}

"""

    with open(output_path, 'w') as f:
        f.write(md_content)

    logger.info(f"Markdown report saved to {output_path}")
