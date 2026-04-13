"""Generate an SVG visualization for the thermal-zone adjacency graph."""

from __future__ import annotations

import html
import math
from pathlib import Path

from thermal_zone_graph import THERMAL_ZONE_EDGES, THERMAL_ZONES


OUTPUT_PATH = Path("outputs/thermal_zone_adjacency_graph.svg")
WIDTH = 3200
HEIGHT = 2200
GRAPH_X = 70
GRAPH_Y = 135
GRAPH_W = 2100
GRAPH_H = 1950
LEGEND_X = 2225
LEGEND_Y = 105

GROUP_COLORS = {
    "HALL": "#2f6f73",
    "COMMERCE": "#c8672e",
    "DINING": "#b08d2f",
    "OFFICE": "#496fa8",
    "RESTROOM": "#7b6aa8",
    "BREAKROOM": "#5d8a3d",
}


def zone_group(zone_name: str) -> str:
    if "HALL" in zone_name:
        return "HALL"
    for group in ("COMMERCE", "DINING", "OFFICE", "RESTROOM", "BREAKROOM"):
        if group in zone_name:
            return group
    return "OTHER"


def fallback_layout() -> dict[int, tuple[float, float]]:
    n = len(THERMAL_ZONES)
    radius = min(GRAPH_W, GRAPH_H) * 0.45
    cx = GRAPH_X + GRAPH_W / 2
    cy = GRAPH_Y + GRAPH_H / 2
    return {
        idx: (
            cx + radius * math.cos(2 * math.pi * idx / n),
            cy + radius * math.sin(2 * math.pi * idx / n),
        )
        for idx in range(n)
    }


def graph_layout() -> dict[int, tuple[float, float]]:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(len(THERMAL_ZONES)))
        for i, j, weight in THERMAL_ZONE_EDGES:
            graph.add_edge(i, j, weight=weight)

        raw_pos = nx.spring_layout(
            graph,
            seed=42,
            k=0.65,
            iterations=350,
            weight="weight",
        )
    except Exception:
        return fallback_layout()

    xs = [xy[0] for xy in raw_pos.values()]
    ys = [xy[1] for xy in raw_pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 95

    def scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
        if source_max == source_min:
            return (target_min + target_max) / 2
        return target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)

    return {
        idx: (
            scale(xy[0], min_x, max_x, GRAPH_X + pad, GRAPH_X + GRAPH_W - pad),
            scale(xy[1], min_y, max_y, GRAPH_Y + pad, GRAPH_Y + GRAPH_H - pad),
        )
        for idx, xy in raw_pos.items()
    }


def draw_svg() -> str:
    pos = graph_layout()
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    parts.append(
        """
<style>
  .title { font: 700 48px sans-serif; fill: #1f2933; }
  .subtitle { font: 26px sans-serif; fill: #52606d; }
  .edge { stroke: #8a97a8; stroke-opacity: 0.55; }
  .node { stroke: #15212f; stroke-width: 2; }
  .node-label { font: 700 21px sans-serif; fill: #ffffff; dominant-baseline: middle; text-anchor: middle; pointer-events: none; }
  .legend-title { font: 700 34px sans-serif; fill: #1f2933; }
  .legend-text { font: 21px sans-serif; fill: #1f2933; }
  .group-text { font: 22px sans-serif; fill: #1f2933; }
</style>
"""
    )
    parts.append('<rect width="100%" height="100%" fill="#f7f4ed"/>')
    parts.append('<rect x="35" y="85" width="2170" height="2070" rx="36" fill="#ffffff" stroke="#d8d4c9"/>')
    parts.append('<rect x="2205" y="85" width="960" height="2070" rx="36" fill="#ffffff" stroke="#d8d4c9"/>')
    parts.append('<text x="70" y="58" class="title">Thermal-Zone Adjacency Graph</text>')
    parts.append(
        '<text x="70" y="94" class="subtitle">56 nodes, 113 undirected edges. Edge width indicates shared inter-zone surface pairs.</text>'
    )

    for i, j, weight in THERMAL_ZONE_EDGES:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        stroke_width = 0.7 + 0.7 * weight
        title = f"{THERMAL_ZONES[i]} -- {THERMAL_ZONES[j]} | shared_surface_pairs={weight}"
        parts.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke-width="{stroke_width:.1f}">'
            f"<title>{html.escape(title)}</title></line>"
        )

    for idx, zone in enumerate(THERMAL_ZONES):
        x, y = pos[idx]
        group = zone_group(zone)
        color = GROUP_COLORS.get(group, "#777777")
        parts.append(
            f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="25" fill="{color}">'
            f"<title>{idx}: {html.escape(zone)} | group={group}</title></circle>"
        )
        parts.append(f'<text class="node-label" x="{x:.1f}" y="{y + 0.5:.1f}">{idx}</text>')

    parts.append(f'<text x="{LEGEND_X}" y="{LEGEND_Y}" class="legend-title">Node Index</text>')
    y = LEGEND_Y + 52
    for group, color in GROUP_COLORS.items():
        parts.append(f'<circle cx="{LEGEND_X + 12}" cy="{y - 7}" r="10" fill="{color}" stroke="#15212f"/>')
        parts.append(f'<text x="{LEGEND_X + 34}" y="{y}" class="group-text">{group}</text>')
        y += 36

    y += 34
    col_width = 440
    row_height = 32
    rows_per_column = math.ceil(len(THERMAL_ZONES) / 2)
    list_y = y
    for idx, zone in enumerate(THERMAL_ZONES):
        group = zone_group(zone)
        color = GROUP_COLORS.get(group, "#777777")
        label = zone.replace("THERMAL ZONE: ", "")
        col = idx // rows_per_column
        row = idx % rows_per_column
        item_x = LEGEND_X + col * col_width
        item_y = list_y + row * row_height
        parts.append(f'<circle cx="{item_x + 10}" cy="{item_y - 7}" r="8" fill="{color}" stroke="#15212f"/>')
        parts.append(
            f'<text x="{item_x + 28}" y="{item_y}" class="legend-text">{idx:02d}. {html.escape(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(draw_svg(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
