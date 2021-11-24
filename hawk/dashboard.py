"""
dashboard.py

This file generates grafana dashboards for ML metrics.
"""

from grafanalib.core import (
    Dashboard,
    Graph,
    OPS_FORMAT,
    single_y_axis,
    Target,
)
from grafanalib._gen import DashboardEncoder

import json


def gen_dashboard(metric_name, metric_description, queries):
    panels = []
    for name, query in queries.items():
        panel = Graph(
            title=name,
            dataSource="default",
            targets=[
                Target(
                    expr=query,
                    legendFormat="{{ handler }}",
                    refId="A",
                ),
            ],
            yAxes=single_y_axis(format=OPS_FORMAT),
        )
        panels.append(panel)

    dashboard = Dashboard(
        title=f"{metric_name} Dashboard",
        description=f"{metric_description}",
        panels=panels,
    ).auto_panel_ids()

    return json.dumps(
        {"dashboard": dashboard.to_json_data()},
        sort_keys=True,
        indent=2,
        cls=DashboardEncoder,
    )
