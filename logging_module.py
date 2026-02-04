#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation logging module.

Simple global lists for tracking simulation events and task execution.
"""

# Logger for Gantt chart data
task_log_px = []

# Human-readable log messages
logs_str = []


def reset_logs():
    """Clear all logged data between simulation runs."""
    task_log_px.clear()
    logs_str.clear()
