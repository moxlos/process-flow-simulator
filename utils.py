#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 2024

@author: lefteris

@subject: function utils 
"""
import pandas as pd
from collections import defaultdict
import copy
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


def _ensure_numeric_columns(df, columns=None):
    """
    Ensure specified columns are numeric (float).
    Handles Timedelta by converting to total_seconds().

    Args:
        df: pandas DataFrame
        columns: list of column names to convert (default: ['Start', 'Finish'])

    Returns:
        DataFrame with numeric columns
    """
    if columns is None:
        columns = ['Start', 'Finish']

    for col in columns:
        if col not in df.columns:
            continue
        # If already numeric, skip
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # If timedelta, convert to seconds (check dtype string for compatibility)
        if 'timedelta' in str(df[col].dtype):
            df[col] = df[col].dt.total_seconds()
        else:
            # Try to convert, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


####################### Plots #################################
# Visualization function for Plotly Gantt chart
def plot_gantt_px(task_log_px):
    # Convert the task log to a DataFrame
    if not task_log_px:  # Check if task_log has data
        print("No data to plot yet.")
        return
    reference_start_time = datetime.now()
    
    task_log = copy.deepcopy(task_log_px)
    # Convert time values in task_log_px to datetimes based on the reference start time
    for entry in task_log:
        entry['Start'] = reference_start_time + timedelta(seconds=entry['Start'])
        entry['Finish'] = reference_start_time + timedelta(seconds=entry['Finish'])

    # Create DataFrame
    df = pd.DataFrame(task_log)
    
    # Sort products by ID to define y-axis order in ascending order
    df = df.sort_values(by="Product", ascending=True)
    
    # Generate Gantt chart with ordered y-axis
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Product", 
        color="Task", 
        title="Factory Simulation Gantt Chart", 
        text="Worker",
        category_orders={"Product": df["Product"].unique().tolist()}  # Custom y-axis ordering
    )
    
    # Update layout for x-axis format
    fig.update_layout(
        xaxis_title="Simulation Time",
        yaxis_title="Products",
        xaxis=dict(tickformat="%H:%M:%S")  # Show hours, minutes, and seconds
    )
    
    fig.show()
    
def plot_gantt_px_st(task_log_px):
    # Convert the task log to a DataFrame
    if not task_log_px:  # Check if task_log has data
        print("No data to plot yet.")
        return
    #reference_start_time = datetime.now()
    reference_start_time = datetime.combine(datetime.today(), datetime.min.time())
    task_log = copy.deepcopy(task_log_px)
    # Convert time values in task_log_px to datetimes based on the reference start time
    for entry in task_log:
        entry['Start'] = reference_start_time + timedelta(seconds=entry['Start'])
        entry['Finish'] = reference_start_time + timedelta(seconds=entry['Finish'])

    # Create DataFrame
    df = pd.DataFrame(task_log)
    
    # Sort products by ID to define y-axis order in ascending order
    df = df.sort_values(by="Product", ascending=True)
    
    # Generate Gantt chart with ordered y-axis
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Product", 
        color="Task", 
        title="Factory Simulation Gantt Chart", 
        text="Worker",
        category_orders={"Product": df["Product"].unique().tolist()}  # Custom y-axis ordering
    )
    
    # Update layout for x-axis format
    fig.update_layout(
        xaxis_title="Simulation Time",
        yaxis_title="Products",
        xaxis=dict(tickformat="%H:%M:%S")  # Show hours, minutes, and seconds
    )
    
    return fig

def plot_worker_util(worker_utilization):
    
    worker_ids = list(worker_utilization.keys())
    utilization_rates = list(worker_utilization.values())
    
    fig = go.Figure(data=[
        go.Bar(x=worker_ids, y=utilization_rates, marker_color='skyblue')
    ])
    
    fig.update_layout(
        title="Worker Utilization Rate",
        xaxis_title="Worker ID",
        yaxis_title="Utilization Rate (%)"
    )
    
    fig.show()
    
def plot_worker_util_st(worker_utilization):
    
    worker_ids = list(worker_utilization.keys())
    utilization_rates = list(worker_utilization.values())
    
    fig = go.Figure(data=[
        go.Bar(x=worker_ids, y=utilization_rates, marker_color='skyblue')
    ])
    
    fig.update_layout(
        title="Worker Utilization Rate",
        xaxis_title="Worker ID",
        yaxis_title="Utilization Rate (%)"
    )
    
    return fig

def plot_worker_cplt(task_completion_time):
    worker_ids = list(task_completion_time.keys())
    avg_task_times = list(task_completion_time.values())
    
    fig = go.Figure(data=[
        go.Bar(x=worker_ids, y=avg_task_times, marker_color='salmon')
    ])
    
    fig.update_layout(
        title="Average Task Completion Time by Worker",
        xaxis_title="Worker ID",
        yaxis_title="Average Task Completion Time (seconds)"
    )
    
    fig.show()

def plot_worker_cplt_st(task_completion_time):
    worker_ids = list(task_completion_time.keys())
    avg_task_times = list(task_completion_time.values())
    
    fig = go.Figure(data=[
        go.Bar(x=worker_ids, y=avg_task_times, marker_color='salmon')
    ])
    
    fig.update_layout(
        title="Average Task Completion Time by Worker",
        xaxis_title="Worker ID",
        yaxis_title="Average Task Completion Time (seconds)"
    )
    
    return fig
    

def plot_task_times(avg_time_per_task):
    task_types = list(avg_time_per_task.keys())
    avg_times = list(avg_time_per_task.values())
    
    fig = go.Figure(data=[
        go.Pie(labels=task_types, values=avg_times, hole=.3)
    ])
    
    fig.update_layout(
        title="Average Time per Task"
    )
    
    fig.show()
    
def plot_task_times_st(avg_time_per_task):
    task_types = list(avg_time_per_task.keys())
    avg_times = list(avg_time_per_task.values())
    
    fig = go.Figure(data=[
        go.Pie(labels=task_types, values=avg_times, hole=.3)
    ])
    
    fig.update_layout(
        title="Average Time per Task"
    )
    
    return fig
    

def plot_product_completion(product_completion_times):
    fig = go.Figure(data=[
        go.Histogram(x=product_completion_times['CompletionTime'], nbinsx=10, marker_color='teal')
    ])
    
    fig.update_layout(
        title="Distribution of Product Completion Times",
        xaxis_title="Product Completion Time (seconds)",
        yaxis_title="Frequency"
    )
    
    fig.show()


def plot_product_completion_st(product_completion_times):
    fig = go.Figure(data=[
        go.Histogram(x=product_completion_times['CompletionTime'], nbinsx=10, marker_color='teal')
    ])
    
    fig.update_layout(
        title="Distribution of Product Completion Times",
        xaxis_title="Product Completion Time (seconds)",
        yaxis_title="Frequency"
    )
    
    return fig

def product_task_completion_times(product_df,task_df):
    # Sort task data by time
    task_df = task_df.sort_values(by="Finish")

    # Sort product data by time
    product_df = product_df.sort_values(by="Finish")
    
    # Compute cumulative counts
    task_df["cumulative_count"] = task_df.groupby("Task").cumcount() + 1
    product_df["cumulative_count"] = range(1, len(product_df) + 1)

    # Create line plots
    task_fig = px.line(
        task_df,
        x="Finish",
        y="cumulative_count",
        color="Task",
        title="Cumulative Task Completion Over Time",
        labels={"time": "Time", "cumulative_count": "Cumulative Tasks Completed"},
    )
    product_fig = px.line(
    product_df,
    x="Finish",
    y="cumulative_count",
    title="Cumulative Product Completion Over Time",
    labels={"time": "Time", "cumulative_count": "Cumulative Products Completed"},
    )
    return task_fig, product_fig


################# Metrics ###########################
def calculate_metrics(task_log_px):
    """
    Calculate various metrics from the task log.
    Parameters:
    task_log_px (list): List of task log entries with 'Task', 'Product', 'Worker', 'Start', 'Finish'.
    
    Returns:
    dict: Calculated metrics including Utilization Rate, Average Task Completion Time by Worker,
          Task Frequency by Worker, Average Product Completion Time, Average Time per Task, and
          Total Simulation Time.
    """
    # Convert the task log to a DataFrame (explicitly convert to list for proxy compatibility)
    df = pd.DataFrame(task_log_px)

    # Ensure Start and Finish are numeric (handles Timedelta correctly)
    df = _ensure_numeric_columns(df, ['Start', 'Finish'])
    df['Duration'] = df['Finish'] - df['Start']

    # Metric 1: Utilization Rate
    total_simulation_time = float(df['Finish'].max() - df['Start'].min())
    utilization_rate = {k: float(v) for k, v in (df.groupby('Worker')['Duration'].sum() / total_simulation_time).to_dict().items()}

    # Metric 2: Average Task Completion Time by Worker
    avg_task_time_by_worker = {k: float(v) for k, v in df.groupby('Worker')['Duration'].mean().to_dict().items()}

    # Metric 3: Task Frequency by Worker
    task_frequency_by_worker = df.groupby(['Worker', 'Task']).size().unstack(fill_value=0).to_dict(orient='index')

    #task completion times
    task_completion_times = df.groupby(['Product','Task']).agg(Start=('Start', 'min'), Finish=('Finish', 'max'))

    # Metric 4: Average Product Completion Time
    product_completion_times = df.groupby('Product').agg(Start=('Start', 'min'), Finish=('Finish', 'max'))
    product_completion_times_det = product_completion_times.copy()
    product_completion_times['CompletionTime'] = product_completion_times['Finish'] - product_completion_times['Start']
    avg_product_completion_time = float(product_completion_times['CompletionTime'].mean())

    # Metric 5: Average Time per Task
    avg_time_per_task = {k: float(v) for k, v in df.groupby('Task')['Duration'].mean().to_dict().items()}

    # Metric 6: Total Simulation Time
    total_simulation_time = float(df['Finish'].max() - df['Start'].min())

    # Compile results into a dictionary
    metrics = {
        'Utilization Rate': utilization_rate,
        'Average Task Completion Time by Worker': avg_task_time_by_worker,
        'Task Frequency by Worker': task_frequency_by_worker,
        'Average Product Completion Time': avg_product_completion_time,
        'Average Time per Task': avg_time_per_task,
        'Total Simulation Time': total_simulation_time
    }

    return metrics, utilization_rate, product_completion_times.reset_index(), product_completion_times_det.reset_index(), task_completion_times.reset_index()

def generate_report(metrics):
    """
    Generate a formatted report for the simulation metrics.
    
    Parameters:
    metrics (dict): Dictionary of calculated metrics.
    
    Returns:
    str: Formatted report as a multi-line string.
    """
    report = []
    
    # Utilization Rate
    report.append("1. Utilization Rate (Worker active time as a percentage of total simulation time):")
    for worker, rate in metrics['Utilization Rate'].items():
        report.append(f"   - Worker {worker}: {rate:.2%}")
    
    # Average Task Completion Time by Worker
    report.append("\n2. Average Task Completion Time by Worker (seconds):")
    for worker, avg_time in metrics['Average Task Completion Time by Worker'].items():
        report.append(f"   - Worker {worker}: {avg_time:.2f} seconds")
    
    # Task Frequency by Worker
    report.append("\n3. Task Frequency by Worker:")
    for worker, tasks in metrics['Task Frequency by Worker'].items():
        report.append(f"   - Worker {worker}:")
        for task, frequency in tasks.items():
            report.append(f"      - {task}: {frequency} times")
    
    # Average Product Completion Time
    avg_product_completion_time = metrics['Average Product Completion Time']
    report.append(f"\n4. Average Product Completion Time (seconds): {avg_product_completion_time:.2f} seconds")
    
    # Average Time per Task
    report.append("\n5. Average Time per Task (seconds):")
    for task, avg_time in metrics['Average Time per Task'].items():
        report.append(f"   - {task}: {avg_time:.2f} seconds")
    
    # Total Simulation Time
    total_simulation_time = metrics['Total Simulation Time']
    report.append(f"\n6. Total Simulation Time (seconds): {total_simulation_time:.2f} seconds")
    
    # Join the report list into a single string
    return "\n".join(report)


def calculate_product_task_wait_times(task_log_px):
    """
    Calculates wait times between tasks for each product based on gaps between task assignments.
    
    Args:
        task_log_px (list of dict): List containing logs of tasks, each with:
            - 'Task': Task name
            - 'Product': Product ID
            - 'Worker': Worker ID
            - 'Start': Start time of task
            - 'Finish': Finish time of task
    
    Returns:
        list of dict: A list with wait times for each gap between tasks per product.
                      Each dict contains:
                      - 'Product': Product ID
                      - 'Waiting Time': Time in seconds between the end of one task and the start of the next
                      - 'Previous Task End': End time of the previous task
                      - 'Next Task Start': Start time of the next task
    """
    # Convert task log to DataFrame (explicitly convert to list for proxy compatibility)
    df = pd.DataFrame(task_log_px)

    # Ensure Start and Finish are numeric (handles Timedelta correctly)
    df = _ensure_numeric_columns(df, ['Start', 'Finish'])

    # Sort by Product and Start time to ensure tasks are in order for each product
    df = df.sort_values(by=["Product", "Start"]).reset_index(drop=True)

    task_wait_times = []

    # Calculate waiting times between tasks for each product
    for product_id in df["Product"].unique():
        # Filter tasks for this specific product
        product_tasks = df[df["Product"] == product_id]

        # Calculate wait times between consecutive tasks for this product
        for i in range(1, len(product_tasks)):
            prev_task_end = product_tasks.iloc[i - 1]["Finish"]
            next_task_start = product_tasks.iloc[i]["Start"]

            # Calculate waiting time (in seconds) between tasks
            wait_time = next_task_start - prev_task_end
            if wait_time > 0:  # Only log positive wait times
                task_wait_times.append({
                    'Product': product_id,
                    'Waiting Time': wait_time,
                    'Previous Task End': prev_task_end,
                    'Next Task Start': next_task_start
                })
    
    return task_wait_times


def calculate_worker_queue_log(task_log_px):
    """
    Calculates and logs the queue length over time for each worker based on task completion timestamps.
    
    Args:
        task_log_px (list of dict): List containing logs of tasks, each with:
            - 'Task': Task name
            - 'Product': Product ID
            - 'Worker': Worker ID
            - 'Start': Start time of task
            - 'Finish': Finish time of task (timestamp of interest)
            - 'Queue Length': Queue length at the task's finish time

    Returns:
        dict: Dictionary where each worker_id maps to a list of dicts with:
              - 'Time': Timestamp at which the queue length was logged
              - 'Queue Length': Queue length at that time
    """
    worker_queue_log = defaultdict(list)

    # Process each task entry in the log
    for entry in task_log_px:
        worker_id = entry['Worker']
        timestamp = entry['Finish']
        queue_length = entry['Queue Length']
        
        # Append queue length log for the worker at the finish time of the task
        worker_queue_log[worker_id].append({
            'Time': timestamp,
            'Queue Length': queue_length
        })
    
    return worker_queue_log

def calculate_worker_idle_times(task_log_px):
    """
    Calculates individual idle times for each worker based on gaps between task assignments,
    including initial idle time (before first task) and final idle time (after last task).

    Args:
        task_log_px (list of dict): List containing logs of tasks, each with:
            - 'Task': Task name
            - 'Product': Product ID
            - 'Worker': Worker ID
            - 'Start': Start time of task
            - 'Finish': Finish time of task

    Returns:
        dict: A dictionary with individual idle times for each worker.
              Format: {worker_id: [{'Idle Time': idle_time, 'Previous Task End': prev_finish, 'Next Task Start': curr_start}, ...]}
    """
    # Convert task log to DataFrame (explicitly convert to list for proxy compatibility)
    df = pd.DataFrame(task_log_px)

    # Ensure Start and Finish are numeric (handles Timedelta correctly)
    df = _ensure_numeric_columns(df, ['Start', 'Finish'])

    # Sort by Worker and Start time to ensure correct ordering
    df = df.sort_values(by=["Worker", "Start"]).reset_index(drop=True)

    # Get simulation boundaries (global start and end times)
    simulation_start = df['Start'].min()
    simulation_end = df['Finish'].max()

    idle_times = {}

    # Calculate individual idle times for each worker
    for worker_id in df["Worker"].unique():
        # Filter tasks for this specific worker
        worker_tasks = df[df["Worker"] == worker_id]

        # Initialize idle time tracking for this worker
        idle_times[worker_id] = []

        # Initial idle time: from simulation start to worker's first task
        first_task_start = worker_tasks.iloc[0]["Start"]
        if first_task_start > simulation_start:
            idle_times[worker_id].append({
                'Idle Time': first_task_start - simulation_start,
                'Previous Task End': simulation_start,
                'Next Task Start': first_task_start
            })

        # Go through each task and calculate idle times between tasks
        for i in range(1, len(worker_tasks)):
            # Get the finish time of the previous task and start time of the current task
            prev_finish = worker_tasks.iloc[i - 1]["Finish"]
            curr_start = worker_tasks.iloc[i]["Start"]

            # Calculate idle time as the gap between tasks
            idle_time = curr_start - prev_finish
            if idle_time > 0:
                # Log each individual idle period
                idle_times[worker_id].append({
                    'Idle Time': idle_time,
                    'Previous Task End': prev_finish,
                    'Next Task Start': curr_start
                })

        # Final idle time: from worker's last task to simulation end
        last_task_finish = worker_tasks.iloc[-1]["Finish"]
        if last_task_finish < simulation_end:
            idle_times[worker_id].append({
                'Idle Time': simulation_end - last_task_finish,
                'Previous Task End': last_task_finish,
                'Next Task Start': simulation_end
            })

    return idle_times



def calculate_statistics(task_log_px):
    """
    Calculate key statistics from task, queue, and idle logs.
    
    Args:
        task_log_px (list of dict): List of task logs with task start and end times.
    Returns:
        dict: Calculated statistics including average wait times, queue lengths, and idle times.
    """
    worker_queue_log = calculate_worker_queue_log(task_log_px)
    worker_idle_times = calculate_worker_idle_times(task_log_px)
    # 1. Average Task Wait Time per Product
    product_wait_times = calculate_product_task_wait_times(task_log_px)
    avg_wait_times_per_product = {}
    for wait_time in product_wait_times:
        product = wait_time["Product"]
        avg_wait_times_per_product.setdefault(product, []).append(wait_time["Waiting Time"])
    
    avg_wait_times_per_product = {product: sum(times) / len(times) for product, times in avg_wait_times_per_product.items()}

    # 2. Average Queue Length per Worker
    avg_queue_length_per_worker = {}
    for worker_id, logs in worker_queue_log.items():
        queue_lengths = [entry["Queue Length"] for entry in logs]
        if queue_lengths:
            avg_queue_length_per_worker[worker_id] = sum(queue_lengths) / len(queue_lengths)
        else:
            avg_queue_length_per_worker[worker_id] = 0

    # 3. Total and Average Idle Time per Worker
    total_idle_time_per_worker = {
        worker_id: sum(entry['Idle Time'] for entry in idle_entries)
        for worker_id, idle_entries in worker_idle_times.items()
    }
    
    avg_idle_time_per_worker = {
    worker_id: (total_idle_time_per_worker[worker_id] / len(idle_entries) if len(idle_entries) > 0 else 0)
    for worker_id, idle_entries in worker_idle_times.items()
}
    # Aggregate results
    statistics = {
        "Average Wait Time per Product": avg_wait_times_per_product,
        "Average Queue Length per Worker": avg_queue_length_per_worker,
        "Total Idle Time per Worker": total_idle_time_per_worker,
        "Average Idle Time per Worker": avg_idle_time_per_worker,
    }
    
    return statistics



def generate_statistics_report(statistics):
    """
    Generates a structured report of calculated statistics for the factory simulation.
    
    Args:
        statistics (dict): Calculated statistics including average wait times,
                           queue lengths, and idle times.

    Returns:
        str: A formatted report summarizing each metric.
    """
    report_lines = []

    report_lines.append("\n----- Factory Simulation Statistics Report -----\n")

    # Average Wait Time per Product
    report_lines.append("Average Task Wait Time per Product:")
    for product, avg_wait_time in statistics["Average Wait Time per Product"].items():
        report_lines.append(f"  {product}: {avg_wait_time:.2f} seconds")

    # Average Queue Length per Worker
    report_lines.append("\nAverage Queue Length per Worker:")
    for worker_id, avg_queue_length in statistics["Average Queue Length per Worker"].items():
        report_lines.append(f"  Worker {worker_id}: {avg_queue_length:.2f} tasks")

    # Total Idle Time per Worker
    report_lines.append("\nTotal Idle Time per Worker:")
    for worker_id, total_idle_time in statistics["Total Idle Time per Worker"].items():
        report_lines.append(f"  Worker {worker_id}: {total_idle_time:.2f} seconds")

    # Average Idle Time per Worker
    report_lines.append("\nAverage Idle Time per Worker:")
    for worker_id, avg_idle_time in statistics["Average Idle Time per Worker"].items():
        report_lines.append(f"  Worker {worker_id}: {avg_idle_time:.2f} seconds")

    report_lines.append("\n----- End of Report -----")

    # Join all lines into a single string separated by newlines
    return "\n".join(report_lines)




def validate_configuration(workers, products):
    """
    Validate that all tasks required by products can be performed by at least one worker.
    This check should be run BEFORE starting the simulation to prevent freezing.

    Args:
        workers: List of Worker objects
        products: List of Product objects

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Collect all tasks that workers can perform
    all_capable_tasks = set()
    for worker in workers:
        all_capable_tasks.update(worker.capable_tasks)

    # Check each product's task sequence
    unperformable_tasks = {}
    for product in products:
        for task in product.task_sequence:
            task_name = task.name if hasattr(task, 'name') else task
            if task_name not in all_capable_tasks:
                if task_name not in unperformable_tasks:
                    unperformable_tasks[task_name] = []
                unperformable_tasks[task_name].append(product.product_id)

    if unperformable_tasks:
        error_lines = ["Configuration Error: The following tasks have no capable workers:"]
        for task_name, product_ids in unperformable_tasks.items():
            error_lines.append(f"  - Task '{task_name}' (required by products: {', '.join(product_ids)})")
        error_lines.append("\nPlease assign at least one worker to each task, or remove the task from products.")
        return False, "\n".join(error_lines)

    return True, None


def validate_task_log(task_log_px, workers, products):
    report = []
    task_log = copy.deepcopy(task_log_px)
    worker_tasks = {worker.worker_id: worker.capable_tasks for worker in workers}
    product_tasks = {product.product_id: product.task_sequence for product in products}

    worker_end_times = {worker.worker_id: 0 for worker in workers}
    
    # Dictionary to track product task completion order and prevent duplicate tasks
    product_task_progress = {product.product_id: [] for product in products}

    for entry in task_log:
        worker_id = entry['Worker']
        product_id = str(entry['Product'].split()[-1])
        task_name = entry['Task']
        start_time = entry['Start']
        finish_time = entry['Finish']

        # 1) Check if the worker is performing an allowed task
        if task_name not in worker_tasks[worker_id]:
            report.append(f"Worker {worker_id} performed task '{task_name}' which is not in their capable tasks.")

        # 2) Check if the task corresponds to the product's input tasks
        expected_tasks = [task.name for task in product_tasks[product_id]]
        if task_name not in expected_tasks:
            report.append(f"Product {product_id} has an unexpected task '{task_name}' in the log.")

        # 3) Check for overlapping tasks for the worker
        if start_time < worker_end_times[worker_id]:
            report.append(f"Worker {worker_id} started task '{task_name}' at {start_time} "
                          f"before completing a previous task at {worker_end_times[worker_id]}.")

        # Update the worker's end time after completing the task
        worker_end_times[worker_id] = finish_time

        # 4) Check if tasks are completed in the correct order and only once for each product
        current_sequence = [task.name for task in product_tasks[product_id]]
        completed_tasks = product_task_progress[product_id]
        
        # Check task order and duplicates
        if current_sequence[len(completed_tasks)] == task_name:
            completed_tasks.append(task_name)
        elif task_name in completed_tasks:
            report.append(f"Product {product_id} has task '{task_name}' completed more than once.")
        else:
            report.append(f"Product {product_id} completed task '{task_name}' out of order.")

    if not report:
        return "All checks passed successfully."
    return "\n".join(report)


