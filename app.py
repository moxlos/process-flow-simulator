#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Flow Simulator - Streamlit Application

A discrete-event simulation tool for modeling and analyzing sequential workflows.
Users can configure workers, products, and tasks with various time distributions.
Applicable to healthcare, manufacturing, finance, government, and other industries.
"""

import streamlit as st
from simulation import Task, Worker, Product
from simulation import factory_simulation, uniform_dist, normal_dist, exponential_dist
from logging_module import task_log_px, logs_str, reset_logs
from utils import validate_task_log, validate_configuration, calculate_metrics, generate_report, calculate_statistics, generate_statistics_report
from utils import plot_gantt_px_st, plot_worker_util_st, plot_worker_cplt_st, plot_task_times_st, plot_product_completion_st, product_task_completion_times
import simpy
import logging
import json


st.set_page_config(layout="wide")


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "task_settings" not in st.session_state:
        st.session_state["task_settings"] = {}
    if "workers" not in st.session_state:
        st.session_state["workers"] = []
    if "products" not in st.session_state:
        st.session_state["products"] = []
    if "env" not in st.session_state:
        st.session_state["env"] = None


def run_simulation_with_reset():
    """Reset logs before running a new simulation."""
    reset_logs()


def main():
    st.title("Process Flow Simulator")

    # Initialize session state
    init_session_state()

    # Setup logging
    logging.basicConfig(filename="factory_simulation.log", level=logging.INFO, format='%(asctime)s %(message)s')

    # Create simpy environment
    env = simpy.Environment()
    st.session_state["env"] = env

    # Create a sidebar with navigation options
    page = st.sidebar.radio("Select a page", ["Configuration", "Simulation",
                                              "Logs", "Report", "Analytics"])

    # Display content based on the selected page
    if page == "Configuration":
        config_method = st.radio("Choose Configuration Method", ["UI", "From File"])
        if config_method == "From File":
            uploaded_file = st.sidebar.file_uploader("Upload Configuration File", type="json")
            if uploaded_file:
                try:
                    task_settings, workers, products = load_configurations_from_file(uploaded_file, env, task_log_px)
                    st.session_state["task_settings"] = task_settings
                    st.session_state["workers"] = workers
                    st.session_state["products"] = products
                except Exception as e:
                    st.error(f"Failed to load configuration: {e}")
            else:
                st.warning("No file selected. Please upload a configuration file.")
        elif config_method == "UI":
            workers, products, task_settings = submit_inputs(env)
            st.session_state["task_settings"] = task_settings
            st.session_state["workers"] = workers
            st.session_state["products"] = products
            st.success("UI parameters submitted successfully.")

        if st.sidebar.button("Run Simulation"):
            workers = st.session_state["workers"]
            products = st.session_state["products"]

            if not workers or not products:
                st.error("Workers and products are not configured. Please configure them first.")
            else:
                # Validate configuration before running
                is_valid, error_msg = validate_configuration(workers, products)
                if not is_valid:
                    st.error(error_msg)
                else:
                    run_simulation_with_reset()
                    run_simulation(env, workers, products)
                    st.success("Simulation completed successfully!")

        # Save Configuration Button
        if st.sidebar.button("Save Configuration"):
            if st.session_state["task_settings"] and st.session_state["workers"] and st.session_state["products"]:
                save_configurations(
                    st.session_state["task_settings"],
                    st.session_state["workers"],
                    st.session_state["products"]
                )
            else:
                st.warning("Please configure tasks, workers, and products before saving.")
    
            
    elif page == "Simulation":
        st.header("Run Simulation")
        # st.write(workers)
        # st.write(products)
        # st.write(task_log_px)
        fig = plot_gantt_px_st(task_log_px)
        st.plotly_chart(fig)
            
    elif page == "Logs":
        #st.text_area("Logs", "\n".join(logs_str), height=500)
        st.text("\n".join(logs_str))
    elif page == "Report":
        st.subheader("Simulation Report")
        sim_reports(task_log_px)
    elif page == "Analytics":
        sim_plots(task_log_px)



def save_configurations(task_settings, workers, products):
    config = {
        "tasks": {name: {"distribution": task.time_distribution.__name__, "parameters": task.params} for name, task in task_settings.items()},
        "workers": [{"id": worker.worker_id, "tasks": worker.capable_tasks, "max_queue_size": worker.max_queue_size} for worker in workers],
        "products": [{"id": product.product_id, "tasks": [task.name for task in product.task_sequence]} for product in products],
    }
    
    with open("data/configurations.json", "w") as f:
        json.dump(config, f, indent=4)
    st.success("Configuration saved successfully.")

def load_configurations_from_file(file, env, task_log_px):
    if file is not None:
        config = json.load(file)
        
        # Recreate tasks
        task_settings = {}
        dist_map = {"normal_dist": normal_dist, "uniform_dist": uniform_dist, "exponential_dist": exponential_dist}
        for name, details in config["tasks"].items():
            dist_func = dist_map[details["distribution"]]
            task_settings[name] = Task(name, dist_func, tuple(details["parameters"]))
        
        # Recreate workers
        workers = [Worker(env, worker["id"], worker["tasks"], task_log_px, worker.get("max_queue_size", 3)) for worker in config["workers"]]
        
        # Recreate products
        products = [Product(prod["id"], [task_settings[task_name] for task_name in prod["tasks"]], env) for prod in config["products"]]
        
        st.success("Configuration loaded successfully.")
        return task_settings, workers, products
    else:
        st.warning("Please upload a configuration file.")
        return None, None, None



def submit_inputs(env):
    # Task Configuration
    task_names = ["Funding Approval", "Application Inspection", "Payment"]
    task_settings = {}
    
    for task_name in task_names:
        st.subheader(f"Task: {task_name}")
        col_dist, col_param1, col_param2 = st.columns([2, 1, 1])
        with col_dist:
            distribution = st.selectbox(f"Distribution", ["Normal", "Uniform", "Exponential"], key=f"{task_name}_dist")
        if distribution == "Normal":
            dist_func = normal_dist
            with col_param1:
                mean = st.number_input(f"Mean", value=10.0, key=f"{task_name}_mean")
            with col_param2:
                stddev = st.number_input(f"Std Dev", value=2.0, key=f"{task_name}_stddev")
            parameters = (mean, stddev)
        elif distribution == "Uniform":
            dist_func = uniform_dist
            with col_param1:
                min_val = st.number_input(f"Min", value=5.0, key=f"{task_name}_min")
            with col_param2:
                max_val = st.number_input(f"Max", value=10.0, key=f"{task_name}_max")
            parameters = (min_val, max_val)
        elif distribution == "Exponential":
            dist_func = exponential_dist
            with col_param1:
                scale = st.number_input(f"Scale", value=8.0, key=f"{task_name}_scale")
            parameters = (scale,)

        task_settings[task_name] = Task(task_name, dist_func, parameters)
    
    # Worker Configuration
    st.header("Configure Workers")
    col_num_workers, _ = st.columns([1, 3])
    with col_num_workers:
        num_workers = st.number_input("Number of Workers", min_value=1, max_value=10, value=2)
    workers = []

    for i in range(num_workers):
        st.subheader(f"Worker {str(i+1).zfill(2)}")
        col1, col2 = st.columns([3, 1])
        with col1:
            worker_tasks = st.multiselect(f"Tasks Worker {str(i+1).zfill(2)} Can Perform", task_names,
                                          default=task_names, key=f"worker_{str(i).zfill(2)}_tasks")
        with col2:
            max_queue = st.number_input(f"Max Queue Size", min_value=1, max_value=20, value=3,
                                        key=f"worker_{str(i).zfill(2)}_queue")
        workers.append(Worker(env, str(i+1).zfill(2), worker_tasks, task_log_px, max_queue))
    
    # Product Configuration
    st.header("Configure Products/Services")
    col_num_products, _ = st.columns([1, 3])
    with col_num_products:
        num_products = st.number_input("Number of Products", min_value=1, max_value=100, value=2)
    products = []

    for i in range(num_products):
        col_id, col_tasks = st.columns([1, 4])
        with col_id:
            product_id = st.text_input(f"Product {str(i+1).zfill(2)} ID", value=f"{str(i+1).zfill(2)}", key=f"product_{str(i).zfill(2)}_id")
        with col_tasks:
            product_tasks = st.multiselect(f"Tasks for Product {str(i+1).zfill(2)}", task_names,
                                           default=task_names, key=f"product_{str(i).zfill(2)}_tasks")
        products.append(Product(product_id, [task_settings[task] for task in product_tasks], env))
    
    # Display Summary of Configuration
    st.header("Summary of Configuration")
    
    # Task Summary
    st.subheader("Task Settings")
    for task_name, task in task_settings.items():
        st.write(f"Task: {task_name}, Distribution: {task.time_distribution}, Parameters: {task.params}")
    
    # Worker Summary
    st.subheader("Workers")
    for worker in workers:
        st.write(f"Worker ID: {worker.worker_id}, Capable Tasks: {worker.capable_tasks}, Max Queue: {worker.max_queue_size}")
    
    # Product Summary
    st.subheader("Products/Services")
    for product in products:
        st.write(f"Product ID: {product.product_id}, Task Sequence: {[task.name for task in product.task_sequence]}")
    
    st.write("Simulation can be run with the configured parameters using this setup.")
    
    
    return workers, products, task_settings

def run_simulation(env, workers, products):
    # Run the simulation
    #print(workers)
    products_cp = products.copy()
    #print(task_log_px)
    env.process(factory_simulation(env, workers, products))
    env.run()
    # print(workers)
    res = validate_task_log(task_log_px, workers, products_cp)
    # Show success message if the process completed
    if res == "All checks passed successfully.":
        st.sidebar.success(res)
    else:
        st.sidebar.error(res)
        

def sim_reports(task_log_px):
    metrics, worker_utilization, product_completion_times, _,_ = calculate_metrics(task_log_px)  # Obtain metrics dictionary
    report_str = generate_report(metrics)  # Generate and print the report
    
    statistics = calculate_statistics(task_log_px)
    report_str += "\n\n" + generate_statistics_report(statistics)
    st.markdown(report_str)

def sim_plots(task_log_px):
    #plots
    metrics, worker_utilization, product_completion_times, product_completion_times_det,task_completion_times = calculate_metrics(task_log_px)
    
    fig_1 = plot_worker_util_st(worker_utilization)
    fig_2 = plot_worker_cplt_st(metrics['Average Task Completion Time by Worker'])
    fig_3 = plot_task_times_st(metrics['Average Time per Task'])
    fig_4 = plot_product_completion_st(product_completion_times)
    fig_5, fig_6 = product_task_completion_times(product_completion_times_det,task_completion_times)
    col1, col2= st.columns(2)
    with col1:
        st.plotly_chart(fig_1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_2)
        
    col3, col4= st.columns(2)
    with col3:
        st.plotly_chart(fig_3, use_container_width=True)
    with col4:
        st.plotly_chart(fig_4)
    
    col5, col6= st.columns(2)
    with col5:
        st.plotly_chart(fig_5, use_container_width=True)
    with col6:
        st.plotly_chart(fig_6)
    

if __name__ == "__main__":
    main()   