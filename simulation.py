#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 2024

@author: lefteris

@subject: change worker class inputs
"""

import simpy
import random
import numpy as np
import logging
from logging_module import task_log_px, logs_str, reset_logs
from utils import validate_task_log, validate_configuration, calculate_metrics, generate_report, calculate_statistics, generate_statistics_report
from utils import plot_gantt_px, plot_worker_util, plot_worker_cplt, plot_task_times, plot_product_completion





# Define task class with a time distribution
class Task:
    def __init__(self, name, time_distribution, params):
        self.name = name
        self.time_distribution = time_distribution
        self.params = params

    def get_time(self):
        return self.time_distribution(*self.params)

# Define worker class in simpy
class Worker:
    def __init__(self, env, worker_id, capable_tasks,task_log_px, max_queue_size=3):
        self.env = env
        self.worker_id = worker_id
        self.capable_tasks = capable_tasks  # Tasks the worker can perform
        self.max_queue_size = max_queue_size  # Max tasks a worker can queue
        self.task_queue = simpy.Store(env)  # Queue to hold tasks for this worker
        # Start the worker's task processing loop
        self.env.process(self.process_tasks())
        self.active_tasks = []  # Track active tasks in the worker's queue
        self.task_log_px = task_log_px
    def perform_task(self, task, product):
        """
        Worker performs a given task, which takes time based on the task's distribution.
        """
        with product.lock.request() as req:
            yield req  # Wait until access to product is granted
            task_start = self.env.now
            task_time = task.get_time()
            logging.info(f"Worker {self.worker_id} starting task {task.name} on product {product.product_id} (time: {task_time:.2f})")
            logs_str.append(f"Worker {self.worker_id} starting task {task.name} on product {product.product_id} (time: {task_time:.2f})")
            yield self.env.timeout(task_time)  # Wait for task time duration
            logging.info(f"Worker {self.worker_id} completed task {task.name} on product {product.product_id}")
            logs_str.append(f"Worker {self.worker_id} completed task {task.name} on product {product.product_id}")
            task_end = self.env.now
            # Log task details in task_log for Plotly Gantt
            self.task_log_px.append({
                'Task': task.name,
                'Product': f"Product {product.product_id}",
                'Worker': self.worker_id,
                'Start': float(task_start),
                'Finish': float(task_end),
                'Queue Length': len(self.task_queue.items)
            })
            # IMPORTANT: Increment task index BEFORE setting in_progress to False
            # This prevents race condition where factory assigns the same task again
            product.complete_current_task()
            product.in_progress = False  # Mark product as ready for the next task
            self.active_tasks.remove((task, product))  # Remove task from active list

            
        
    def can_do_task(self, task):
        return task.name in self.capable_tasks
    
    def process_tasks(self):
        while True:
            # Get the next task from the queue
            task, product = yield self.task_queue.get()
            yield self.env.process(self.perform_task(task, product))

            
    def is_available(self):
            return len(self.task_queue.items) < self.max_queue_size

# Define product class
class Product:
    def __init__(self, product_id, task_sequence,env):
        self.product_id = product_id
        self.task_sequence = task_sequence
        self.current_task_index = 0
        self.lock = simpy.Resource(env, capacity=1)  # Ensure exclusive access per product
        self.in_progress = False  # Flag to indicate if the product is currently assigned to a task
    def get_next_task(self):
        if self.current_task_index < len(self.task_sequence):
            task = self.task_sequence[self.current_task_index]
            #self.current_task_index += 1
            return task
        return None
    def complete_current_task(self):
        """
        Move to the next task in the sequence when the current task is done.
        """
        self.current_task_index += 1


def worker_select_by_queue(task,workers, product):
    # Sort workers by availability, capability, and queue length
    available_workers = sorted(
        (w for w in workers if w.is_available() and w.can_do_task(task) and (task, product) not in w.active_tasks),
        key=lambda w: len(w.task_queue.items)
    )
    return available_workers[0] if available_workers else None

def worker_selct_random(task,workers, product):

    capable_workers = [w for w in workers if w.is_available() and w.can_do_task(task) and (task, product) not in w.active_tasks]
    return random.choice(capable_workers) if capable_workers else None
    
        

# Simulation function
def factory_simulation(env, workers, products, update_interval=5):
    while products:
        for product in products[:]:
            task = product.get_next_task()
            if task is None:
                logging.info(f"Product {product.product_id} is completed.")
                logs_str.append(f"Product {product.product_id} is completed.")
                products.remove(product)
                continue
            
            # Skip the product if it's currently in progress
            if product.in_progress:
                continue
            
            available_worker = worker_select_by_queue(task, workers, product)
            if available_worker:
                # Send the task to the worker's queue
                logging.info(f"Worker {available_worker.worker_id} gets task {task.name} for product {product.product_id} in queue")
                logs_str.append(f"Worker {available_worker.worker_id} gets task {task.name} for product {product.product_id} in queue")
                available_worker.task_queue.put((task, product))
                available_worker.active_tasks.append((task, product))  # Track as an active task
                product.in_progress = True  # Mark product as in-progress to prevent reassignment
                        

        yield env.timeout(1)  # Wait before rechecking available workers and tasks
    
    
# Define some distributions for task times
def uniform_dist(low, high):
    return random.uniform(low, high)

def normal_dist(mean, stddev):
    return np.random.normal(mean, stddev)

def exponential_dist(scale):
    return np.random.exponential(scale)

# Example setup
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(filename="factory_simulation.log", level=logging.INFO, format='%(asctime)s %(message)s')

    # Reset logs for clean state
    reset_logs()

    # Create simpy environment
    env = simpy.Environment()

    # Create tasks with different time distributions
    task1 = Task("Assemble", normal_dist, (10, 2))    # Normal distribution: mean=10, stddev=2
    task2 = Task("Inspect", uniform_dist, (5, 10))    # Uniform distribution: min=5, max=10
    task3 = Task("Package", exponential_dist, (8,))   # Exponential distribution: scale=8

    # Create workers, each capable of performing a subset of tasks
    workers = [Worker(env, 1, ["Assemble", "Inspect"], task_log_px),
               Worker(env, 2, ["Package"],task_log_px),
               Worker(env, 3, ["Inspect", "Package"], task_log_px)]

    # Create products, each with a sequence of tasks
    products = [
        Product('01', [task1, task2, task3], env),
        Product('02', [task2, task3, task1], env),
        Product('03', [task3, task1, task2], env),
        Product('04', [task1, task3, task2], env),
        Product('05', [task2, task1, task3], env),
        Product('06', [task3, task2, task1], env),
        Product('07', [task1, task2, task3], env),
        Product('08', [task2, task3, task1], env),
        Product('09', [task3, task1, task2], env),
        Product('10', [task1, task3, task2], env)
    ]
    
    products_cp = products.copy()

    # Validate configuration before running
    is_valid, error_msg = validate_configuration(workers, products_cp)
    if not is_valid:
        print(error_msg)
        exit(1)

    # Run the simulation
    env.process(factory_simulation(env, workers, products))
    env.run()
    plot_gantt_px(task_log_px)
    print(validate_task_log(task_log_px, workers, products_cp))
    metrics, worker_utilization, product_completion_times, product_completion_times_det,_ = calculate_metrics(task_log_px)  # Obtain metrics dictionary
    report_str = generate_report(metrics)  # Generate and print the report
    
    statistics = calculate_statistics(task_log_px)
    report_str += "\n\n" + generate_statistics_report(statistics)
    print(report_str)

    #plots
    plot_worker_util(worker_utilization)
    plot_worker_cplt(metrics['Average Task Completion Time by Worker'])
    plot_task_times(metrics['Average Time per Task'])
    plot_product_completion(product_completion_times)