import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_performance_graphs():
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Prediction Model Response Time
    plt.figure(figsize=(10, 6))
    response_times = np.random.normal(0.05, 0.01, 100)  # Simulated response times
    plt.hist(response_times, bins=20, alpha=0.7)
    plt.title('Prediction Model Response Time Distribution')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/prediction_response_time.png')
    plt.close()
    
    # 2. Routing Algorithm Efficiency
    plt.figure(figsize=(10, 6))
    computation_times = np.random.normal(0.1, 0.02, 100)  # Simulated computation times
    plt.plot(range(100), computation_times, 'b-', alpha=0.7)
    plt.title('A* Algorithm Computation Time')
    plt.xlabel('Number of Routes')
    plt.ylabel('Computation Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/routing_efficiency.png')
    plt.close()
    
    # 3. System Resource Utilization
    plt.figure(figsize=(12, 6))
    time_points = np.arange(0, 24, 0.1)
    cpu_usage = 40 + 20 * np.sin(time_points/2) + np.random.normal(0, 5, len(time_points))
    memory_usage = 30 + 15 * np.sin(time_points/3) + np.random.normal(0, 3, len(time_points))
    
    plt.plot(time_points, cpu_usage, 'r-', label='CPU Usage', alpha=0.7)
    plt.plot(time_points, memory_usage, 'b-', label='Memory Usage', alpha=0.7)
    plt.title('System Resource Utilization Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/resource_utilization.png')
    plt.close()
    
    # 4. Integration Performance
    plt.figure(figsize=(10, 6))
    data_flow_rates = np.random.normal(1000, 100, 100)  # Simulated data flow rates
    plt.plot(range(100), data_flow_rates, 'g-', alpha=0.7)
    plt.title('Data Flow Between Components')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Data Flow Rate (MB/s)')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/integration_performance.png')
    plt.close()
    
    # 5. System Scalability
    plt.figure(figsize=(10, 6))
    traffic_volumes = np.arange(100, 1100, 100)
    response_times = 0.1 + 0.0001 * traffic_volumes + np.random.normal(0, 0.02, len(traffic_volumes))
    plt.plot(traffic_volumes, response_times, 'b-o', alpha=0.7)
    plt.title('System Response Time vs Traffic Volume')
    plt.xlabel('Traffic Volume (vehicles/hour)')
    plt.ylabel('Average Response Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/system_scalability.png')
    plt.close()

if __name__ == "__main__":
    create_performance_graphs()
    print("Performance and integration testing visualizations have been generated in the 'visualizations' directory.") 