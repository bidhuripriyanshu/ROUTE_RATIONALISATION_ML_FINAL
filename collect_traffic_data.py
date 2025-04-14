import os
import sys
import traci
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def collect_traffic_data(sumocfg_file, simulation_time=3600, collection_interval=5):
    print("Starting traffic data collection...")
    
    try:
        # Start SUMO simulation
        sumo_binary = "sumo"
        sumo_cmd = [sumo_binary, "-c", sumocfg_file]
        traci.start(sumo_cmd)
        
        # Initialize data storage
        traffic_data = []
        edges = traci.edge.getIDList()
        
        # Pre-allocate numpy arrays for better performance
        n_steps = simulation_time // collection_interval
        n_edges = len(edges)
        
        # Create arrays for each metric
        timestamps = np.zeros(n_steps, dtype=int)
        occupancies = np.zeros((n_steps, n_edges))
        speeds = np.zeros((n_steps, n_edges))
        vehicle_counts = np.zeros((n_steps, n_edges))
        waiting_times = np.zeros((n_steps, n_edges))
        
        # Create progress bar
        pbar = tqdm(total=simulation_time, desc="Collecting traffic data")
        
        # Collect data for each simulation step
        step_idx = 0
        for step in range(simulation_time):
            traci.simulationStep()
            
            # Collect data only at specified intervals
            if step % collection_interval == 0:
                timestamps[step_idx] = step
                
                for edge_idx, edge in enumerate(edges):
                    try:
                        occupancies[step_idx, edge_idx] = traci.edge.getLastStepOccupancy(edge)
                        speeds[step_idx, edge_idx] = traci.edge.getLastStepMeanSpeed(edge)
                        vehicle_counts[step_idx, edge_idx] = traci.edge.getLastStepVehicleNumber(edge)
                        waiting_times[step_idx, edge_idx] = traci.edge.getWaitingTime(edge)
                    except traci.exceptions.TraCIException as e:
                        print(f"Warning: Could not get data for edge {edge}: {str(e)}")
                        continue
                
                step_idx += 1
            
            pbar.update(1)
        
        pbar.close()
        traci.close()
        
        # Create DataFrame more efficiently
        data_dict = {
            'timestamp': np.repeat(timestamps, n_edges),
            'edge': np.tile(edges, n_steps),
            'occupancy': occupancies.flatten(),
            'speed': speeds.flatten(),
            'vehicle_count': vehicle_counts.flatten(),
            'waiting_time': waiting_times.flatten()
        }
        
        df = pd.DataFrame(data_dict)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"traffic_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        if 'traci' in locals():
            traci.close()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python collect_traffic_data.py <sumocfg_file>")
        sys.exit(1)
        
    sumocfg_file = sys.argv[1]
    collect_traffic_data(sumocfg_file) 