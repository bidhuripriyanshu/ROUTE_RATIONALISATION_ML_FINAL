import os
import sys
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
from collections import deque
from colorama import init, Fore, Back, Style
from datetime import datetime

# Initialize colorama
init()

class TrafficPredictor:
    def __init__(self, model_path, scaler_path, sequence_length=10, prediction_horizon=5):
        self.model = load_model(model_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Load scaler parameters
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        self.scaler = MinMaxScaler()
        self.scaler.scale_ = np.array(scaler_params['scale_'])
        self.scaler.min_ = np.array(scaler_params['min_'])
        
        # Initialize data buffer with numpy array for better performance
        self.data_buffer = {}  # Dictionary to store data buffer for each edge
        self.buffer_indices = {}  # Dictionary to store buffer index for each edge
        self.buffer_full = {}  # Dictionary to track if buffer is full for each edge
    
    def predict(self, current_data_dict):
        """
        Predict future congestion for each edge
        
        Args:
            current_data_dict: Dictionary mapping edge IDs to current congestion values
            
        Returns:
            Dictionary mapping edge IDs to predicted future congestion values
        """
        predictions = {}
        
        for edge, current_data in current_data_dict.items():
            # Initialize buffer for this edge if not exists
            if edge not in self.data_buffer:
                self.data_buffer[edge] = np.zeros(self.sequence_length)
                self.buffer_indices[edge] = 0
                self.buffer_full[edge] = False
            
            # Update buffer
            self.data_buffer[edge][self.buffer_indices[edge]] = current_data
            self.buffer_indices[edge] = (self.buffer_indices[edge] + 1) % self.sequence_length
            
            if not self.buffer_full[edge]:
                if self.buffer_indices[edge] == 0:
                    self.buffer_full[edge] = True
                else:
                    # For edges with not enough history, use current value as prediction
                    predictions[edge] = current_data
                    continue
            
            # Prepare data for prediction
            scaled_data = self.scaler.transform(self.data_buffer[edge].reshape(-1, 1))
            X = scaled_data.reshape(1, self.sequence_length, 1)
            
            try:
                # Make prediction
                prediction = self.model.predict(X, verbose=0)  # Disable verbose output
                predictions[edge] = self.scaler.inverse_transform(prediction)[0][0]
                
                # Ensure prediction is not negative
                predictions[edge] = max(0, predictions[edge])
                
                # Add some randomness to make predictions more realistic
                # This helps simulate the uncertainty in traffic prediction
                predictions[edge] = min(1.0, predictions[edge] * (1 + np.random.normal(0, 0.1)))
            except Exception as e:
                print(f"{Fore.RED}Error predicting for edge {edge}: {str(e)}{Style.RESET_ALL}")
                # Fallback to current value if prediction fails
                predictions[edge] = current_data
        
        return predictions

class AStarRouter:
    def __init__(self):
        self.graph = {}
        self.edge_cache = {}  # Cache for edge connections
        self.valid_connections = set()  # Cache for valid edge connections
        
    def build_graph(self, edges):
        print(f"{Fore.CYAN}Building road network graph...{Style.RESET_ALL}")
        
        # Pre-allocate graph structure
        self.graph = {edge: [] for edge in edges}
        
        # Get all edges in the network once
        all_edges = set(traci.edge.getIDList())
        
        # First pass: collect all valid connections
        for edge in edges:
            try:
                # Get the number of lanes for this edge
                num_lanes = traci.edge.getLaneNumber(edge)
                
                # For each lane, get its connections
                for lane_idx in range(num_lanes):
                    lane_id = f"{edge}_{lane_idx}"
                    try:
                        # Get the connections from this lane
                        connections = traci.lane.getLinks(lane_id)
                        
                        # Store valid connections
                        for connection in connections:
                            target_lane = connection[0]
                            target_edge = target_lane.split('_')[0]
                            
                            if target_edge != edge and target_edge in all_edges:
                                self.valid_connections.add((edge, target_edge))
                    except traci.exceptions.TraCIException:
                        continue
                
            except traci.exceptions.TraCIException:
                continue
        
        # Second pass: build graph using only valid connections
        for edge in edges:
            self.graph[edge] = [target for source, target in self.valid_connections if source == edge]
        
        print(f"{Fore.GREEN}Graph built with {len(self.valid_connections)} valid connections{Style.RESET_ALL}")
    
    def is_valid_connection(self, source, target):
        """Check if there is a valid connection between two edges"""
        return (source, target) in self.valid_connections
    
    def heuristic(self, a, b, congestion_data):
       
        try:
            
            num_lanes = traci.edge.getLaneNumber(a)
           
            lane_factor = 1.0 / max(1, num_lanes)
            
            congestion = congestion_data.get(a, 0)
            
            base_cost = 10.0 if ':' in a or '#' in a else 5.0
            
            return base_cost * (1 + congestion * 2) * lane_factor
            
        except traci.exceptions.TraCIException:
            # If we can't get lane info, use a simpler heuristic
            congestion = congestion_data.get(a, 0)
            base_cost = 10.0 if ':' in a or '#' in a else 5.0
            return base_cost * (1 + congestion * 2)
    
    def get_path(self, start, goal, congestion_data):
        if start not in self.graph or goal not in self.graph:
            return []
            
        # Verify start and goal are connected
        if not self.is_valid_connection(start, self.graph[start][0]) if self.graph[start] else False:
            return []
            
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = frontier.pop(0)[1]
            
            if current == goal:
                break
                
            for next_edge in self.graph[current]:
                # Verify connection is valid
                if not self.is_valid_connection(current, next_edge):
                    continue
                    
                new_cost = cost_so_far[current] + self.heuristic(current, next_edge, congestion_data)
                
                if next_edge not in cost_so_far or new_cost < cost_so_far[next_edge]:
                    cost_so_far[next_edge] = new_cost
                    priority = new_cost + self.heuristic(next_edge, goal, congestion_data)
                    frontier.append((priority, next_edge))
                    frontier.sort()
                    came_from[next_edge] = current
        
        # Build path and verify all connections are valid
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        # Verify all connections in the path are valid
        if len(path) < 2:
            return []
            
        for i in range(len(path) - 1):
            if not self.is_valid_connection(path[i], path[i + 1]):
                return []
                
        return path if path[0] == start else []

def print_route_info(flow_source, flow_dest, old_route, new_route, old_congestion, new_congestion):
    """Print route information with color-coded congestion values"""
    # ANSI escape codes for colors
    GOLD = "\033[38;2;255;215;0m"  # Golden yellow for headers
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    
    # Print original route with congestion levels
    print(f"\n{GOLD}Original Route with Congestion:{RESET}")
    for edge in old_route:
        congestion = old_congestion.get(edge, 0) * 100  # Convert to percentage
        # Color code: red for high congestion (>30%), green for low congestion
        color = RED if congestion > 30 else GREEN
        print(f"  {edge}: {color}{congestion:.1f}%{RESET}")
    
    # Print new route with congestion levels
    print(f"\n{GOLD}New Route with Congestion:{RESET}")
    for edge in new_route:
        congestion = new_congestion.get(edge, 0) * 100  # Convert to percentage
        # Color code: red for high congestion (>30%), green for low congestion
        color = RED if congestion > 30 else GREEN
        print(f"  {edge}: {color}{congestion:.1f}%{RESET}")

def run_simulation(sumocfg_file, model_path, scaler_path, simulation_time=7200): 
    print(f"{Fore.CYAN}Starting SUMO simulation with prediction-based flow rerouting...{Style.RESET_ALL}")
    predictor = TrafficPredictor(model_path, scaler_path)
    router = AStarRouter()
    sumo_binary = "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--no-step-log", "true", "--no-warnings", "true", "--scale", "0.1"]
    traci.start(sumo_cmd)
    edges = traci.edge.getIDList()
    router.build_graph(edges)
    n_steps = simulation_time // 100 + 1
    actual_congestion = np.zeros(n_steps)
    predicted_congestion = np.zeros(n_steps)
    timestamps = np.zeros(n_steps, dtype=int)
    step_idx = 0
    STEP_SIZE = 1  
    PREDICTION_INTERVAL = 20
    REROUTING_INTERVAL = 40
    VISUALIZATION_INTERVAL = 100
    total_flows = 0
    rerouted_flows = 0
    high_congestion_events = 0
    skipped_flows = 0
    rerouted_flow_keys = set()
    try:
        for step in range(0, simulation_time, STEP_SIZE):
            try:
                traci.simulationStep()
                current_congestion = {edge: traci.edge.getLastStepOccupancy(edge) for edge in edges}
                avg_congestion = np.mean(list(current_congestion.values()))
                
                # Store actual congestion for visualization
                if step % VISUALIZATION_INTERVAL == 0:
                    actual_congestion[step_idx] = avg_congestion
                    timestamps[step_idx] = step
                    step_idx += 1
                
                # Predict future congestion and reroute flows less frequently
                if step % PREDICTION_INTERVAL == 0:
                    print(f"\n{Fore.CYAN}Making predictions at step {step}...{Style.RESET_ALL}")
                    predicted_congestion_by_edge = predictor.predict(current_congestion)
                    
                    # Calculate average predicted congestion for visualization
                    if step % VISUALIZATION_INTERVAL == 0:
                        valid_predictions = [p for p in predicted_congestion_by_edge.values() if p is not None]
                        if valid_predictions:
                            predicted_congestion[step_idx-1] = np.mean(valid_predictions)
                    
                    # Check for rerouting less frequently
                    if step % REROUTING_INTERVAL == 0:
                        # Find edges with high predicted congestion
                        high_congestion_edges = {edge: pred for edge, pred in predicted_congestion_by_edge.items() 
                                              if pred is not None and pred > 0.3}
                        
                        if high_congestion_edges:
                            high_congestion_events += 1
                            print(f"\n{Fore.YELLOW}Step {step}: High congestion predicted for {len(high_congestion_edges)} edges{Style.RESET_ALL}")
                            
                            # Get all vehicle flows
                            vehicles = traci.vehicle.getIDList()
                            if vehicles:
                                # Group vehicles by source-destination pairs
                                flow_groups = {}
                                for vehicle in vehicles:
                                    try:
                                        current_edge = traci.vehicle.getRoadID(vehicle)
                                        if current_edge.startswith(":"):  # Skip vehicles on internal edges
                                            continue
                                        route = traci.vehicle.getRoute(vehicle)
                                        source = route[0]
                                        destination = route[-1]
                                        flow_key = (source, destination)
                                        
                                        # Skip if this flow has already been rerouted
                                        if flow_key in rerouted_flow_keys:
                                            continue
                                        
                                        if flow_key not in flow_groups:
                                            flow_groups[flow_key] = {
                                                'vehicles': [],
                                                'current_route': route
                                            }
                                        flow_groups[flow_key]['vehicles'].append(vehicle)
                                    except:
                                        continue
                                
                                # Process each flow group
                                rerouted_in_this_step = 0
                                skipped_in_this_step = 0
                            
                                for (source, dest), flow_data in flow_groups.items():
                                    current_route = flow_data['current_route']
                                    if any(edge.startswith(":") for edge in current_route):
                                        skipped_flows += 1
                                        skipped_in_this_step += 1
                                        continue
                            
                                    route_has_congestion = any(edge in high_congestion_edges for edge in current_route)
                                    
                                    if route_has_congestion:
                                        total_flows += 1
                                        new_route = router.get_path(source, dest, predicted_congestion_by_edge)
                                        if not new_route:  
                                            print(f"{Fore.YELLOW}No valid alternative route found for flow {source}->{dest}, keeping current route{Style.RESET_ALL}")
                                            skipped_flows += 1
                                            skipped_in_this_step += 1
                                            continue
                                        if new_route == current_route:  
                                            print(f"{Fore.YELLOW}Alternative route same as current for flow {source}->{dest}, keeping current route{Style.RESET_ALL}")
                                            skipped_flows += 1
                                            skipped_in_this_step += 1
                                            continue
                                        old_route_congestion = {edge: predicted_congestion_by_edge.get(edge, 0) for edge in current_route}
                                        new_route_congestion = {edge: predicted_congestion_by_edge.get(edge, 0) for edge in new_route}
                                        old_avg_congestion = np.mean([c for c in old_route_congestion.values() if c is not None])
                                        new_avg_congestion = np.mean([c for c in new_route_congestion.values() if c is not None])
                                        if new_avg_congestion >= old_avg_congestion * 0.95:  
                                            print(f"{Fore.YELLOW}New route not significantly better for flow {source}->{dest}, keeping current route{Style.RESET_ALL}")
                                            skipped_flows += 1
                                            skipped_in_this_step += 1
                                            continue
                                            
                                        # Try to apply the new route
                                        success = False
                                        for vehicle in flow_data['vehicles']:
                                            try:
                                                traci.vehicle.setRoute(vehicle, new_route)
                                                success = True
                                            except traci.exceptions.TraCIException:
                                                continue  # Skip this vehicle if rerouting fails
                                        
                                        if success:
                                            rerouted_flows += 1
                                            rerouted_in_this_step += 1
                                            # Add this flow to the set of rerouted flows
                                            rerouted_flow_keys.add(flow_key)
                                            
                                            print(f"\n{Fore.GREEN}Successfully rerouted flow from {source} to {dest}{Style.RESET_ALL}")
                                            print(f"{Fore.CYAN}Flow Details:{Style.RESET_ALL}")
                                            print(f"  Flow ID: {flow_key}")
                                            print(f"  Vehicles in flow: {', '.join(flow_data['vehicles'])}")
                                            print(f"{Fore.CYAN}Congestion Statistics:{Style.RESET_ALL}")
                                            print(f"  Original route average congestion: {old_avg_congestion:.2%}")
                                            print(f"  New route average congestion: {new_avg_congestion:.2%}")
                                            print(f"  Congestion reduction: {(old_avg_congestion - new_avg_congestion):.2%}")
                                            print_route_info(source, dest, current_route, new_route, old_route_congestion, new_route_congestion)
                                        else:
                                            print(f"{Fore.YELLOW}Failed to reroute any vehicles in flow {source}->{dest}, keeping current routes{Style.RESET_ALL}")
                                            print(f"  Flow ID: {flow_key}")
                                            print(f"  Vehicles in flow: {', '.join(flow_data['vehicles'])}")
                                            skipped_flows += 1
                                            skipped_in_this_step += 1
                                
                                # Print summary of rerouting in this step
                                print(f"\n{Fore.CYAN}Step {step} Summary:{Style.RESET_ALL}")
                                print(f"  Flows rerouted: {rerouted_in_this_step}")
                                print(f"  Flows skipped: {skipped_in_this_step}")
                                print(f"  Total flows rerouted so far: {len(rerouted_flow_keys)}")
                
                # Print progress less frequently
                if step % VISUALIZATION_INTERVAL == 0:
                    print(f"\n{Fore.CYAN}Simulation Progress:{Style.RESET_ALL}")
                    print(f"  Step: {step}/{simulation_time}")
                    print(f"  Vehicles: {len(traci.vehicle.getIDList())}")
                    print(f"  Average congestion: {avg_congestion:.2f}")
                    print(f"  Total flows processed: {total_flows}")
                    print(f"  Flows rerouted: {rerouted_flows}")
                    print(f"  Flows skipped: {skipped_flows}")
                    print(f"  High congestion events: {high_congestion_events}")
            
            except traci.exceptions.TraCIException as e:
                print(f"{Fore.RED}TraCI error at step {step}: {str(e)}{Style.RESET_ALL}")
                continue  # Continue with next step even if current step fails
    
    except Exception as e:
        print(f"{Fore.RED}Error in simulation: {str(e)}{Style.RESET_ALL}")
    finally:
        traci.close()
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        plt.plot(timestamps[:step_idx], actual_congestion[:step_idx], label='Actual Congestion')
        plt.plot(timestamps[:step_idx], predicted_congestion[:step_idx], label='Predicted Congestion')
        plt.title('Traffic Congestion Over Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Average Congestion')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/prediction_results.png')
        plt.close()
        
        # Print final statistics
        print(f"\n{Fore.GREEN}Simulation completed!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Final Statistics:{Style.RESET_ALL}")
        print(f"  Total flows processed: {total_flows}")
        print(f"  Flows rerouted: {rerouted_flows}")
        print(f"  Flows skipped: {skipped_flows}")
        print(f"  High congestion events: {high_congestion_events}")
        print(f"  Average congestion: {np.mean(actual_congestion[:step_idx]):.2f}")
        print(f"  Maximum congestion: {np.max(actual_congestion[:step_idx]):.2f}")
        print(f"  Total unique flows rerouted: {len(rerouted_flow_keys)}")

def main():
    if len(sys.argv) != 4:
        print(f"{Fore.RED}Usage: python predict_and_simulate.py <sumocfg_file> <model_path> <scaler_path>{Style.RESET_ALL}")
        sys.exit(1)
    
    sumocfg_file = sys.argv[1]
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]
    
    # Run simulation for 2 hours (7200 steps)
    run_simulation(sumocfg_file, model_path, scaler_path, simulation_time=7200)

if __name__ == "__main__":
    main() 