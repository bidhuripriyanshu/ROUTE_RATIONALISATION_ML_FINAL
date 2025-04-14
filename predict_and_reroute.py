import os
import sys
import traci
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from collections import deque

class AStarRouter:
    def __init__(self):
        self.graph = {}
        
    def build_graph(self, edges):
        print("Building road network graph...")
        for edge in edges:
            if edge not in self.graph:
                self.graph[edge] = []
            for connection in traci.edge.getConnections(edge):
                target = traci.edge.getConnections(edge)[connection]
                if target not in self.graph:
                    self.graph[target] = []
                self.graph[edge].append(target)
    
    def heuristic(self, a, b):
        return traci.edge.getTravelTime(a)
    
    def get_path(self, start, goal):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = frontier.pop(0)[1]
            
            if current == goal:
                break
                
            for next_edge in self.graph[current]:
                new_cost = cost_so_far[current] + self.heuristic(current, next_edge)
                
                if next_edge not in cost_so_far or new_cost < cost_so_far[next_edge]:
                    cost_so_far[next_edge] = new_cost
                    priority = new_cost + self.heuristic(next_edge, goal)
                    frontier.append((priority, next_edge))
                    frontier.sort()
                    came_from[next_edge] = current
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path if path[0] == start else []

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
        
        # Initialize data buffer
        self.data_buffer = deque(maxlen=sequence_length)
    
    def predict(self, current_data):
        # Add current data to buffer
        self.data_buffer.append(current_data)
        
        # If buffer is not full, return None
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        # Prepare data for prediction
        scaled_data = self.scaler.transform(np.array(self.data_buffer).reshape(-1, 1))
        X = scaled_data.reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)[0]

class TrafficRerouter:
    def __init__(self, model_path, scaler_path, congestion_threshold=0.7):
        self.predictor = TrafficPredictor(model_path, scaler_path)
        self.router = AStarRouter()
        self.congestion_threshold = congestion_threshold
    
    def reroute_vehicles(self, sumocfg_file, simulation_time=3600):
        print("Starting SUMO simulation with rerouting...")
        
        # Start SUMO
        sumo_binary = "sumo-gui"
        sumo_cmd = [sumo_binary, "-c", sumocfg_file]
        traci.start(sumo_cmd)
        
        # Build road network
        edges = traci.edge.getIDList()
        self.router.build_graph(edges)
        
        # Main simulation loop
        for step in range(simulation_time):
            traci.simulationStep()
            
            # Collect current congestion data
            current_congestion = np.array([traci.edge.getLastStepOccupancy(edge) for edge in edges])
            avg_congestion = np.mean(current_congestion)
            
            # Predict future congestion
            predicted_congestion = self.predictor.predict(avg_congestion)
            
            if predicted_congestion is not None:
                # Check if any future time step has high congestion
                if np.any(predicted_congestion > self.congestion_threshold):
                    print(f"Step {step}: High congestion predicted, rerouting vehicles...")
                    vehicles = traci.vehicle.getIDList()
                    
                    for vehicle in vehicles:
                        current_edge = traci.vehicle.getRoadID(vehicle)
                        target_edge = traci.vehicle.getRoute(vehicle)[-1]
                        
                        # Find alternative route using A*
                        new_route = self.router.get_path(current_edge, target_edge)
                        if new_route:
                            traci.vehicle.setRoute(vehicle, new_route)
                            print(f"Rerouted vehicle {vehicle} to new route")
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Simulation step {step}/{simulation_time}")
        
        traci.close()
        print("Simulation completed!")

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict_and_reroute.py <sumocfg_file> <model_path> <scaler_path>")
        sys.exit(1)
    
    sumocfg_file = sys.argv[1]
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]
    
    rerouter = TrafficRerouter(model_path, scaler_path)
    rerouter.reroute_vehicles(sumocfg_file)

if __name__ == "__main__":
    main() 