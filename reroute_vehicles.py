import os
import sys
import traci
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

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

class TrafficRerouter:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        self.scaler = MinMaxScaler()
        self.scaler.scale_ = np.array(scaler_params['scale_'])
        self.scaler.min_ = np.array(scaler_params['min_'])
        self.sequence_length = 10
        self.router = AStarRouter()
        
    def predict_congestion(self, current_data):
        scaled_data = self.scaler.transform(current_data.reshape(-1, 1))
        X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)[0][0]
    
    def reroute_vehicles(self, sumocfg_file, simulation_time=3600):
        print("Starting SUMO simulation with rerouting...")
        
        
        sumo_binary = "sumo-gui"
        sumo_cmd = [sumo_binary, "-c", sumocfg_file]
        traci.start(sumo_cmd)
        
        
        edges = traci.edge.getIDList()
        self.router.build_graph(edges)
        
        # Initialize congestion history
        congestion_history = []
        
        # Main simulation loop
        for step in range(simulation_time):
            traci.simulationStep()
            
            # Collect current congestion data
            current_congestion = np.array([traci.edge.getLastStepOccupancy(edge) for edge in edges])
            congestion_history.append(current_congestion)
            
            if len(congestion_history) >= self.sequence_length:
                # Predict future congestion
                predicted_congestion = self.predict_congestion(np.array(congestion_history))
                
                # If predicted congestion is high, reroute vehicles
                if predicted_congestion > 0.7:  # Threshold for congestion
                    print(f"Step {step}: High congestion predicted ({predicted_congestion:.2f}), rerouting vehicles...")
                    vehicles = traci.vehicle.getIDList()
                    
                    for vehicle in vehicles:
                        current_edge = traci.vehicle.getRoadID(vehicle)
                        target_edge = traci.vehicle.getRoute(vehicle)[-1]
                        
                        # Find alternative route using A*
                        new_route = self.router.get_path(current_edge, target_edge)
                        if new_route:
                            traci.vehicle.setRoute(vehicle, new_route)
                            print(f"Rerouted vehicle {vehicle} to new route")
            
            # Keep only the last sequence_length elements
            if len(congestion_history) > self.sequence_length:
                congestion_history.pop(0)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Simulation step {step}/{simulation_time}")
        
        traci.close()
        print("Simulation completed!")

def save_scaler_params(scaler, filename):
    params = {
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(params, f)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python reroute_vehicles.py <sumocfg_file> <model_path> <scaler_path>")
        sys.exit(1)
    
    sumocfg_file = sys.argv[1]
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]
    
    rerouter = TrafficRerouter(model_path, scaler_path)
    rerouter.reroute_vehicles(sumocfg_file) 