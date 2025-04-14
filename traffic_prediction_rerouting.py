import os
import sys
import traci
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class TrafficPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, batch_size=32):
        X, y = self.prepare_data(data)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, data):
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)[0][0]

class AStarRouter:
    def __init__(self):
        self.graph = {}
        
    def build_graph(self, edges):
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

def collect_traffic_data(sumocfg_file, simulation_time=3600):
    sumo_binary = "sumo"
    sumo_cmd = [sumo_binary, "-c", sumocfg_file]
    
    traci.start(sumo_cmd)
    
    congestion_data = []
    edges = traci.edge.getIDList()
    
    for step in range(simulation_time):
        traci.simulationStep()
        
        # Collect congestion data (using edge occupancy as a metric)
        step_data = []
        for edge in edges:
            occupancy = traci.edge.getLastStepOccupancy(edge)
            step_data.append(occupancy)
        congestion_data.append(step_data)
    
    traci.close()
    return np.array(congestion_data)

def main():
    # Configuration
    sumocfg_file = "real_word_3.sumocfg"
    simulation_time = 3600  # 1 hour simulation
    
    # Collect traffic data
    print("Collecting traffic data...")
    congestion_data = collect_traffic_data(sumocfg_file, simulation_time)
    
    # Train LSTM model
    print("Training LSTM model...")
    predictor = TrafficPredictor()
    predictor.train(congestion_data.flatten())
    
    # Initialize A* router
    router = AStarRouter()
    
    # Start SUMO for rerouting
    sumo_binary = "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", sumocfg_file]
    traci.start(sumo_cmd)
    
    edges = traci.edge.getIDList()
    router.build_graph(edges)
    
    # Main simulation loop with rerouting
    for step in range(simulation_time):
        traci.simulationStep()
        
        # Predict congestion for next time step
        current_congestion = np.array([traci.edge.getLastStepOccupancy(edge) for edge in edges])
        predicted_congestion = predictor.predict(current_congestion)
        
        # If predicted congestion is high, reroute vehicles
        if predicted_congestion > 0.7:  # Threshold for congestion
            vehicles = traci.vehicle.getIDList()
            for vehicle in vehicles:
                current_edge = traci.vehicle.getRoadID(vehicle)
                target_edge = traci.vehicle.getRoute(vehicle)[-1]
                
                # Find alternative route using A*
                new_route = router.get_path(current_edge, target_edge)
                if new_route:
                    traci.vehicle.setRoute(vehicle, new_route)
    
    traci.close()

if __name__ == "__main__":
    main() 