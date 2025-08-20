Route Rationalization Model Using Machine Learning for Real-Time Traffic Management
Project Overview
This project presents an intelligent traffic management system that combines machine learning-based congestion prediction with dynamic vehicle rerouting to optimize traffic flow in urban networks. The primary goal is to proactively reduce traffic congestion by rerouting vehicles before they enter congested areas.

The system leverages a three-tier architecture:

A Traffic Predictor using a neural network to forecast future congestion levels.

An A 
∗
  routing algorithm enhanced with congestion awareness to find optimal paths.

A Simulation Controller that manages real-time vehicle rerouting within the SUMO (Simulation of Urban Mobility) environment.

Our findings show that this system successfully reduces traffic congestion by an average of 19.4% through proactive rerouting, demonstrating its potential for real-world application in smart city infrastructure.

Key Features
Proactive Congestion Prediction: Uses a neural network to forecast congestion up to 5 time steps ahead based on historical data.

Congestion-Aware Routing: The A 
∗
  algorithm calculates route costs based on real-time traffic conditions, road length, and number of lanes to find the most efficient path.

Dynamic Vehicle Rerouting: Vehicles are rerouted in real-time when a new path offers a significant improvement (>5% reduction in congestion).

Real-Time Simulation: The system is integrated with the SUMO traffic simulator to create a realistic testing environment.

Performance Analysis: The system logs detailed rerouting decisions and congestion statistics to monitor and evaluate performance over extended periods.

Technical Details
Architecture
The system is built on a modular, three-tier architecture:

Data Processing Layer: Collects and normalizes real-time traffic data from the SUMO simulation.

Prediction Layer: Implements the neural network model for traffic forecasting using TensorFlow/Keras.

Routing Layer: Utilizes the congestion-aware A 
∗
  algorithm to optimize vehicle routes.

Simulation Layer: Manages the SUMO environment and handles vehicle movements and rerouting.

Requirements
Hardware: Intel Core i5 or equivalent CPU, 8GB RAM, 256GB SSD storage.

Software:

Python 3.8+

SUMO (Simulation of Urban Mobility)

TensorFlow/Keras

NumPy

Matplotlib

Getting Started
Installation
(Provide instructions on how to install the necessary dependencies, e.g., pip install tensorflow numpy etc., and how to get SUMO.)

Usage
(Explain how to run the simulation and interact with the system.)

Authors
Aman Patre 
priyanshu bidhuri

License
(Add license information here, e.g., MIT License)
