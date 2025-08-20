🚦 Route Rationalization Model Using Machine Learning for Real-Time Traffic Management

📌 Project Overview
This project presents an intelligent traffic management system that integrates machine learning-based congestion prediction with dynamic vehicle rerouting to optimize traffic flow in urban networks.

The primary goal is to proactively reduce traffic congestion by rerouting vehicles before they enter congested areas.


🔑 Highlights

Proactive Congestion Prediction → Neural network forecasts congestion up to 5 time steps ahead.

Congestion-Aware Routing → Enhanced A* algorithm considers real-time traffic, road length, and lane count.

Dynamic Vehicle Rerouting → Vehicles rerouted in real time if new path offers >5% congestion reduction.

Real-Time Simulation → Integrated with SUMO (Simulation of Urban Mobility).

Performance Analysis → Logs rerouting decisions and congestion statistics for evaluation.

👉 Experimental results show a 19.4% reduction in traffic congestion on average.



🏗️ System Architecture

The system is built on a modular, three-tier architecture:

Data Processing Layer

Collects & normalizes real-time traffic data from SUMO.

Prediction Layer

Neural network (TensorFlow/Keras) for traffic forecasting.

Routing Layer

Congestion-aware A* algorithm for optimal pathfinding.

Simulation Layer

Controls SUMO environment and vehicle rerouting.



⚙️ Technical Requirements
Hardware

Intel Core i5 (or equivalent)

8GB RAM

256GB SSD



Software

Python 3.8+

SUMO (Simulation of Urban Mobility)

TensorFlow / Keras

NumPy

Matplotlib

🚀 Getting Started
🔧 Installation

Clone the repository:

git clone https://github.com/bidhuripriyanshu/ROUTE_RATIONALISATION_ML_FINAL.git



Install dependencies:

pip install tensorflow numpy matplotlib


Install SUMO:

Download & Install SUMO

Ensure SUMO binaries are added to your system PATH.



▶️ Usage

Start SUMO simulation with your network file.

Run the traffic prediction and routing script:

python main.py




Monitor real-time rerouting and congestion reduction logs.

📊 Performance Evaluation

Average 19.4% congestion reduction achieved.

System logs all rerouting decisions and congestion statistics.

Results visualized using Matplotlib.



👨‍💻 Authors

Aman Patre 
priyanshu bidhuri

