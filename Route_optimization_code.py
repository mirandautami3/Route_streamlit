import tensorflow as tf
import numpy as np
import random
from collections import deque
import osmnx as ox
import networkx as nx
import folium
from matplotlib import cm
import matplotlib.colors as mcolors
import osmnx as ox
from geopy.geocoders import Nominatim
import time

# Fungsi untuk mencari koordinat berdasarkan nama tempat (menggunakan geopy)
def get_location_coordinates(location_name):
    geolocator = Nominatim(user_agent="route_optimizer")
    try:
        location = geolocator.geocode(location_name)
        if location:
            print(f"Found location: {location_name} at latitude {location.latitude}, longitude {location.longitude}")
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Location '{location_name}' not found.")
    except Exception as e:
        print(f"Error occurred while geocoding {location_name}: {e}")
        return None
    
# Fungsi untuk mengambil lokasi dari input atau sumber lain
def get_locations_from_input():
    # Anda bisa mengganti bagian ini untuk membaca input dari pengguna, API, file, dll
    depot_location_name = input("Enter depot location: ")  # Input nama lokasi depot
    customer_count = int(input("Enter number of customers: "))  # Input jumlah pelanggan
    customer_locations_names = []
    
    for i in range(customer_count):
        customer_location_name = input(f"Enter location for Customer {i+1}: ")
        customer_locations_names.append(customer_location_name)
    
    # Ambil lokasi depot dan pelanggan
    depot_location = get_location_coordinates(depot_location_name)
    if not depot_location:
        print(f"Failed to get coordinates for depot location: {depot_location_name}")
        return None, None

    customer_locations = []
    for customer_location_name in customer_locations_names:
            customer_location = get_location_coordinates(customer_location_name)
            if not customer_location:
                print(f"Failed to get coordinates for customer location: {customer_location_name}")
            else:
                customer_locations.append(customer_location)
    
    # Gabungkan depot dan pelanggan dalam satu list
    locations = [depot_location] + customer_locations
    return locations, depot_location

# Ambil lokasi depot dan pelanggan secara dinamis
locations, depot_location = get_locations_from_input()

# Fetch road network with extended distance
G = ox.graph_from_point(depot_location, dist=25000, network_type='drive')

# Function to validate nearest nodes and map locations to the road network
def map_to_nearest_nodes(locations, graph, distance_threshold=5000):
    valid_nodes = []
    valid_demands = []
    for idx, (lat, lng) in enumerate(locations):
        nearest_node = ox.nearest_nodes(graph, lng, lat)
        nearest_point = (graph.nodes[nearest_node]['y'], graph.nodes[nearest_node]['x'])
        dist_to_nearest = ox.distance.great_circle(lat, lng, nearest_point[0], nearest_point[1])
        print(f"Location {idx}: Distance to nearest node = {dist_to_nearest:.2f} meters")
        if dist_to_nearest > distance_threshold:  # Skip locations too far from the network
            print(f"Warning: Location {idx} is far from the road network! Skipping this location.")
            continue
        valid_nodes.append(nearest_node)
        if idx > 0:  # Exclude depot from demands
            valid_demands.append(10)
    return valid_nodes, valid_demands

# Map locations to nearest nodes
valid_nodes, valid_demands = map_to_nearest_nodes(locations, G)

# Convert graph to undirected for connectivity check
G_undirected = G.to_undirected()

# Check connectivity of the graph
if not nx.is_connected(G_undirected):
    print("Warning: The road network graph is not fully connected. Consider increasing the distance or changing locations.")
else:
    print("The road network graph is fully connected.")

# Build distance matrix for valid nodes
def build_distance_matrix(valid_nodes, graph):
    distance_matrix = np.zeros((len(valid_nodes), len(valid_nodes)))
    paths = {}
    for i, start_node in enumerate(valid_nodes):
        for j, end_node in enumerate(valid_nodes):
            if i != j:
                try:
                    path = nx.shortest_path(graph, start_node, end_node, weight='length')
                    distance = nx.shortest_path_length(graph, start_node, end_node, weight='length')
                    distance_matrix[i][j] = distance
                    paths[(i, j)] = path
                except nx.NetworkXNoPath:
                    print(f"Warning: No path between nodes {start_node} and {end_node}")
                    distance_matrix[i][j] = float('inf')
                    paths[(i, j)] = []
    return distance_matrix, paths

# Build distance matrix and paths
distance_matrix, paths = build_distance_matrix(valid_nodes, G)

# Print summary
print(f"Valid nodes: {valid_nodes}")
print(f"Distance matrix:\n{distance_matrix}")


# Define DQN model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# Define VRP Agent
class VRPAgent:
    def __init__(self, num_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.replay_buffer = ReplayBuffer()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, unvisited_nodes):
        state_tensor = tf.convert_to_tensor([[state]], dtype=tf.float32)
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(unvisited_nodes))
        else:
            q_values = self.model(state_tensor).numpy()[0]
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[list(unvisited_nodes)] = q_values[list(unvisited_nodes)]
            return np.argmax(masked_q_values)

    def store_experience(self, state, action, reward, next_state, done):
        state = np.atleast_2d(state)
        next_state = np.atleast_2d(next_state)
        self.replay_buffer.add((state, action, reward, next_state, done))
        
    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = tf.convert_to_tensor(np.array(states).squeeze(axis=1), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states).squeeze(axis=1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_q_values = self.target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.keras.losses.MSE(targets, q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
def train_vrp_agent(agent, num_vehicles, capacity, customer_demands, distance_matrix, num_episodes=150, batch_size=32, update_target_every=10):
    best_routes = []
    best_total_distance = float('inf')

    for episode in range(num_episodes):
        total_distance = 0
        cumulative_reward = 0
        episode_routes = []
        unvisited_nodes = set(range(1, len(customer_demands) + 1))  # Adjusted for valid_nodes (exclude depot)

        for vehicle_idx in range(num_vehicles):
            route = [0]  # Start from depot (node index 0)
            load = 0
            current_node = 0

            while unvisited_nodes:
                action = agent.choose_action(current_node, unvisited_nodes)

                if action in unvisited_nodes:
                    dist = distance_matrix[current_node][action]
                    if load + customer_demands[action - 1] <= capacity:
                        route.append(action)
                        load += customer_demands[action - 1]
                        total_distance += dist
                        unvisited_nodes.remove(action)

                        reward = max(10 / (dist + 1e-5), 1)  # Reward is inversely proportional to distance
                        reward += 10  # Extra reward for visiting a customer

                        cumulative_reward += reward
                        agent.store_experience(np.array([current_node]), action, reward, np.array([action]), False)

                        current_node = action
                    else:
                        reward = -100  # Penalize if the vehicle exceeds capacity
                        agent.store_experience(np.array([current_node]), action, reward, np.array([current_node]), True)
                        break
                else:
                    reward = -50  # Penalize if the action is invalid (not in unvisited nodes)
                    agent.store_experience(np.array([current_node]), action, reward, np.array([current_node]), True)
                    break

            if unvisited_nodes or vehicle_idx < num_vehicles - 1:
                route.append(0)  # Return to depot
                total_distance += distance_matrix[current_node][0]
                reward = 100  # Reward for returning to depot
                cumulative_reward += reward
            elif vehicle_idx == num_vehicles - 1:
                print(f"Last route ends at node {current_node} without returning to depot.")

            episode_routes.append(route)

        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_routes = episode_routes

        print(f"Episode {episode + 1}/{num_episodes} - Total Distance: {total_distance} - Reward: {cumulative_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

        agent.train(batch_size=batch_size)
        if episode % update_target_every == 0:
            agent.update_target_model()

    print(f"Best Total Distance: {best_total_distance}")
    return best_routes


# Initialize and train the VRP Agent
capacity = 100
num_vehicles = 1
# Initialize and train the VRP Agent
num_actions = len(distance_matrix[0])
agent = VRPAgent(num_actions=num_actions)
routes = train_vrp_agent(agent, num_vehicles=1, capacity=100, customer_demands=valid_demands, distance_matrix=distance_matrix)


# Visualization with folium
m = folium.Map(location=depot_location, zoom_start=13)
folium.Marker(depot_location, popup="Depot", icon=folium.Icon(color="red")).add_to(m)

# Add markers for customers
for idx, loc in enumerate(locations[1:], start=1):
    folium.Marker(loc, popup=f"Customer {idx}", icon=folium.Icon(color="blue")).add_to(m)

# Color palette for routes
num_routes = len(routes)
color_palette = cm.get_cmap('rainbow', num_routes)

for i, route in enumerate(routes):
    rgba_color = color_palette(i / num_routes)
    route_color = mcolors.to_hex(rgba_color)

    for j in range(len(route) - 1):
        path = paths.get((route[j], route[j + 1]), [])
        if not path:
            print(f"Warning: No valid path between node {route[j]} and {route[j + 1]}")
            continue

        edge_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
        folium.PolyLine(
            edge_coords,
            color=route_color,
            weight=5,
            opacity=1,
            tooltip=f"Route {i + 1}"
        ).add_to(m)

# Save the map
m.save("vrp_routes_petra.html")
print("Routes visualized in 'vrp_routes.html'")

import streamlit as st
import folium
import osmnx as ox
from geopy.geocoders import Nominatim
from io import BytesIO
import tensorflow as tf
import numpy as np
import random
from collections import deque
import networkx as nx
from matplotlib import cm
import matplotlib.colors as mcolors

# Fungsi untuk mencari koordinat berdasarkan nama tempat (menggunakan geopy)
def get_location_coordinates(location_name):
    geolocator = Nominatim(user_agent="route_optimizer")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    return None

# Fungsi untuk memetakan lokasi ke node terdekat dalam jaringan jalan
def map_to_nearest_nodes(locations, graph, distance_threshold=5000):
    valid_nodes = []
    for lat, lng in locations:
        nearest_node = ox.nearest_nodes(graph, lng, lat)
        nearest_point = (graph.nodes[nearest_node]['y'], graph.nodes[nearest_node]['x'])
        dist_to_nearest = ox.distance.great_circle(lat, lng, nearest_point[0], nearest_point[1])
        if dist_to_nearest > distance_threshold:
            continue
        valid_nodes.append(nearest_node)
    return valid_nodes

# Inisialisasi dan input di Streamlit
st.title("Optimasi Rute dengan VRP (Vehicle Routing Problem)")

# Input lokasi depot dan pelanggan
depot_location_name = st.text_input("Masukkan lokasi depot:")
customer_location_names = st.text_area("Masukkan lokasi pelanggan (pisahkan dengan koma):")

if st.button("Generate Route"):
    if depot_location_name and customer_location_names:
        # Memproses input lokasi
        customer_locations = customer_location_names.split(",")
        depot_location = get_location_coordinates(depot_location_name)
        locations = [depot_location] + [get_location_coordinates(loc) for loc in customer_locations]

        if None in locations:
            st.error("Salah satu lokasi tidak ditemukan.")
        else:
            # Ambil jaringan jalan menggunakan OSMN
            G = ox.graph_from_point(depot_location, dist=25000, network_type='drive')

            # Map lokasi ke node terdekat
            valid_nodes = map_to_nearest_nodes(locations, G)

            # Fungsi untuk membangun matriks jarak (distance matrix)
            def build_distance_matrix(valid_nodes, graph):
                distance_matrix = np.zeros((len(valid_nodes), len(valid_nodes)))
                for i, start_node in enumerate(valid_nodes):
                    for j, end_node in enumerate(valid_nodes):
                        if i != j:
                            path = nx.shortest_path(graph, start_node, end_node, weight='length')
                            distance = nx.shortest_path_length(graph, start_node, end_node, weight='length')
                            distance_matrix[i][j] = distance
                return distance_matrix

            # Bangun matriks jarak dan jalur
            distance_matrix = build_distance_matrix(valid_nodes, G)

            # Menampilkan peta dengan folium
            m = folium.Map(location=depot_location, zoom_start=13)
            folium.Marker(depot_location, popup="Depot", icon=folium.Icon(color="red")).add_to(m)

            for idx, loc in enumerate(locations[1:], start=1):
                folium.Marker(loc, popup=f"Customer {idx}", icon=folium.Icon(color="blue")).add_to(m)

            # Menyimpan peta sebagai HTML
            map_html = f"<html><body>{m._repr_html_()}</body></html>"
            st.components.v1.html(map_html, height=500)

    else:
        st.warning("Silakan masukkan lokasi depot dan pelanggan terlebih dahulu.")
