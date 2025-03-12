from flask import Flask, request, jsonify
import json
import math
import numpy as np
from scipy.spatial import cKDTree
from collections import deque

app = Flask(__name__)

# Load data on startup
with open("SolarSystem_Cycle2.txt", "r") as f:
    data = json.load(f)

# Constants
LY_IN_METERS = 9.461e15

# Preprocess data
ids, coords_list, id_to_name = [], [], {}
for sys_obj in data.values():
    sys_id = sys_obj["solarSystemId"]
    coord = (sys_obj["location"]["x"], sys_obj["location"]["y"], sys_obj["location"]["z"])
    ids.append(sys_id)
    coords_list.append(coord)
    id_to_name[sys_id] = sys_obj["solarSystemName"]

coords_array = np.array(coords_list)
tree = cKDTree(coords_array)
id_to_index = {sys_id: idx for idx, sys_id in enumerate(ids)}

# Helper functions
def find_system(name):
    for sys_obj in data.values():
        if sys_obj.get("solarSystemName") == name:
            return sys_obj
    return None

def reconstruct_path(predecessors, start_id, dest_id):
    path, current = [], dest_id
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return path[::-1]

def bfs(start_id, jump_range_m):
    predecessors = {start_id: None}
    queue = deque([start_id])
    while queue:
        current_id = queue.popleft()
        current_coords = coords_array[id_to_index[current_id]]
        neighbors = tree.query_ball_point(current_coords, r=jump_range_m)
        for idx in neighbors:
            neighbor_id = ids[idx]
            if neighbor_id not in predecessors:
                predecessors[neighbor_id] = current_id
                queue.append(neighbor_id)
    return predecessors

def calculate_route(start, destination, jump_range_ly):
    start_sys = find_system(start)
    dest_sys = find_system(destination)
    if not start_sys or not dest_sys:
        return {"error": "System not found"}

    start_id, dest_id = start_sys["solarSystemId"], dest_sys["solarSystemId"]
    jump_range_m = jump_range_ly * LY_IN_METERS

    predecessors = bfs(start_id, jump_range_m)
    if dest_id not in predecessors:
        return {"error": "No route found"}

    path = reconstruct_path(predecessors, start_id, dest_id)
    return {"route": [id_to_name[sys_id] for sys_id in path]}

@app.route("/route", methods=["GET"])
def route():
    start = request.args.get("start")
    destination = request.args.get("destination")
    jump_range_ly = float(request.args.get("jump_range", 100))

    if not start or not destination:
        return jsonify({"error": "Missing parameters"}), 400

    result = calculate_route(start, destination, jump_range_ly)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
