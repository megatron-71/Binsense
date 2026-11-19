
"""
BinSense - Controlled Routing Algorithm Comparison

Compares:
1. Nearest Neighbour heuristic
2. Google OR-Tools CVRP

Both algorithms use the same:
- Selected bins
- Demand values
- Road-network distance matrix
- Depot
- Fleet size and vehicle capacities
"""

import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.config import (
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
)
from src.simulate import (
    generate_bins,
    generate_fill_history,
    get_current_state,
)
from src.network import load_road_graph, build_distance_matrix
from src.predict import predict_all_bins
from src.priority import score_all_bins
from src.select import select_bins
from src.routing import solve_routes, format_route_results


def nearest_neighbour_routes(distance_matrix, demands):
    """
    Multi-vehicle nearest-neighbour heuristic.

    Each vehicle repeatedly visits its nearest feasible unvisited bin.
    Every route starts and ends at depot index 0.
    """

    unvisited = set(range(1, len(distance_matrix)))
    routes = []

    for vehicle_id in range(NUM_VEHICLES):
        current = 0
        load = 0
        distance = 0
        stops = []

        while unvisited:
            feasible = [
                node for node in unvisited
                if load + demands[node] <= VEHICLE_CAPACITY_LITERS
            ]

            if not feasible:
                break

            next_node = min(
                feasible,
                key=lambda node: distance_matrix[current][node]
            )

            distance += int(distance_matrix[current][next_node])
            load += demands[next_node]
            stops.append(next_node)

            unvisited.remove(next_node)
            current = next_node

        if stops:
            distance += int(distance_matrix[current][0])

        routes.append({
            "vehicle_id": vehicle_id,
            "num_stops": len(stops),
            "distance_m": distance,
            "load_liters": load,
            "utilization_pct": round(
                load / VEHICLE_CAPACITY_LITERS * 100, 1
            ),
            "stops": stops,
        })

    return {
        "routes": routes,
        "total_distance": sum(r["distance_m"] for r in routes),
        "total_load": sum(r["load_liters"] for r in routes),
        "visited_count": sum(r["num_stops"] for r in routes),
        "unvisited": sorted(unvisited),
    }


def main():

    print("=" * 65)
    print("BinSense - Controlled Routing Comparison")
    print("=" * 65)

    # Phase 1: Generate the same input pipeline
    print("\n[1] Preparing simulation data...")

    G = load_road_graph()
    bins = generate_bins(G)
    history = generate_fill_history(bins)
    state = get_current_state(bins, history)

    predictions = predict_all_bins(state, history)
    scored = score_all_bins(predictions)
    selection = select_bins(scored)

    selected = selection["selected"]

    print(f"Selected bins: {len(selected)}")
    print(f"Estimated demand: {selection['total_demand_liters']:.0f} L")
    print(f"Fleet capacity: {selection['fleet_capacity_liters']:.0f} L")
    print(f"Effective capacity: {selection['effective_capacity_liters']:.0f} L")

    if selected.empty:
        print("No bins selected. Experiment stopped.")
        return

    # Phase 2: Build one common road-distance matrix
    print("\n[2] Building common road-distance matrix...")

    matrix_result = build_distance_matrix(
        G,
        selected_bins_df=selected
    )

    matrix = matrix_result["distance_matrix"]
    matrix_bin_ids = matrix_result["bin_ids"]

    # Use exactly the bins represented in the matrix.
    selected_by_id = selected.set_index("bin_id")

    demands = [0] + [
        int(round(
            selected_by_id.loc[bin_id, "current_fill_pct"]
            / 100
            * selected_by_id.loc[bin_id, "capacity_liters"]
        ))
        for bin_id in matrix_bin_ids
    ]

    print(f"Matrix dimensions: {matrix.shape}")
    print(f"Bins included in experiment: {len(matrix_bin_ids)}")

    # Phase 3: Nearest Neighbour
    print("\n[3] Running Nearest Neighbour...")

    start = time.perf_counter()

    nn_result = nearest_neighbour_routes(matrix, demands)

    nn_runtime = time.perf_counter() - start

    # Phase 4: OR-Tools CVRP
    print("\n[4] Running OR-Tools CVRP...")

    start = time.perf_counter()

    cvrp_result = solve_routes(
        matrix,
        demands,
        num_vehicles=NUM_VEHICLES,
        vehicle_capacities=[
            VEHICLE_CAPACITY_LITERS
        ] * NUM_VEHICLES,
    )

    cvrp_runtime = time.perf_counter() - start

    cvrp_routes = format_route_results(
        cvrp_result,
        matrix_bin_ids,
        vehicle_capacities=[
            VEHICLE_CAPACITY_LITERS
        ] * NUM_VEHICLES,
    )

    cvrp_visited = sum(
        route["num_stops"] for route in cvrp_routes
    )

    # Phase 5: Report
    print("\n" + "=" * 65)
    print("CONTROLLED EXPERIMENT RESULTS")
    print("=" * 65)

    print("\nNearest Neighbour:")
    print(f"  Distance: {nn_result['total_distance'] / 1000:.3f} km")
    print(f"  Load: {nn_result['total_load']} L")
    print(f"  Bins visited: {nn_result['visited_count']}")
    print(f"  Bins unvisited: {len(nn_result['unvisited'])}")
    print(f"  Runtime: {nn_runtime:.4f} seconds")

    print("\nOR-Tools CVRP:")
    print(f"  Solver status: {cvrp_result['status']}")
    print(f"  Distance: {cvrp_result['total_distance'] / 1000:.3f} km")
    print(f"  Load: {cvrp_result['total_load']} L")
    print(f"  Bins visited: {cvrp_visited}")
    print(f"  Bins dropped: {len(cvrp_result.get('dropped_nodes', []))}")
    print(f"  Runtime: {cvrp_runtime:.4f} seconds")

    print("\nVehicle-wise Nearest Neighbour:")
    for route in nn_result["routes"]:
        print(
            f"  VEH-{route['vehicle_id'] + 1:02d}: "
            f"{route['num_stops']} stops, "
            f"{route['distance_m'] / 1000:.3f} km, "
            f"{route['load_liters']} L, "
            f"{route['utilization_pct']}% utilization"
        )

    print("\nVehicle-wise OR-Tools:")
    for route in cvrp_routes:
        print(
            f"  {route['vehicle_label']}: "
            f"{route['num_stops']} stops, "
            f"{route['distance_km']} km, "
            f"{route['load_liters']} L, "
            f"{route['utilization_pct']}% utilization"
        )

    if (
        nn_result["visited_count"] == len(matrix_bin_ids)
        and cvrp_visited == len(matrix_bin_ids)
    ):
        saved = nn_result["total_distance"] - cvrp_result["total_distance"]

        print("\nSame-workload distance comparison:")
        print(f"  Distance difference: {saved / 1000:.3f} km")

        if nn_result["total_distance"] > 0:
            improvement = saved / nn_result["total_distance"] * 100
            print(f"  CVRP distance reduction: {improvement:.2f}%")
    else:
        print("\nWARNING: The algorithms did not visit the same number of bins.")
        print("Do not interpret the distance difference as a fair comparison.")

    print("\n" + "=" * 65)
    print("Controlled routing experiment completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
