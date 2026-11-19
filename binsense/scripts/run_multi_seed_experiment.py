
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import time


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import NUM_VEHICLES, VEHICLE_CAPACITY_LITERS
from src.simulate import (
    generate_bins,
    generate_fill_history,
    get_current_state,
)
from src.network import load_road_graph, build_distance_matrix
from src.predict import predict_fill, predict_all_bins
from src.priority import score_all_bins
from src.select import select_bins
from src.routing import solve_routes, format_route_results


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = list(range(42, 52))

END_TIME = pd.Timestamp("2026-11-12 12:00:00")
FORECAST_HORIZON_HOURS = 12
HISTORY_HOURS = 72

NUM_VEHICLES = NUM_VEHICLES
VEHICLE_CAPACITIES = [VEHICLE_CAPACITY_LITERS] * NUM_VEHICLES

SOLVER_TIME_LIMIT_SECONDS = 10

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

RESULTS_FILE = DATA_DIR / "multi_seed_results.csv"
VEHICLE_RESULTS_FILE = DATA_DIR / "multi_seed_vehicle_results.csv"
SUMMARY_FILE = DATA_DIR / "multi_seed_summary.csv"


# ============================================================
# NEAREST NEIGHBOUR BASELINE
# ============================================================

def nearest_neighbour_routes(distance_matrix, demands, bin_ids):
    """
    Greedy nearest-neighbour routing with vehicle capacity constraints.

    Uses the same distance matrix, demands, depot, and fleet
    capacities as the OR-Tools CVRP experiment.
    """

    unvisited = set(range(1, len(distance_matrix)))
    routes = []

    for vehicle_id, capacity in enumerate(VEHICLE_CAPACITIES):

        current_node = 0
        current_load = 0
        route_distance = 0
        stops = []

        while True:

            candidates = [
                node for node in unvisited
                if current_load + demands[node] <= capacity
            ]

            if not candidates:
                break

            next_node = min(
                candidates,
                key=lambda node: distance_matrix[current_node][node]
            )

            route_distance += int(
                distance_matrix[current_node][next_node]
            )

            current_load += demands[next_node]
            stops.append(next_node)
            unvisited.remove(next_node)

            current_node = next_node

        if stops:
            route_distance += int(distance_matrix[current_node][0])

        stop_bin_ids = [
            bin_ids[node - 1] for node in stops
        ]

        routes.append({
            "vehicle_id": vehicle_id,
            "vehicle_label": f"VEH-{vehicle_id + 1:02d}",
            "num_stops": len(stops),
            "stop_bin_ids": stop_bin_ids,
            "distance_m": route_distance,
            "distance_km": round(route_distance / 1000, 2),
            "load_liters": current_load,
            "capacity_liters": capacity,
            "utilization_pct": round(
                current_load / capacity * 100, 1
            ) if capacity > 0 else 0,
        })

    visited_count = sum(route["num_stops"] for route in routes)

    return {
        "routes": routes,
        "total_distance": sum(
            route["distance_m"] for route in routes
        ),
        "total_load": sum(
            route["load_liters"] for route in routes
        ),
        "visited_count": visited_count,
        "unvisited": sorted(unvisited),
    }


# ============================================================
# PREDICTION EVALUATION
# ============================================================

def evaluate_prediction(history_df, seed):

    forecast_origin = (
        END_TIME - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    )

    training_history = history_df[
        history_df["timestamp"] <= forecast_origin
    ].copy()

    actual_history = history_df[
        history_df["timestamp"] == END_TIME
    ].copy()

    if training_history.empty or actual_history.empty:
        raise ValueError(
            f"Seed {seed}: missing forecast training or actual data."
        )

    predicted_values = []
    actual_values = []

    for bin_id in history_df["bin_id"].unique():

        bin_training = training_history[
            training_history["bin_id"] == bin_id
        ].sort_values("timestamp")

        bin_actual = actual_history[
            actual_history["bin_id"] == bin_id
        ]

        if bin_training.empty or len(bin_actual) != 1:
            raise ValueError(
                f"Seed {seed}: invalid prediction data for {bin_id}"
            )

        prediction = predict_fill(
            bin_training,
            horizon_hours=FORECAST_HORIZON_HOURS,
        )

        predicted_values.append(
            prediction["predicted_fill_pct"]
        )

        actual_values.append(
            float(bin_actual.iloc[0]["fill_pct"])
        )

    mae = mean_absolute_error(actual_values, predicted_values)

    rmse = np.sqrt(
        mean_squared_error(actual_values, predicted_values)
    )

    return {
        "prediction_mae": float(mae),
        "prediction_rmse": float(rmse),
        "prediction_bins_evaluated": len(actual_values),
    }


# ============================================================
# VEHICLE RESULT COLLECTION
# ============================================================

def collect_vehicle_results(seed, method, routes):

    records = []

    for route in routes:

        records.append({
            "seed": seed,
            "method": method,
            "vehicle_id": route["vehicle_id"],
            "vehicle_label": route["vehicle_label"],
            "num_stops": route["num_stops"],
            "distance_km": route["distance_km"],
            "load_liters": route["load_liters"],
            "capacity_liters": route["capacity_liters"],
            "utilization_pct": route["utilization_pct"],
        })

    return records


# ============================================================
# ONE SEED EXPERIMENT
# ============================================================

def run_one_seed(G, seed):

    print("\n" + "=" * 65)
    print(f"RUNNING EXPERIMENT — SEED {seed}")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Generate reproducible simulation data
    # --------------------------------------------------------

    bins = generate_bins(G, seed=seed)

    history = generate_fill_history(
        bins,
        seed=seed,
        history_hours=HISTORY_HOURS,
        interval_hours=1,
        end_time=END_TIME,
    )

    print(f"Bins: {len(bins)}")
    print(f"History readings: {len(history)}")

    # --------------------------------------------------------
    # 2. Evaluate 12-hour prediction
    # --------------------------------------------------------

    prediction_metrics = evaluate_prediction(history, seed)

    print(
        f"Prediction MAE: "
        f"{prediction_metrics['prediction_mae']:.3f}"
    )

    print(
        f"Prediction RMSE: "
        f"{prediction_metrics['prediction_rmse']:.3f}"
    )

    # --------------------------------------------------------
    # 3. Run BinSense selection pipeline
    # --------------------------------------------------------

    current_state = get_current_state(
        bins,
        history,
        as_of=END_TIME,
    )

    predicted_bins = predict_all_bins(
        current_state,
        history,
        horizon_hours=FORECAST_HORIZON_HOURS,
    )

    scored_bins = score_all_bins(predicted_bins)

    selection = select_bins(
        scored_bins,
        fleet_capacity_liters=sum(VEHICLE_CAPACITIES),
    )

    selected_bins = selection["selected"].copy()

    selected_count = len(selected_bins)

    selected_demand = float(
        selection["total_demand_liters"]
    )

    print(f"Selected bins: {selected_count}")
    print(f"Selected demand: {selected_demand:.0f} L")
    print(f"Deferred bins: {selection['deferred_count']}")

    if selected_bins.empty:
        raise ValueError(
            f"Seed {seed}: no bins selected. "
            "Check the selection configuration."
        )

    # --------------------------------------------------------
    # 4. Build common road-distance matrix
    # --------------------------------------------------------

    matrix_data = build_distance_matrix(
        G,
        selected_bins_df=selected_bins,
        weight="length",
    )

    distance_matrix = matrix_data["distance_matrix"]
    matrix_bin_ids = matrix_data["bin_ids"]

    selected_ids = set(selected_bins["bin_id"])

    reachable_ids = set(matrix_bin_ids)

    unreachable_ids = selected_ids - reachable_ids

    reachable_count = len(matrix_bin_ids)

    print(f"Reachable bins: {reachable_count}")
    print(f"Unreachable bins: {len(unreachable_ids)}")

    if reachable_count == 0:
        raise ValueError(
            f"Seed {seed}: no selected bins are reachable."
        )

    # --------------------------------------------------------
    # 5. Build one common demand vector
    # --------------------------------------------------------

    demand_lookup = (
        selected_bins
        .set_index("bin_id")["estimated_demand_liters"]
        .to_dict()
    )

    demands = [0] + [
        int(round(float(demand_lookup[bin_id])))
        for bin_id in matrix_bin_ids
    ]

    routed_demand = sum(demands[1:])

    print(f"Routed demand: {routed_demand} L")

    if len(demands) != len(distance_matrix):
        raise ValueError(
            f"Seed {seed}: demand and matrix dimensions do not match."
        )

    # --------------------------------------------------------
    # 6. Run Nearest Neighbour
    # --------------------------------------------------------

    nn_start = time.perf_counter()

    nn_result = nearest_neighbour_routes(
        distance_matrix,
        demands,
        matrix_bin_ids,
    )

    nn_runtime = time.perf_counter() - nn_start

    nn_routes = nn_result["routes"]

    nn_distance = nn_result["total_distance"]

    nn_visited = nn_result["visited_count"]

    nn_unvisited = len(nn_result["unvisited"])

    # --------------------------------------------------------
    # 7. Run OR-Tools CVRP
    # --------------------------------------------------------

    cvrp_start = time.perf_counter()

    cvrp_result = solve_routes(
        distance_matrix,
        demands,
        num_vehicles=NUM_VEHICLES,
        vehicle_capacities=VEHICLE_CAPACITIES,
        depot_index=0,
        time_limit_seconds=SOLVER_TIME_LIMIT_SECONDS,
    )

    cvrp_runtime = time.perf_counter() - cvrp_start

    cvrp_routes = format_route_results(
        cvrp_result,
        matrix_bin_ids,
        vehicle_capacities=VEHICLE_CAPACITIES,
    )

    cvrp_distance = cvrp_result["total_distance"]

    cvrp_visited = sum(
        route["num_stops"] for route in cvrp_routes
    )

    dropped_nodes = cvrp_result.get("dropped_nodes", [])

    # --------------------------------------------------------
    # 8. Calculate comparable routing improvement
    # --------------------------------------------------------

    both_methods_visited_all = (
        nn_visited == reachable_count
        and cvrp_visited == reachable_count
    )

    if both_methods_visited_all and nn_distance > 0:
        distance_reduction_pct = (
            (nn_distance - cvrp_distance)
            / nn_distance
        ) * 100
    else:
        distance_reduction_pct = np.nan

    # --------------------------------------------------------
    # 9. Collect vehicle-level results
    # --------------------------------------------------------

    vehicle_records = []

    vehicle_records.extend(
        collect_vehicle_results(
            seed, "Nearest Neighbour", nn_routes
        )
    )

    vehicle_records.extend(
        collect_vehicle_results(
            seed, "OR-Tools CVRP", cvrp_routes
        )
    )

    # --------------------------------------------------------
    # 10. Collect seed-level results
    # --------------------------------------------------------

    result = {
        "seed": seed,

        "prediction_mae": prediction_metrics["prediction_mae"],
        "prediction_rmse": prediction_metrics["prediction_rmse"],
        "prediction_bins_evaluated":
            prediction_metrics["prediction_bins_evaluated"],

        "selected_bins": selected_count,
        "selected_demand_liters": selected_demand,
        "deferred_bins": selection["deferred_count"],
        "capacity_exceeded": selection["capacity_exceeded"],

        "reachable_bins": reachable_count,
        "unreachable_bins": len(unreachable_ids),
        "routed_demand_liters": routed_demand,

        "nn_distance_km": nn_distance / 1000,
        "nn_visited_bins": nn_visited,
        "nn_unvisited_bins": nn_unvisited,
        "nn_total_load_liters": nn_result["total_load"],
        "nn_runtime_seconds": nn_runtime,

        "cvrp_status": cvrp_result["status"],
        "cvrp_distance_km": cvrp_distance / 1000,
        "cvrp_visited_bins": cvrp_visited,
        "cvrp_dropped_bins": len(dropped_nodes),
        "cvrp_total_load_liters": cvrp_result["total_load"],
        "cvrp_runtime_seconds": cvrp_runtime,

        "both_methods_visited_all": both_methods_visited_all,
        "distance_reduction_pct": distance_reduction_pct,
    }

    print("\nRouting comparison:")
    print(f"Nearest Neighbour: {nn_distance / 1000:.3f} km")
    print(f"OR-Tools CVRP:     {cvrp_distance / 1000:.3f} km")
    print(f"NN visited:        {nn_visited}/{reachable_count}")
    print(f"CVRP visited:      {cvrp_visited}/{reachable_count}")

    if both_methods_visited_all:
        print(
            f"Distance reduction: "
            f"{distance_reduction_pct:.2f}%"
        )
    else:
        print(
            "Distance reduction: NOT COMPARABLE "
            "(one or both methods did not visit every reachable bin)"
        )

    return result, vehicle_records


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BINSENSE — 10-SEED EXPERIMENT")
    print("=" * 65)

    print(f"Seeds: {SEEDS}")
    print(f"Fixed end time: {END_TIME}")
    print(f"Vehicles: {NUM_VEHICLES}")
    print(f"Capacity per vehicle: {VEHICLE_CAPACITY_LITERS} L")

    # Load the road graph only once.
    print("\nLoading road graph...")
    G = load_road_graph()

    all_results = []
    all_vehicle_results = []

    for seed in SEEDS:

        try:
            result, vehicle_results = run_one_seed(G, seed)

            all_results.append(result)
            all_vehicle_results.extend(vehicle_results)

        except Exception as error:

            print(f"\nERROR in seed {seed}: {error}")

            raise

    # --------------------------------------------------------
    # Save individual results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    vehicle_df = pd.DataFrame(all_vehicle_results)

    results_df.to_csv(RESULTS_FILE, index=False)

    vehicle_df.to_csv(VEHICLE_RESULTS_FILE, index=False)

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    numeric_columns = results_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_columns = [
        column for column in numeric_columns
        if column != "seed"
    ]

    summary_records = []

    for column in numeric_columns:

        values = results_df[column].dropna()

        summary_records.append({
            "metric": column,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "valid_runs": len(values),
        })

    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("10-SEED EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"Successful runs: {len(results_df)}/{len(SEEDS)}")

    print("\nKey results:")

    key_metrics = [
        "prediction_mae",
        "prediction_rmse",
        "nn_distance_km",
        "cvrp_distance_km",
        "distance_reduction_pct",
    ]

    print(
        summary_df[
            summary_df["metric"].isin(key_metrics)
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(RESULTS_FILE)
    print(VEHICLE_RESULTS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
