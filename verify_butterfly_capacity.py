"""
Finite verification for the 5/6 achievability conditions
in reduced butterfly-like networks under one-hop interference.

This script tests both cases:
    (1) I = D
    (2) I != D

For each reduced configuration, the script:
    - builds the corresponding directed graph;
    - constructs all hyperedges induced by outgoing neighborhoods;
    - enumerates all matchings of the hypergraph;
    - solves a linear program over these matchings;
    - compares the LP optimum with the structural conditions.

Dependencies:
    networkx
    pulp

Install if needed:
    pip install networkx pulp
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import time

import networkx as nx
import pulp


TARGET = 5.0 / 6.0
TOL = 1e-7
INNER_SUBPATH_NAMES = frozenset({"c", "d", "g", "h", "i"})


# ============================================================
# 1. Graph construction
# ============================================================

def add_subpath(G, name, start, end, n_relays):
    """
    Add a directed subpath from start to end with n_relays intermediate nodes.

    If n_relays = 0, this simply adds the edge start -> end.
    If n_relays > 0, the intermediate nodes are named name_1, ..., name_n.
    """
    if n_relays == 0:
        G.add_edge(start, end, subpath=name)
        return

    relays = [f"{name}_{i}" for i in range(1, n_relays + 1)]
    G.add_edge(start, relays[0], subpath=name)

    for i in range(n_relays - 1):
        G.add_edge(relays[i], relays[i + 1], subpath=name)

    G.add_edge(relays[-1], end, subpath=name)


def build_butterfly_graph(config, variant):
    """
    Build the reduced butterfly-like directed graph.

    Parameters
    ----------
    config : dict
        Contains relay counts n_a, n_b, n_c, n_d, n_e, n_f, n_g, n_h,
        and additionally n_i for the case I != D.

    variant : str
        Either "I_eq_D" or "I_neq_D".

    Returns
    -------
    G : networkx.DiGraph
    """
    G = nx.DiGraph()

    if variant == "I_eq_D":
        # Nodes: S, A, B, I, X, Y, where I = D.
        main_nodes = ["S", "A", "B", "I", "X", "Y"]
        G.add_nodes_from(main_nodes)

        add_subpath(G, "a", "S", "A", config["n_a"])
        add_subpath(G, "b", "S", "B", config["n_b"])
        add_subpath(G, "c", "A", "I", config["n_c"])
        add_subpath(G, "d", "B", "I", config["n_d"])
        add_subpath(G, "e", "A", "X", config["n_e"])
        add_subpath(G, "f", "B", "Y", config["n_f"])
        add_subpath(G, "g", "I", "X", config["n_g"])
        add_subpath(G, "h", "I", "Y", config["n_h"])

    elif variant == "I_neq_D":
        # Nodes: S, A, B, I, D, X, Y, where I != D.
        main_nodes = ["S", "A", "B", "I", "D", "X", "Y"]
        G.add_nodes_from(main_nodes)

        add_subpath(G, "a", "S", "A", config["n_a"])
        add_subpath(G, "b", "S", "B", config["n_b"])
        add_subpath(G, "c", "A", "I", config["n_c"])
        add_subpath(G, "d", "B", "I", config["n_d"])
        add_subpath(G, "e", "A", "X", config["n_e"])
        add_subpath(G, "f", "B", "Y", config["n_f"])
        add_subpath(G, "i", "I", "D", config["n_i"])
        add_subpath(G, "g", "D", "X", config["n_g"])
        add_subpath(G, "h", "D", "Y", config["n_h"])

    else:
        raise ValueError("variant must be either 'I_eq_D' or 'I_neq_D'.")

    return G


# ============================================================
# 2. Hyperedge construction and one-hop conflict model
# ============================================================

def generate_hyperedges(G):
    """
    Generate all hyperedges induced by outgoing neighborhoods.

    For each node u with out-neighbors Gamma^+(u), every nonempty subset
    B of Gamma^+(u) gives a hyperedge (u, B).

    A hyperedge is represented as:
        (tail, tuple(sorted(head_set)))
    """
    hyperedges = []

    for u in G.nodes():
        out_neighbors = sorted(list(G.successors(u)))
        if not out_neighbors:
            continue

        for r in range(1, len(out_neighbors) + 1):
            for heads in itertools.combinations(out_neighbors, r):
                hyperedges.append((u, tuple(heads)))

    return hyperedges


def incident_nodes(hyperedge):
    """
    Return the set of nodes incident to a hyperedge.
    """
    tail, heads = hyperedge
    return {tail} | set(heads)


def hyperedges_conflict(e1, e2):
    """
    Two hyperedges conflict under one-hop interference iff
    they are incident to a common node.
    """
    return len(incident_nodes(e1) & incident_nodes(e2)) > 0


def enumerate_matchings(hyperedges):
    """
    Enumerate all matchings of the hypergraph, including the empty matching.

    A matching is a set of pairwise non-incident hyperedges.
    We construct the conflict graph on hyperedges and enumerate all
    independent sets, equivalently all cliques in the complement graph.

    It is not sufficient to retain only maximal matchings here. Extending a
    nonmaximal matching can increase an induced inner-edge capacity and violate
    the equality constraint that fixes that capacity to 1/3.

    Returns
    -------
    matchings : list of tuple of int
        Each matching is represented by a tuple of hyperedge indices.
    """
    conflict_graph = nx.Graph()
    conflict_graph.add_nodes_from(range(len(hyperedges)))

    for i in range(len(hyperedges)):
        for j in range(i + 1, len(hyperedges)):
            if hyperedges_conflict(hyperedges[i], hyperedges[j]):
                conflict_graph.add_edge(i, j)

    complement_graph = nx.complement(conflict_graph)

    matchings = [tuple()]
    for clique in nx.enumerate_all_cliques(complement_graph):
        matchings.append(tuple(sorted(clique)))

    return matchings


# ============================================================
# 3. Min-cut LP over all matchings
# ============================================================

def build_dummy_arcs(hyperedges):
    """
    Build the dummy-node representation of the hypergraph.

    For each hyperedge e=(u,B), introduce a dummy node d_e,
    an arc u -> d_e, and arcs d_e -> v for all v in B.

    Each such arc is associated with the same hyperedge index.
    """
    arcs = []
    dummy_nodes = []

    for idx, (tail, heads) in enumerate(hyperedges):
        dummy = ("dummy", idx)
        dummy_nodes.append(dummy)

        arcs.append((tail, dummy, idx))
        for h in heads:
            arcs.append((dummy, h, idx))

    return dummy_nodes, arcs


def get_inner_subpath_edges(G):
    """Return all directed edges on the constrained inner subpaths."""
    return [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data.get("subpath") in INNER_SUBPATH_NAMES
    ]


def solve_capacity_lp(
    G,
    hyperedges,
    matchings,
    source="S",
    destinations=("X", "Y"),
    cut_tolerance=TOL,
    max_cut_iterations=10000,
):
    """
    Maximize the minimum source-destination cut over resource allocations.

    Variables:
        omega_sigma >= 0 for each matching sigma
        R >= 0

    Hyperedge capacity:
        c_e = sum_{sigma contains e} omega_sigma

    The LP contains R <= capacity(K) for every source-destination cut K.
    Rather than enumerate all cuts in advance, violated cut constraints are
    generated iteratively. After each LP solve, the current minimum cut for
    every destination is computed in the dummy-node graph and added to the LP
    whenever its capacity is smaller than R.

    Every induced edge on subpaths c, d, g, and h, and also on subpath i when
    present, is constrained to have capacity 1/3.

    Returns
    -------
    capacity : float
        The optimized minimum cut over all destinations.
    status : str
        PuLP status string.
    solution : dict
        Contains nonzero matching weights, final minimum cuts, and the number
        of generated cut constraints.
    """
    model = pulp.LpProblem("Butterfly_5_6_Mincut_Verification", pulp.LpMaximize)

    R = pulp.LpVariable("R", lowBound=0, upBound=1)

    omega = {
        j: pulp.LpVariable(f"omega_{j}", lowBound=0)
        for j in range(len(matchings))
    }

    model += R
    model += pulp.lpSum(omega[j] for j in omega) == 1, "resource_sum"

    # Hyperedge capacity expressions
    containing_matchings = {i: [] for i in range(len(hyperedges))}
    for j, matching in enumerate(matchings):
        for e_idx in matching:
            containing_matchings[e_idx].append(j)

    c_expr = {
        i: pulp.lpSum(omega[j] for j in containing_matchings[i])
        for i in range(len(hyperedges))
    }

    # Fix the capacity induced by every directed edge on an inner subpath.
    inner_subpath_edges = get_inner_subpath_edges(G)
    for constraint_idx, (u, v) in enumerate(inner_subpath_edges):
        covering_hyperedges = {
            e_idx
            for e_idx, (tail, heads) in enumerate(hyperedges)
            if tail == u and v in heads
        }
        covering_matchings = [
            j
            for j, matching in enumerate(matchings)
            if any(e_idx in covering_hyperedges for e_idx in matching)
        ]
        model += (
            pulp.lpSum(omega[j] for j in covering_matchings) == 1.0 / 3.0,
            f"inner_edge_capacity_{constraint_idx}",
        )

    dummy_nodes, arcs = build_dummy_arcs(hyperedges)
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1)
    added_cuts = set()
    final_min_cut_values = None
    final_cut_partitions = None

    for iteration in range(1, max_cut_iterations + 1):
        model.solve(solver)
        status = pulp.LpStatus[model.status]

        if status != "Optimal":
            return None, status, None

        r_value = pulp.value(R)
        hyperedge_capacity_values = {
            e_idx: max(0.0, float(pulp.value(expr) or 0.0))
            for e_idx, expr in c_expr.items()
        }

        dummy_graph = nx.DiGraph()
        dummy_graph.add_nodes_from(G.nodes())
        dummy_graph.add_nodes_from(dummy_nodes)
        for u, v, e_idx in arcs:
            dummy_graph.add_edge(
                u,
                v,
                capacity=hyperedge_capacity_values[e_idx],
                hyperedge_index=e_idx,
            )

        violated_cuts = []
        current_min_cut_values = {}
        current_cut_partitions = {}

        for t in destinations:
            cut_value, (source_side, destination_side) = nx.minimum_cut(
                dummy_graph,
                source,
                t,
                capacity="capacity",
            )
            source_side = frozenset(source_side)
            destination_side = frozenset(destination_side)
            current_min_cut_values[t] = float(cut_value)
            current_cut_partitions[t] = (source_side, destination_side)

            if cut_value < r_value - cut_tolerance:
                cut_key = (t, source_side)
                if cut_key in added_cuts:
                    raise RuntimeError(
                        "A previously added cut remains violated beyond the "
                        "separation tolerance."
                    )
                violated_cuts.append((t, source_side))

        if not violated_cuts:
            final_min_cut_values = current_min_cut_values
            final_cut_partitions = current_cut_partitions
            break

        for t, source_side in violated_cuts:
            cut_capacity = pulp.lpSum(
                c_expr[e_idx]
                for u, v, e_idx in arcs
                if u in source_side and v not in source_side
            )
            model += R <= cut_capacity, f"cut_{t}_{len(added_cuts)}"
            added_cuts.add((t, source_side))
    else:
        return None, "CutGenerationLimit", None

    capacity = min(final_min_cut_values.values())

    solution = {
        "nonzero_weights": {
            j: pulp.value(omega[j])
            for j in omega
            if pulp.value(omega[j]) is not None and pulp.value(omega[j]) > 1e-9
        },
        "matchings": matchings,
        "hyperedges": hyperedges,
        "objective_R": float(pulp.value(R)),
        "minimum_cuts": final_min_cut_values,
        "cut_partitions": final_cut_partitions,
        "generated_cut_constraints": len(added_cuts),
        "cut_iterations": iteration,
        "inner_subpath_edges": inner_subpath_edges,
    }

    return capacity, status, solution


# ============================================================
# 4. Structural conditions
# ============================================================

def check_conditions(config, variant):
    """
    Check conditions 1)--7) of Theorem 10, with the replacement
    n_g -> n_g' and n_h -> n_h' for the case I != D.

    Returns
    -------
    (bool, str)
    """
    n_a = config["n_a"]
    n_b = config["n_b"]
    n_c = config["n_c"]
    n_d = config["n_d"]
    n_e = config["n_e"]
    n_f = config["n_f"]
    n_g = config["n_g"]
    n_h = config["n_h"]

    if variant == "I_neq_D":
        n_i = config["n_i"]
        n_g_eff = n_g + n_i + 1
        n_h_eff = n_h + n_i + 1
    else:
        n_g_eff = n_g
        n_h_eff = n_h

    # Condition 1:
    # n_g > 0 or n_h > 0, if n_a+n_e and n_b+n_f have the same parity.
    if (n_a + n_e) % 2 == (n_b + n_f) % 2:
        if not (n_g_eff > 0 or n_h_eff > 0):
            return False, "Condition 1 failed."

    # Condition 2:
    # n_c > 0 or n_g > 0, if n_e is odd.
    if n_e % 2 == 1:
        if not (n_c > 0 or n_g_eff > 0):
            return False, "Condition 2 failed."

    # Condition 3:
    # n_d > 0 or n_h > 0, if n_f is odd.
    if n_f % 2 == 1:
        if not (n_d > 0 or n_h_eff > 0):
            return False, "Condition 3 failed."

    # Condition 4:
    # n_c != 1, if n_e is even, n_a and n_b have the same parity,
    # and n_d = n_g = 0.
    if n_e % 2 == 0 and n_a % 2 == n_b % 2 and n_d == 0 and n_g_eff == 0:
        if n_c == 1:
            return False, "Condition 4 failed."

    # Condition 5:
    # n_g != 1, if n_e is even, n_a and n_b have different parities,
    # and n_c = n_d = 0.
    if n_e % 2 == 0 and n_a % 2 != n_b % 2 and n_c == 0 and n_d == 0:
        if n_g_eff == 1:
            return False, "Condition 5 failed."

    # Condition 6:
    # n_d != 1, if n_f is even, n_a and n_b have the same parity,
    # and n_c = n_h = 0.
    if n_f % 2 == 0 and n_a % 2 == n_b % 2 and n_c == 0 and n_h_eff == 0:
        if n_d == 1:
            return False, "Condition 6 failed."

    # Condition 7:
    # n_h != 1, if n_f is even, n_a and n_b have different parities,
    # and n_c = n_d = 0.
    if n_f % 2 == 0 and n_a % 2 != n_b % 2 and n_c == 0 and n_d == 0:
        if n_h_eff == 1:
            return False, "Condition 7 failed."

    return True, "All conditions satisfied."


# ============================================================
# 5. Exhaustive enumeration
# ============================================================

def enumerate_configurations(variant):
    """
    Enumerate all reduced configurations.

    For both variants:
        n_a,n_b,n_e,n_f in {0,1}
        n_c,n_d,n_g,n_h in {0,1,2}

    For I != D:
        n_i in {0,1,2}
    """
    outer_values = [0, 1]
    inner_values = [0, 1, 2]

    if variant == "I_eq_D":
        for n_a, n_b, n_e, n_f in itertools.product(outer_values, repeat=4):
            for n_c, n_d, n_g, n_h in itertools.product(inner_values, repeat=4):
                yield {
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_c": n_c,
                    "n_d": n_d,
                    "n_e": n_e,
                    "n_f": n_f,
                    "n_g": n_g,
                    "n_h": n_h,
                }

    elif variant == "I_neq_D":
        for n_a, n_b, n_e, n_f in itertools.product(outer_values, repeat=4):
            for n_c, n_d, n_g, n_h, n_i in itertools.product(inner_values, repeat=5):
                yield {
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_c": n_c,
                    "n_d": n_d,
                    "n_e": n_e,
                    "n_f": n_f,
                    "n_g": n_g,
                    "n_h": n_h,
                    "n_i": n_i,
                }

    else:
        raise ValueError("variant must be either 'I_eq_D' or 'I_neq_D'.")


def test_variant(variant, verbose=False):
    """
    Test one variant exhaustively.

    Returns
    -------
    summary : dict
    """
    total_cases = 0
    condition_true_cases = 0
    achieved_cases = 0
    matching_cases = 0
    mismatches = []

    for config in enumerate_configurations(variant):
        total_cases += 1

        theory, message = check_conditions(config, variant)

        G = build_butterfly_graph(config, variant)
        hyperedges = generate_hyperedges(G)
        matchings = enumerate_matchings(hyperedges)

        capacity, status, solution = solve_capacity_lp(
            G,
            hyperedges,
            matchings,
            source="S",
            destinations=("X", "Y"),
        )

        if status != "Optimal":
            achieved = False
            cap_value = None
        else:
            cap_value = capacity
            achieved = cap_value >= TARGET - TOL

        if theory:
            condition_true_cases += 1
        if achieved:
            achieved_cases += 1
        if theory == achieved:
            matching_cases += 1
        else:
            mismatches.append({
                "config": config,
                "theory": theory,
                "achieved": achieved,
                "capacity": cap_value,
                "status": status,
                "message": message,
            })

        if verbose and total_cases % 100 == 0:
            print(f"[{variant}] processed {total_cases} cases...")

    summary = {
        "variant": variant,
        "total_cases": total_cases,
        "condition_true_cases": condition_true_cases,
        "achieved_cases": achieved_cases,
        "matching_cases": matching_cases,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }

    return summary


def print_summary(summary, max_mismatches_to_print=10):
    """
    Print a compact summary.
    """
    print("\n" + "=" * 70)
    print(f"Variant: {summary['variant']}")
    print("=" * 70)
    print(f"Total reduced configurations: {summary['total_cases']}")
    print(f"Configurations satisfying conditions: {summary['condition_true_cases']}")
    print(f"Configurations achieving capacity >= 5/6: {summary['achieved_cases']}")
    print(f"Configurations where theory matches LP: {summary['matching_cases']}")
    print(f"Mismatches: {summary['mismatch_count']}")

    if summary["mismatch_count"] > 0:
        print("\nMismatch examples:")
        for item in summary["mismatches"][:max_mismatches_to_print]:
            if "theory" in item:
                theory = item["theory"]
                achieved = item["achieved"]
                status = item["status"]
                message = item["message"]
            else:
                theory = item["condition_satisfied"]
                achieved = item["achieves_target"]
                status = item["lp_status"]
                message = item["condition_message"]
            print("-" * 70)
            print(f"config   = {item['config']}")
            print(f"theory   = {theory}")
            print(f"achieved = {achieved}")
            print(f"capacity = {item['capacity']}")
            print(f"status   = {status}")
            print(f"message  = {message}")
    else:
        print("No mismatches found.")


def evaluate_configuration(task):
    """Evaluate one reduced configuration and return a JSON-serializable record."""
    variant, case_index, config = task
    theory, message = check_conditions(config, variant)

    G = build_butterfly_graph(config, variant)
    hyperedges = generate_hyperedges(G)
    matchings = enumerate_matchings(hyperedges)
    capacity, status, solution = solve_capacity_lp(
        G,
        hyperedges,
        matchings,
        source="S",
        destinations=("X", "Y"),
    )

    achieved = (
        status == "Optimal"
        and capacity is not None
        and capacity >= TARGET - TOL
    )

    return {
        "variant": variant,
        "case_index": case_index,
        "config": config,
        "condition_satisfied": theory,
        "condition_message": message,
        "lp_status": status,
        "capacity": capacity,
        "achieves_target": achieved,
        "matches_theory": theory == achieved,
        "number_of_hyperedges": len(hyperedges),
        "number_of_matchings": len(matchings),
        "objective_R": None if solution is None else solution["objective_R"],
        "minimum_cuts": None if solution is None else solution["minimum_cuts"],
        "generated_cut_constraints": (
            None if solution is None else solution["generated_cut_constraints"]
        ),
        "cut_iterations": None if solution is None else solution["cut_iterations"],
    }


def summarize_records(records, variant):
    """Summarize the records for one of the two graph variants."""
    selected = [record for record in records if record["variant"] == variant]
    mismatches = [record for record in selected if not record["matches_theory"]]
    return {
        "variant": variant,
        "total_cases": len(selected),
        "optimal_cases": sum(
            record["lp_status"] == "Optimal" for record in selected
        ),
        "condition_true_cases": sum(
            record["condition_satisfied"] for record in selected
        ),
        "achieved_cases": sum(record["achieves_target"] for record in selected),
        "matching_cases": sum(record["matches_theory"] for record in selected),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def load_existing_records(path):
    """Load a partial JSONL result file for an interrupted run."""
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from exc
    return records


def run_exhaustive_verification(
    workers,
    output_dir,
    progress_every=50,
    resume=False,
    limit_per_variant=None,
):
    """Run the finite verification in parallel and save complete results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "verification_results.jsonl"
    summary_path = output_dir / "verification_summary.json"

    tasks = []
    for variant in ("I_eq_D", "I_neq_D"):
        configurations = enumerate_configurations(variant)
        for case_index, config in enumerate(configurations, start=1):
            if limit_per_variant is not None and case_index > limit_per_variant:
                break
            tasks.append((variant, case_index, config))

    loaded_records = load_existing_records(records_path) if resume else []
    existing_by_key = {
        (record["variant"], record["case_index"]): record
        for record in loaded_records
    }
    existing_records = list(existing_by_key.values())
    completed_keys = {
        (record["variant"], record["case_index"])
        for record in existing_records
    }
    pending_tasks = [
        task for task in tasks if (task[0], task[1]) not in completed_keys
    ]

    file_mode = "a" if resume else "w"
    records = list(existing_records)
    total_cases = len(tasks)
    completed_cases = len(existing_records)
    start_time = time.perf_counter()

    print(
        f"Starting verification with {workers} worker(s): "
        f"{completed_cases}/{total_cases} cases already available.",
        flush=True,
    )

    with records_path.open(file_mode, encoding="utf-8") as result_stream:
        if workers == 1:
            futures_or_records = map(evaluate_configuration, pending_tasks)
            for record in futures_or_records:
                records.append(record)
                result_stream.write(json.dumps(record, sort_keys=True) + "\n")
                result_stream.flush()
                completed_cases += 1
                if (
                    completed_cases % progress_every == 0
                    or completed_cases == total_cases
                ):
                    elapsed = time.perf_counter() - start_time
                    newly_completed = completed_cases - len(existing_records)
                    rate = newly_completed / elapsed if elapsed > 0 else 0.0
                    remaining = total_cases - completed_cases
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"Processed {completed_cases}/{total_cases} cases "
                        f"({rate:.2f} cases/s, ETA {eta / 60:.1f} min).",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluate_configuration, task): task
                    for task in pending_tasks
                }
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    result_stream.write(json.dumps(record, sort_keys=True) + "\n")
                    result_stream.flush()
                    completed_cases += 1
                    if (
                        completed_cases % progress_every == 0
                        or completed_cases == total_cases
                    ):
                        elapsed = time.perf_counter() - start_time
                        newly_completed = completed_cases - len(existing_records)
                        rate = newly_completed / elapsed if elapsed > 0 else 0.0
                        remaining = total_cases - completed_cases
                        eta = remaining / rate if rate > 0 else float("inf")
                        print(
                            f"Processed {completed_cases}/{total_cases} cases "
                            f"({rate:.2f} cases/s, ETA {eta / 60:.1f} min).",
                            flush=True,
                        )

    variant_order = {"I_eq_D": 0, "I_neq_D": 1}
    records.sort(key=lambda record: (
        variant_order[record["variant"]],
        record["case_index"],
    ))
    with records_path.open("w", encoding="utf-8") as result_stream:
        for record in records:
            result_stream.write(json.dumps(record, sort_keys=True) + "\n")

    elapsed_seconds = time.perf_counter() - start_time
    summaries = {
        variant: summarize_records(records, variant)
        for variant in ("I_eq_D", "I_neq_D")
    }
    summary = {
        "target": TARGET,
        "tolerance": TOL,
        "workers": workers,
        "elapsed_seconds": elapsed_seconds,
        "complete_enumeration": limit_per_variant is None,
        "total_cases": len(records),
        "optimal_cases": sum(
            record["lp_status"] == "Optimal" for record in records
        ),
        "matching_cases": sum(record["matches_theory"] for record in records),
        "mismatch_count": sum(
            not record["matches_theory"] for record in records
        ),
        "variants": summaries,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    for variant in ("I_eq_D", "I_neq_D"):
        print_summary(summaries[variant])

    if summary["mismatch_count"] == 0:
        print(
            "\nFinal conclusion: the structural conditions match the LP "
            "verification for both I=D and I!=D.",
            flush=True,
        )
    else:
        print(
            "\nFinal conclusion: mismatches were found. "
            "Please inspect the saved records.",
            flush=True,
        )

    print(f"Results: {records_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return summary


def parse_args():
    """Parse command-line arguments."""
    default_workers = min(4, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively verify the 5/6 structural characterization for "
            "reduced butterfly-like networks."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"number of worker processes (default: {default_workers})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for JSONL records and the JSON summary",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print progress after this many completed cases",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from an existing verification_results.jsonl file",
    )
    parser.add_argument(
        "--limit-per-variant",
        type=int,
        default=None,
        help="test only the first N configurations of each variant",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    if args.limit_per_variant is not None and args.limit_per_variant < 1:
        parser.error("--limit-per-variant must be at least 1")
    return args


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    arguments = parse_args()
    run_exhaustive_verification(
        workers=arguments.workers,
        output_dir=arguments.output_dir,
        progress_every=arguments.progress_every,
        resume=arguments.resume,
        limit_per_variant=arguments.limit_per_variant,
    )
