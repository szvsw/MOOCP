import torch
import numpy as np

from linkages.linkage_utils import (
    sort_mech,
    batch_random_generator,
    batch_get_mat,
    batch_get_oriented_both,
    solve_rev_vectorized_batch_wds,
)


def batch_generate_mechs(config):
    BATCH_SIZE = config["BATCH_SIZE"]
    NODE_COUNT = config["NODE_COUNT"]
    TIMESTEPS = config["TIMESTEPS"]
    SPATIAL_DIMS = config["SPATIAL_DIMS"]
    SOLVE = config["SOLVE"]

    mechanisms = batch_random_generator(
        BATCH_SIZE,
        N_min=NODE_COUNT,
        N_max=NODE_COUNT,
        show_progress=True,
        strategy="rand",
    )
    # For clarity, convert the mechanisms into labeled data
    with torch.no_grad():
        mechanisms_dict = [
            {
                "C_mat": mechanism[0],
                "X0": mechanism[1],
                "fixed_nodes": mechanism[2],
                "motor": mechanism[3],
            }
            for mechanism in mechanisms
        ]
        if not SOLVE:
            return mechanisms_dict

        # Sort the mechanisms
        mechanism_sorted = [
            sort_mech(
                A=mech["C_mat"],
                x0=mech["X0"],
                motor=mech["motor"],
                fixed_nodes=mech["fixed_nodes"],
            )
            for mech in mechanisms_dict
        ]

        # Convert the sorted mechanisms into dicts for clarity
        mechanism_sorted_dict = [
            {
                "C_mat": sorted_mech[0],
                "X0": sorted_mech[1],
                "fixed_nodes": sorted_mech[2],
                "sorted_order": sorted_mech[3],
            }
            for sorted_mech in mechanism_sorted
        ]

        # Get the connectivity mats
        C_mats_sorted = torch.stack(
            [torch.tensor(mech["C_mat"], dtype=bool) for mech in mechanism_sorted_dict]
        )

        # Get the initial positions
        X0_sorted = torch.stack(
            [
                torch.tensor(mech["X0"], dtype=torch.float32)
                for mech in mechanism_sorted_dict
            ]
        )

        # Get material
        material_usage = batch_get_mat(X0_sorted, C_mats_sorted)

        # Get the indices of fixed nodes
        fixed_node_inds = [
            torch.tensor(mech["fixed_nodes"], dtype=int)
            for mech in mechanism_sorted_dict
        ]

        # Get the Assign the fixed node indices
        node_types_sorted = torch.zeros(size=(BATCH_SIZE, NODE_COUNT, 1), dtype=bool)
        for i, inds in enumerate(fixed_node_inds):
            node_types_sorted[i, inds, 0] = True

        # Generate theta
        thetas = torch.Tensor(np.linspace(0, np.pi * 2, TIMESTEPS + 1)[0:TIMESTEPS])

        # Solve
        sol, cos = solve_rev_vectorized_batch_wds(
            As=C_mats_sorted, x0s=X0_sorted, node_types=node_types_sorted, thetas=thetas
        )

        torch.cuda.empty_cache()
        sol = sol.to("cuda")
        curve = torch.zeros(BATCH_SIZE, NODE_COUNT, TIMESTEPS, SPATIAL_DIMS)
        curve_rot = torch.zeros(BATCH_SIZE, NODE_COUNT, TIMESTEPS, SPATIAL_DIMS)
        for k in range(BATCH_SIZE):
            a, b = batch_get_oriented_both(sol[k])
            curve[k] = a
            curve_rot[k] = b
        sol = sol.cpu()
        torch.cuda.empty_cache()

        return {
            "sol": sol,
            "curve": curve,
            "curve_rot": curve_rot,
            "material_usage": material_usage,
            "mechanisms": mechanisms_dict,
            "mechanism_sorted": mechanism_sorted_dict,
        }
