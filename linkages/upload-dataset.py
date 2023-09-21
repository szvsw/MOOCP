import wandb

solved_schema = {
    "file_structure": "root/nodes_n/batchsize_b",
    "data_structure": {
        "sol": {
            "description": "Solved path",
            "shape": ("batch_size", "n_nodes", "t_timesteps", "d_spatial_dims"),
            "type": "torch.Tensor",
        },
        "curve": {
            "description": "Solved path, oriented",
            "shape": ("batch_size", "n_nodes", "t_timesteps", "d_spatial_dims"),
            "type": "torch.Tensor",
        },
        "curve_rot": {
            "description": "Solved path, oriented and rotated 180",
            "shape": ("batch_size", "n_nodes", "t_timesteps", "d_spatial_dims"),
            "type": "torch.Tensor",
        },
        "material_usage": {
            "description": "Linkage material size",
            "shape": ("batch_size"),
            "type": "torch.Tensor",
        },
        "mechanisms": {
            "description": "List of mechanisms as dictionaries",
            "shape": "list[dict[str,np.ndarray]]",
            "type": "torch.Tensor",
            "extra": "Each mechanism is stored with C_mat (adjacency matrix), X0 (node locations), fixed_nodes (which nodes are fixed), and motor",
        },
    },
}
artifact = wandb.Artifact(
    name="unsolved-mechanisms",
    type="dataset",
    description="Dataset of unsolved pre-generated mechanisms for various number of nodes and batch sizes.",
    metadata={
        "file_structure": [
            "full_<batchsize>/nodes_<nodecount>.npy",
            "nodes_<nodecount>/batchsize_<batchsize>/<minibatch_id>.npy",
        ],
    },
)
artifact.add_dir("unsolved", "unsolved")
with wandb.init(
    job_type="upload",
    name="upload unsolved linkage mechanisms datasets",
    project="linkages",
) as run:
    run.log_artifact(artifact)
