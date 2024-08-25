import time
import json
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import multiprocessing as mp
import torch
import pickle

from folding.utils.file_management import file_manager
from folding.utils import protein
from folding.relax import relax_me


def predict_structure(
    prefix: str,
    result_dir: Path,
    feature_dict: Dict[str, Any],
    is_complex: bool,
    use_templates: bool,
    sequences_lengths: List[int],
    pad_len: int,
    model_type: str,
    model_runner_and_params: List[Tuple[str, Any, Any]],
    num_relax: int = 0,
    relax_max_iterations: int = 0,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    rank_by: str = "auto",
    random_seed: int = 0,
    num_seeds: int = 1,
    stop_at_score: float = 100,
    prediction_callback: Optional[Callable] = None,
    use_gpu_relax: bool = False,
    save_all: bool = False,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_recycles: bool = False,
    num_gpus: int = None,
    jobs_per_gpu: int = 1,
):
    """Predicts structure using AlphaFold for the given sequence."""
    mean_scores = []
    conf = []
    unrelaxed_pdb_lines = []
    prediction_times = []
    model_names = []
    files = file_manager(prefix, result_dir)
    seq_len = sum(sequences_lengths)

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    # Create a pool of worker processes
    pool = mp.Pool(num_gpus * jobs_per_gpu)

    # Prepare jobs for parallel execution
    jobs = []
    for seed_num, seed in enumerate(range(random_seed, random_seed + num_seeds)):
        for model_num, (model_name, model_runner, params) in enumerate(model_runner_and_params):
            gpu_id = (seed_num * len(model_runner_and_params) +
                      model_num) % num_gpus
            jobs.append((seed_num, seed, model_num, model_name,
                        model_runner, params, gpu_id))

    # Function to process a single job
    def process_job(job):
        seed_num, seed, model_num, model_name, model_runner, params, gpu_id = job
        torch.cuda.set_device(gpu_id)

        # swap params to avoid recompiling
        model_runner.params = params

        # process input features
        if "multimer" in model_type:
            if model_num == 0 and seed_num == 0:
                input_features = feature_dict.copy()
                input_features["asym_id"] = input_features["asym_id"] - \
                    input_features["asym_id"][..., 0]
        else:
            if model_num == 0:
                input_features = model_runner.process_features(
                    feature_dict, random_seed=seed)
                r = input_features["aatype"].shape[0]
                input_features["asym_id"] = np.tile(
                    feature_dict["asym_id"], r).reshape(r, -1)
                if seq_len < pad_len:
                    input_features = pad_input(input_features, model_runner,
                                               model_name, pad_len, use_templates)
                    print(f"Padding length to {pad_len}")

        tag = f"{model_type}_{model_name}_seed_{seed:03d}"

        # predict
        start = time.time()

        def callback(result, recycles):
            if recycles == 0:
                result.pop("tol", None)
            if not is_complex:
                result.pop("iptm", None)
            print_line = ""
            for x, y in [["mean_plddt", "pLDDT"], ["ptm", "pTM"], ["iptm", "ipTM"], ["tol", "tol"]]:
                if x in result:
                    print_line += f" {y}={result[x]:.3g}"
            print(f"{tag} recycle={recycles}{print_line}")

            if save_recycles:
                final_atom_mask = result["structure_module"]["final_atom_mask"]
                b_factors = result["plddt"][:, None] * final_atom_mask
                unrelaxed_protein = protein.from_prediction(
                    features=input_features,
                    result=result, b_factors=b_factors,
                    remove_leading_feature_dimension=("multimer" not in model_type))
                files.get("unrelaxed", f"r{recycles}.pdb").write_text(
                    protein.to_pdb(unrelaxed_protein))

                if save_all:
                    with files.get("all", f"r{recycles}.pickle").open("wb") as handle:
                        pickle.dump(result, handle)
                del unrelaxed_protein

        return_representations = save_all or save_single_representations or save_pair_representations

        # predict
        result, recycles = \
            model_runner.predict(input_features,
                                 random_seed=seed,
                                 return_representations=return_representations,
                                 callback=callback)

        prediction_time = time.time() - start

        # parse results
        mean_score = result["ranking_confidence"]
        if recycles == 0:
            result.pop("tol", None)
        if not is_complex:
            result.pop("iptm", None)
        print_line = ""
        conf_entry = {}
        for x, y in [["mean_plddt", "pLDDT"], ["ptm", "pTM"], ["iptm", "ipTM"]]:
            if x in result:
                print_line += f" {y}={result[x]:.3g}"
                conf_entry[x] = float(result[x])
        conf_entry["print_line"] = print_line
        print(f"{tag} took {prediction_time:.1f}s ({recycles} recycles)")

        # create protein object
        final_atom_mask = result["structure_module"]["final_atom_mask"]
        b_factors = result["plddt"][:, None] * final_atom_mask
        unrelaxed_protein = protein.from_prediction(
            features=input_features,
            result=result,
            b_factors=b_factors,
            remove_leading_feature_dimension=("multimer" not in model_type))

        # save results
        protein_lines = protein.to_pdb(unrelaxed_protein)

        if save_all:
            with files.get("all", "pickle").open("wb") as handle:
                pickle.dump(result, handle)
        if save_single_representations:
            np.save(files.get("single_repr", "npy"),
                    result["representations"]["single"])
        if save_pair_representations:
            np.save(files.get("pair_repr", "npy"),
                    result["representations"]["pair"])

        # write an easy-to-use format (pAE and pLDDT)
        scores = {}
        with files.get("scores", "json").open("w") as handle:
            plddt = result["plddt"][:seq_len]
            scores = {"plddt": np.around(plddt.astype(float), 2).tolist()}
            if "predicted_aligned_error" in result:
                pae = result["predicted_aligned_error"][:seq_len, :seq_len]
                scores.update({"max_pae": pae.max().astype(float).item(),
                               "pae": np.around(pae.astype(float), 2).tolist()})
                for k in ["ptm", "iptm"]:
                    if k in conf_entry:
                        scores[k] = np.around(conf_entry[k], 2).item()
                del pae
            del plddt
            json.dump(scores, handle)

        return tag, mean_score, conf_entry, protein_lines, prediction_time, scores

    # Execute jobs in parallel
    results = pool.map(process_job, jobs)

    # Process results
    for tag, mean_score, conf_entry, protein_lines, prediction_time, scores in results:
        mean_scores.append(mean_score)
        conf.append(conf_entry)
        unrelaxed_pdb_lines.append(protein_lines)
        prediction_times.append(prediction_time)
        model_names.append(tag)

        files.set_tag(tag)
        files.get("unrelaxed", "pdb").write_text(protein_lines)

        # early stop criteria fulfilled
        if mean_score > stop_at_score:
            break

    pool.close()
    pool.join()

    # rerank models based on predicted confidence
    rank, metric = [], []
    result_files = []
    print(f"reranking models by '{rank_by}' metric")
    model_rank = np.array(mean_scores).argsort()[::-1]
    for n, key in enumerate(model_rank):
        metric.append(conf[key])
        tag = model_names[key]
        files.set_tag(tag)
        # save relaxed pdb
        if n < num_relax:
            start = time.time()
            pdb_lines = relax_me(
                pdb_lines=unrelaxed_pdb_lines[key],
                max_iterations=relax_max_iterations,
                tolerance=relax_tolerance,
                stiffness=relax_stiffness,
                max_outer_iterations=relax_max_outer_iterations,
                use_gpu=use_gpu_relax)
            files.get("relaxed", "pdb").write_text(pdb_lines)
            print(f"Relaxation took {(time.time() - start):.1f}s")

        # rename files to include rank
        new_tag = f"rank_{(n+1):03d}_{tag}"
        rank.append(new_tag)
        print(f"{new_tag}{metric[-1]['print_line']}")
        for x, ext, file in files.files[tag]:
            new_file = result_dir.joinpath(f"{prefix}_{x}_{new_tag}.{ext}")
            file.rename(new_file)
            result_files.append(new_file)

    return {"rank": rank,
            "metric": metric,
            "result_files": result_files}


def pad_input(
    input_features: Dict[str, Any],
    model_runner: Any,
    model_name: str,
    pad_len: int,
    use_templates: bool,
) -> Dict[str, Any]:
    from folding.core.features import make_fixed_size

    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa
    # templates models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = max_msa_clusters

    max_msa_clusters = pad_msa_clusters

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
        extra_msa_size=max_extra_msa,  # extra_msa (4, 5120, 68)
        num_res=pad_len,  # aatype (4, 68)
        num_templates=4,
    )  # template_mask (4, 4) second value
    return input_fix
