import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import os

from folding.core.prediction import predict_structure
from folding.core.msa import get_msa_and_templates, msa_to_str
from folding.core.features import generate_input_feature
from folding.utils.common import safe_filename, get_commit, setup_logging
from folding.models.model_runner import load_models_and_params
from folding.visualization.plots import plot_msa_v2, plot_paes, plot_plddts

logger = logging.getLogger(__name__)

def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
    result_dir: Union[str, Path],
    num_models: int,
    is_complex: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    model_order: List[int] = [1,2,3,4,5],
    num_ensemble: int = 1,
    model_type: str = "auto",
    msa_mode: str = "mmseqs2_uniref_env",
    use_templates: bool = False,
    custom_template_path: str = None,
    num_relax: int = 0,
    relax_max_iterations: int = 0,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired_paired",
    pairing_strategy: str = "greedy",
    data_dir: Union[str, Path] = ".",
    host_url: str = "",
    user_agent: str = "",
    random_seed: int = 0,
    num_seeds: int = 1,
    recompile_padding: Union[int, float] = 10,
    zip_results: bool = False,
    prediction_callback: Optional[callable] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    jobname_prefix: Optional[str] = None,
    save_all: bool = False,
    save_recycles: bool = False,
    use_dropout: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score: float = 100,
    dpi: int = 200,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    use_cluster_profile: bool = True,
    num_gpus: int = 1,
    jobs_per_gpu: int = 1,
):
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)

    # Set up GPU environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(num_gpus)])
    
    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "num_relax": num_relax,
        "relax_max_iterations": relax_max_iterations,
        "relax_tolerance": relax_tolerance,
        "relax_stiffness": relax_stiffness,
        "relax_max_outer_iterations": relax_max_outer_iterations,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
        "num_ensemble": num_ensemble,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "max_seq": max_seq,
        "max_extra_seq": max_extra_seq,
        "pair_mode": pair_mode,
        "pairing_strategy": pairing_strategy,
        "host_url": host_url,
        "user_agent": user_agent,
        "stop_at_score": stop_at_score,
        "random_seed": random_seed,
        "num_seeds": num_seeds,
        "recompile_padding": recompile_padding,
        "commit": get_commit(),
        "use_dropout": use_dropout,
        "use_cluster_profile": use_cluster_profile,
        "num_gpus": num_gpus,
        "jobs_per_gpu": jobs_per_gpu,
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))

    pad_len = 0
    ranks, metrics = [], []
    first_job = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        if jobname_prefix is not None:
            jobname = safe_filename(jobname_prefix) + f"_{job_number:03d}"
        else:
            jobname = safe_filename(raw_jobname)

        logger.info(f"Query {job_number + 1}/{len(queries)}: {jobname}")

        try:
            (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = \
                get_msa_and_templates(jobname, query_sequence, a3m_lines, result_dir, msa_mode, use_templates,
                                      custom_template_path, pair_mode, pairing_strategy, host_url, user_agent)

            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

            (feature_dict, domain_names) = \
                generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                       template_features, is_complex, model_type, max_seq=max_seq)

            msa_plot = plot_msa_v2(feature_dict, dpi=dpi)
            msa_plot.savefig(str(result_dir.joinpath(f"{jobname}_coverage.png")), bbox_inches='tight')
            msa_plot.close()

            if num_models > 0:
                if first_job:
                    model_runner_and_params = load_models_and_params(
                        num_models=num_models,
                        use_templates=use_templates,
                        num_recycles=num_recycles,
                        num_ensemble=num_ensemble,
                        model_order=model_order,
                        model_type=model_type,
                        data_dir=data_dir,
                        stop_at_score=stop_at_score,
                        rank_by=rank_by,
                        use_dropout=use_dropout,
                        max_seq=max_seq,
                        max_extra_seq=max_extra_seq,
                        use_cluster_profile=use_cluster_profile,
                        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                        use_fuse=True,
                        use_bfloat16=True,
                        save_all=save_all,
                    )
                    first_job = False

                results = predict_structure(
                    prefix=jobname,
                    result_dir=result_dir,
                    feature_dict=feature_dict,
                    is_complex=is_complex,
                    use_templates=use_templates,
                    sequences_lengths=[len(seq) for seq in query_seqs_unique],
                    pad_len=pad_len,
                    model_type=model_type,
                    model_runner_and_params=model_runner_and_params,
                    num_relax=num_relax,
                    relax_max_iterations=relax_max_iterations,
                    relax_tolerance=relax_tolerance,
                    relax_stiffness=relax_stiffness,
                    relax_max_outer_iterations=relax_max_outer_iterations,
                    rank_by=rank_by,
                    stop_at_score=stop_at_score,
                    prediction_callback=prediction_callback,
                    use_gpu_relax=use_gpu_relax,
                    random_seed=random_seed,
                    num_seeds=num_seeds,
                    save_all=save_all,
                    save_single_representations=save_single_representations,
                    save_pair_representations=save_pair_representations,
                    save_recycles=save_recycles,
                    num_gpus=num_gpus,
                    jobs_per_gpu=jobs_per_gpu,
                )

                ranks.append(results["rank"])
                metrics.append(results["metric"])

                # Generate plots
                scores = []
                for r in results["rank"][:5]:
                    scores_file = result_dir.joinpath(f"{jobname}_scores_{r}.json")
                    with scores_file.open("r") as handle:
                        scores.append(json.load(handle))

                if "pae" in scores[0]:
                    paes_plot = plot_paes([x["pae"] for x in scores],
                        [len(seq) for seq in query_seqs_unique], dpi=dpi)
                    paes_plot.savefig(str(result_dir.joinpath(f"{jobname}_pae.png")), bbox_inches='tight')
                    paes_plot.close()

                plddt_plot = plot_plddts([x["plddt"] for x in scores],
                    [len(seq) for seq in query_seqs_unique], dpi=dpi)
                plddt_plot.savefig(str(result_dir.joinpath(f"{jobname}_plddt.png")), bbox_inches='tight')
                plddt_plot.close()

        except Exception as e:
            logger.exception(f"Error processing {jobname}: {e}")

    logger.info("Done")
    return {"rank": ranks, "metric": metrics}

def set_model_type(is_complex: bool, model_type: str) -> str:
    if model_type == "auto":
        if is_complex:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    return model_type

def main():
    parser = argparse.ArgumentParser(description="Run CyberChipped-Fold predictions")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("results", help="Results output directory")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models to run")
    parser.add_argument("--use_templates", action="store_true", help="Use templates")
    parser.add_argument("--msa_mode", default="mmseqs2_uniref_env", help="MSA mode")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--jobs_per_gpu", type=int, default=1, help="Number of jobs to run on each GPU")
    
    args = parser.parse_args()

    setup_logging(Path(args.results).joinpath("log.txt"))

    # Here you would implement the logic to parse the input and create the queries
    queries = []  # This should be populated based on the input

    run(queries=queries,
        result_dir=args.results,
        num_models=args.num_models,
        is_complex=False,  # You might want to determine this based on the input
        use_templates=args.use_templates,
        msa_mode=args.msa_mode,
        num_gpus=args.num_gpus,
        jobs_per_gpu=args.jobs_per_gpu)

if __name__ == "__main__":
    main()