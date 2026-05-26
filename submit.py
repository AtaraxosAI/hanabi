#!/usr/bin/env /iliad/u/hengyuan/miniconda3/envs/ptn/bin/python
import pyrallis
import os
import random
from dataclasses import dataclass
from typing import Optional
import pprint
import yaml
import shlex

import submitit


@dataclass
class Config:
    # run related
    dry: int = 1
    save_dir: str = ""
    # pass_save_dir: int = 1
    prefix: Optional[str] = None
    # compute related, the easiest way is to just specify a config
    compute: Optional[str] = None
    partition: Optional[str] = None
    # program related
    # option1: specify main and use args
    ddp: int = 0  # may be inferred
    main: Optional[str] = None
    args: Optional[str] = None  # args should support --args x=1,2,3 y=1,2 z=1
    # option2: just specify a command
    cmd: Optional[str] = None

    def __post_init__(self):
        self._infer_save_dir()

    def _infer_save_dir(self):
        if self.save_dir:
            return

        if self.cmd is not None:
            tokens = shlex.split(self.cmd)
            if "--save_dir" in tokens:
                idx = tokens.index("--save_dir")
                self.save_dir = tokens[idx + 1]
            elif "--config_path" in tokens:
                idx = tokens.index("--config_path")
                cfg = yaml.safe_load(open(tokens[idx + 1], "r"))
                if "save_dir" in cfg:
                    self.save_dir = cfg["save_dir"]

            print(f"inferred save_dir as {self.save_dir}")


def script_accepts_save_dir(script_path: str) -> bool:
    import ast

    with open(script_path, "r") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name) and item.target.id == "save_dir":
                        return True
    return False


def process_main_args(main_args: list[str]):
    from_config = {}

    full_args = [from_config]
    if main_args is None:
        return full_args, []

    override_keys = []
    for arg in main_args:
        new_full_args = []
        key, vals = arg.split("=")
        override_keys.append(key)
        vals = vals.split(",")
        for val in vals:
            for args in full_args:
                new_args = args.copy()
                new_args[key] = val
                new_full_args.append(new_args)
        full_args = new_full_args

    return full_args, override_keys


def get_all_commands(args):
    all_main_args, overrides = process_main_args(args.args.split(" "))

    # get executable
    if args.ddp == 0:
        base_cmd = ["python", "-u", args.main]
    else:
        base_cmd = [
            "torchrun",
            # "--rdzv-backend=c10d",
            # "--rdzv_endpoint=localhost:0",
            f"--nproc-per-node={args.ddp}",
            args.main,
        ]

    pass_save_dir = script_accepts_save_dir(args.main)
    if pass_save_dir:
        print(f"{args.main} need --save_dir, will pass it through.")
    else:
        print(f"{args.main} does not need --save_dir.")

    # construct the sweep
    name2commands = {}
    for main_args in all_main_args:
        cmd = base_cmd.copy()

        if args.ddp:
            if "master_port" in main_args:
                print(f"found master port specified by user: {main_args['master_port']}")
                port = main_args["master_port"]
            else:
                print(f"randomly assign a master port")
                port = random.randint(30000, 60000)
            cmd = [
                "torchrun",
                f"--nproc-per-node={args.ddp}",
                f"--master_port={port}",
                args.main,
            ]

        name_entries = []
        # if args.program_config is not None:
        #     name_entries.append(args.program_config.split("/")[-1].rsplit(".", 1)[0])

        for key, val in main_args.items():
            cmd.append(f"--{key}")
            cmd.append(str(val))
            if key in overrides and key not in ["config_path"]:
                if "." in key:
                    key = key.split(".")[-1]
                    # if "hidden_dim" in keys:
                    #     key = "_".join(keys[-2:])
                    # else:
                    # key = keys[-1]

                # avoid adding path to the val
                if "/" not in val and key not in ["use_wb", "wandb_project", "use_wandb"]:
                    name_entries.append(f"{key}{val}")
                # else:
                #     name_entries.append(f"{key}")

        job_name = "_".join(name_entries)
        if args.prefix is not None:
            job_name = f"{args.prefix}_{job_name}"

        save_dir = os.path.join(args.save_dir, job_name)
        if pass_save_dir:
            cmd.append("--save_dir")
            cmd.append(save_dir)

        name2commands[job_name] = (cmd, save_dir)
    return name2commands


def submit(args: Config, job_name, command, save_dir):
    assert args.compute is not None
    from_config = yaml.safe_load(open(args.compute, "r"))

    if args.ddp > 0:
        from_config["cpus"] *= args.ddp
        from_config["mem_gb"] *= args.ddp
        from_config["gpus"] *= args.ddp

    print("-" * 100)
    print(f"job:\n{job_name}")
    print("compute config:")
    pprint.pprint(from_config)
    print(f"command:\n{' '.join(command)}")
    print("-" * 100)

    os.makedirs(save_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=os.path.join(save_dir, "submitit"))

    if from_config.get("gpu_type") is not None:
        gres = f"gpu:{from_config['gpu_type']}:{from_config['gpus']}"
    elif from_config.get("gpus") is not None:
        gres = f"gpu:{from_config['gpus']}"
    else:
        gres = None

    additional_parameters = {}
    if from_config.get("nodelist") is not None:
        additional_parameters["nodelist"] = from_config["nodelist"]
    if from_config.get("exclude") is not None:
        additional_parameters["exclude"] = from_config["exclude"]

    executor.update_parameters(
        slurm_account=from_config.get("account", "iliad"),
        slurm_partition=args.partition or from_config.get("partition", "iliad"),
        cpus_per_task=from_config["cpus"],
        slurm_gres=gres,
        slurm_mem=f'{from_config["mem_gb"]}gb',
        slurm_time=from_config["time"],
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    if args.dry:
        return None

    job = executor.submit(submitit.helpers.CommandFunction(command))
    return job


def submit_cmd(args: Config):
    assert args.compute is not None, "--compute is required"
    assert args.save_dir, "--save_dir is required"
    assert args.cmd is not None, "--cmd is required"

    compute = yaml.safe_load(open(args.compute, "r"))
    command = shlex.split(args.cmd)

    script_token = next(t for t in command if t.endswith(".py"))
    script_name = script_token.split("/")[-1].replace(".py", "")
    job_name = args.prefix if args.prefix is not None else script_name
    save_dir = args.save_dir

    print("-" * 100)
    print("compute config:")
    pprint.pprint(compute)
    print(f"job: {job_name}, command:\n{' '.join(command)}")
    print("-" * 100)

    os.makedirs(save_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=os.path.join(save_dir, "submitit"))

    if compute.get("gpu_type") is not None:
        gres = f"gpu:{compute['gpu_type']}:{compute['gpus']}"
    elif compute.get("gpus") is not None:
        gres = f"gpu:{compute['gpus']}"
    else:
        gres = None

    additional_parameters = {}
    if compute.get("nodelist") is not None:
        additional_parameters["nodelist"] = compute["nodelist"]
    if compute.get("exclude") is not None:
        additional_parameters["exclude"] = compute["exclude"]

    executor.update_parameters(
        slurm_account=compute.get("account", "iliad"),
        slurm_partition=args.partition or compute.get("partition", "iliad"),
        cpus_per_task=compute["cpus"],
        slurm_gres=gres,
        slurm_mem=f'{compute["mem_gb"]}gb',
        slurm_time=compute["time"],
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    if args.dry:
        return None

    job = executor.submit(submitit.helpers.CommandFunction(command))
    print(f"submitted job {job.job_id}")
    return job


if __name__ == "__main__":
    args = pyrallis.parse(config_class=Config)  # type: ignore

    if args.cmd is not None:
        submit_cmd(args)
    else:
        name2commands = get_all_commands(args)

        jobs = []
        print("will submit these commands:")
        for name, (cmd, save_dir) in name2commands.items():
            job = submit(args, name, cmd, save_dir)
            jobs.append(job)

    if args.dry:
        print("dry run, no job launched!")
    else:
        print("all job launched!")
