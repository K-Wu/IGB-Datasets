# Launcher for use in slurm script (srun)
# From https://github.com/dmlc/dgl/blob/e181ef15d7df465af713472ddcc8fba9b233f708/tools/launch.py
# Usage: https://docs.dgl.ai/tutorials/dist/1_node_classification.html#launch-the-distributed-training-job
"""Launching tool for use in slurm script for DGL distributed training"""
import argparse
import json
import logging
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import time
# from functools import partial
from threading import Thread
from typing import Optional, List, Dict


def get_ip_address_to_node_mapping() -> Dict[str, str]:
    # Get node lists via scontrol show hostnames $SLURM_JOB_NODELIST
    node_lists = subprocess.check_output(
        f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']}",
        shell=True,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    node_lists = node_lists.split("\n")
    results = {}
    for node in node_lists:
        ip = subprocess.check_output("getent ahostsv4 "+ node+ "-ib0 | grep STREAM | head -n 1  | awk '{ print $1 }'", shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        results[ip] = node
    return results



def cleanup_proc(get_all_remaining_steps, conn):
    """This process tries to clean up the remote training tasks."""
    print("cleanup process runs")
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == "exit":
        sys.exit(0)
    else:
        # CHANGE: get step ids instead get_all_remote_pids -> get_all_remaining_steps
        remaining_steps = get_all_remaining_steps()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for step in remaining_steps:
            # CHANGE: kill_process -> kill_step
            kill_step(step)
    print("cleanup process exits")

def kill_step(step_id):
    """kill the specified step via scancel"""
    job_id = os.environ.get("SLURM_JOB_ID")
    cmd = "scancel --signal=KILL {}.{}".format(job_id, step_id)
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        print("Error when killing step {}: {}".format(step_id, e))



def execute_remote(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
) -> Thread:
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        state_q: A queue collecting Thread exit states.
        ip: The ip-address of the host to run the command on.
        thread_list:
        username: Optional. If given, this will specify a username to use when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a username to avoid permission issues.

    Returns:
        thread: The Thread whose run() is to run the `cmd` on the remote host. Returns when the cmd completes
            on the remote host.
    """
    # CHANGE: switch ssh execute to srun -w=<nodelist>
    # Construct srun command that executes `cmd` on the select node as one process
    srun_cmd = "srun -w {node_id} -ln1 --overlap {cmd}".format(
        node_id=get_ip_address_to_node_mapping()[ip],
        cmd=cmd,
    )

    # thread func to run the job
    def run(srun_cmd, state_q):
        try:
            subprocess.check_call(srun_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception:
            state_q.put(-1)

    thread = Thread(
        target=run,
        args=(
            srun_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread


def get_all_remaining_steps() -> List[int]:
    """Get the still-running srun steps."""
    # List all active jobs via sacct -j40313 --format=JobID,Start,End,Elapsed,NCPUS
    # All active jobs are 1) with JobID in the form of 40313.<step_id> and 2) with End=Unknown
    sacct_output = subprocess.check_output(
        "sacct --format=JobID,Start,End,Elapsed,NCPUS",
        shell=True,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    results = []
    for line in sacct_output.split("\n"):
        jobid, start, end, elapsed, ncpus = line.strip().split()
        if jobid.startswith(os.environ["SLURM_JOB_ID"] + ".") and end.lower() == "unknown":
            results.append(jobid.split(".")[1])
    return results


def construct_torch_dist_launcher_cmd(
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Constructs the torch distributed launcher command.
    Helper function.

    Args:
        num_trainers:
        num_nodes:
        node_rank:
        master_addr:
        master_port:

    Returns:
        cmd_str.
    """
    torch_cmd_template = (
        "-m torch.distributed.run "
        "--nproc_per_node={nproc_per_node} "
        "--nnodes={nnodes} "
        "--node_rank={node_rank} "
        "--master_addr={master_addr} "
        "--master_port={master_port}"
    )
    return torch_cmd_template.format(
        nproc_per_node=num_trainers,
        nnodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


def wrap_udf_in_torch_dist_launcher(
    udf_command: str,
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Wraps the user-defined function (udf_command) with the torch.distributed.run module.

     Example: if udf_command is "python3 run/some/trainer.py arg1 arg2", then new_df_command becomes:
         "python3 -m torch.distributed.run <TORCH DIST ARGS> run/some/trainer.py arg1 arg2

    udf_command is assumed to consist of pre-commands (optional) followed by the python launcher script (required):
    Examples:
        # simple
        python3.7 path/to/some/trainer.py arg1 arg2

        # multi-commands
        (cd some/dir && python3.7 path/to/some/trainer.py arg1 arg2)

    IMPORTANT: If udf_command consists of multiple python commands, then this will result in undefined behavior.

    Args:
        udf_command:
        num_trainers:
        num_nodes:
        node_rank:
        master_addr:
        master_port:

    Returns:

    """
    torch_dist_cmd = construct_torch_dist_launcher_cmd(
        num_trainers=num_trainers,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    # Auto-detect the python binary that kicks off the distributed trainer code.
    # Note: This allowlist order matters, this will match with the FIRST matching entry. Thus, please add names to this
    #       from most-specific to least-specific order eg:
    #           (python3.7, python3.8) -> (python3)
    # The allowed python versions are from this: https://www.dgl.ai/pages/start.html
    python_bin_allowlist = (
        "python3.6",
        "python3.7",
        "python3.8",
        "python3.9",
        "python3",
        # for backwards compatibility, accept python2 but technically DGL is a py3 library, so this is not recommended
        "python2.7",
        "python2",
    )
    # If none of the candidate python bins match, then we go with the default `python`
    python_bin = "python"
    for candidate_python_bin in python_bin_allowlist:
        if candidate_python_bin in udf_command:
            python_bin = candidate_python_bin
            break

    # transforms the udf_command from:
    #     python path/to/dist_trainer.py arg0 arg1
    # to:
    #     python -m torch.distributed.run [DIST TORCH ARGS] path/to/dist_trainer.py arg0 arg1
    # Note: if there are multiple python commands in `udf_command`, this may do the Wrong Thing, eg launch each
    #       python command within the torch distributed launcher.
    new_udf_command = udf_command.replace(
        python_bin, f"{python_bin} {torch_dist_cmd}"
    )

    return new_udf_command


def construct_dgl_server_env_vars(
    num_samplers: int,
    num_server_threads: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL server-specific env vars string that are required for DGL code to behave in the correct
    server role.
    Convenience function.

    Args:
        num_samplers:
        num_server_threads:
        tot_num_clients:
        part_config: Partition config.
            Relative path to workspace.
        ip_config: IP config file containing IP addresses of cluster hosts.
            Relative path to workspace.
        num_servers:
        graph_format:
        pythonpath: Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        server_env_vars: The server-specific env-vars in a string format, friendly for CLI execution.

    """
    server_env_vars_template = (
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "OMP_NUM_THREADS={OMP_NUM_THREADS} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "{suffix_optional_envvars}"
    )
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return server_env_vars_template.format(
        DGL_ROLE="server",
        DGL_NUM_SAMPLER=num_samplers,
        OMP_NUM_THREADS=num_server_threads,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        suffix_optional_envvars=suffix_optional_envvars,
    )


def construct_dgl_client_env_vars(
    num_samplers: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    num_omp_threads: int,
    group_id: int,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL client-specific env vars string that are required for DGL code to behave in the correct
    client role.
    Convenience function.

    Args:
        num_samplers:
        tot_num_clients:
        part_config: Partition config.
            Relative path to workspace.
        ip_config: IP config file containing IP addresses of cluster hosts.
            Relative path to workspace.
        num_servers:
        graph_format:
        num_omp_threads:
        group_id:
            Used in client processes to indicate which group it belongs to.
        pythonpath: Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        client_env_vars: The client-specific env-vars in a string format, friendly for CLI execution.

    """
    client_env_vars_template = (
        "DGL_DIST_MODE={DGL_DIST_MODE} "
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "OMP_NUM_THREADS={OMP_NUM_THREADS} "
        "DGL_GROUP_ID={DGL_GROUP_ID} "
        "{suffix_optional_envvars}"
    )
    # append optional additional env-vars
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return client_env_vars_template.format(
        DGL_DIST_MODE="distributed",
        DGL_ROLE="client",
        DGL_NUM_SAMPLER=num_samplers,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        OMP_NUM_THREADS=num_omp_threads,
        DGL_GROUP_ID=group_id,
        suffix_optional_envvars=suffix_optional_envvars,
    )


def wrap_cmd_with_local_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired env vars with the following properties:
    (1) env vars persist for the entire `cmd`, even if it consists of multiple "chained" commands like:
        cmd = "ls && pwd && python run/something.py"
    (2) env vars don't pollute the environment after `cmd` completes.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_with_local_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A string containing env vars, eg "VAR1=val1 VAR2=val2"

    Returns:
        cmd_with_env_vars:

    """
    # use `export` to persist env vars for entire cmd block. required if udf_command is a chain of commands
    # also: wrap in parens to not pollute env:
    #     https://stackoverflow.com/a/45993803
    return f"(export {env_vars}; {cmd})"


def wrap_cmd_with_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra env vars

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_with_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A list of strings containing env vars, e.g., ["VAR1=value1", "VAR2=value2"]

    Returns:
        cmd_with_env_vars:
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_with_local_envvars(cmd, env_vars)


def get_available_port(ip):
    """Get available port with specified ip."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(1234, 65535):
        try:
            sock.connect((ip, port))
        except:
            return port
    raise RuntimeError("Failed to get available port for ip~{}".format(ip))


def submit_jobs(args, udf_command, dry_run=False):
    """Submit distributed jobs (server and client processes) via ssh"""
    if dry_run:
        print(
            "Currently it's in dry run mode which means no jobs will be launched."
        )
    servers_cmd = []
    clients_cmd = []
    hosts = [] # TODO: change this to node id
    thread_list = []
    server_count_per_machine = 0

    # Get the IP addresses of the cluster.
    ip_config = os.path.join(args.workspace, args.ip_config)
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of ip_config.")
            server_count_per_machine = args.num_servers
    # Get partition info of the graph data
    part_config = os.path.join(args.workspace, args.part_config)
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert "num_parts" in part_metadata, "num_parts does not exist."
    # The number of partitions must match the number of machines in the cluster.
    assert part_metadata["num_parts"] == len(
        hosts
    ), "The number of graph partitions has to match the number of machines in the cluster."

    state_q = queue.Queue()
    tot_num_clients = args.num_trainers * (1 + args.num_samplers) * len(hosts)
    # launch server tasks
    server_env_vars = construct_dgl_server_env_vars(
        num_samplers=args.num_samplers,
        num_server_threads=args.num_server_threads,
        tot_num_clients=tot_num_clients,
        part_config=args.part_config,
        ip_config=args.ip_config,
        num_servers=args.num_servers,
        graph_format=args.graph_format,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )
    for i in range(len(hosts) * server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        server_env_vars_cur = f"{server_env_vars} DGL_SERVER_ID={i}"
        cmd = wrap_cmd_with_local_envvars(udf_command, server_env_vars_cur)
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        cmd = "cd " + str(args.workspace) + "; " + cmd
        servers_cmd.append(cmd)
        if not dry_run:
            thread_list.append(
                execute_remote(
                    cmd,
                    state_q,
                    ip,
                )
            )

    # launch client tasks
    client_env_vars = construct_dgl_client_env_vars(
        num_samplers=args.num_samplers,
        tot_num_clients=tot_num_clients,
        part_config=args.part_config,
        ip_config=args.ip_config,
        num_servers=args.num_servers,
        graph_format=args.graph_format,
        num_omp_threads=os.environ.get(
            "OMP_NUM_THREADS", str(args.num_omp_threads)
        ),
        group_id=0,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )

    master_addr = hosts[0][0]
    master_port = get_available_port(master_addr)
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # Transform udf_command to follow torch's dist launcher format: `PYTHON_BIN -m torch.distributed.run ... UDF`
        torch_dist_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=args.num_trainers,
            num_nodes=len(hosts),
            node_rank=node_id,
            master_addr=master_addr,
            master_port=master_port,
        )
        cmd = wrap_cmd_with_local_envvars(
            torch_dist_udf_command, client_env_vars
        )
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        cmd = "cd " + str(args.workspace) + "; " + cmd
        clients_cmd.append(cmd)
        if not dry_run:
            thread_list.append(
                execute_remote(
                    cmd, state_q, ip
                )
            )

    # return commands of clients/servers directly if in dry run mode
    if dry_run:
        return clients_cmd, servers_cmd

    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1, conn2 = multiprocessing.Pipe()
    #func = partial(get_all_remote_pids, hosts, udf_command)
    process = multiprocessing.Process(target=cleanup_proc, args=(get_all_remaining_steps, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info("Stop launcher")
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send("cleanup")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            # Record err_code
            # We record one of the error if there are multiple
            err = err_code

    # The training processes complete. We should tell the cleanup process to exit.
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)


def main():
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument(
        "--workspace",
        type=str,
        help="Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd",
    )
    parser.add_argument(
        "--num_trainers",
        type=int,
        help="The number of trainer processes per machine",
    )
    parser.add_argument(
        "--num_omp_threads",
        type=int,
        help="The number of OMP threads per trainer",
    )
    parser.add_argument(
        "--num_samplers",
        type=int,
        default=0,
        help="The number of sampler processes per trainer process",
    )
    parser.add_argument(
        "--num_servers",
        type=int,
        help="The number of server processes per machine",
    )
    parser.add_argument(
        "--part_config",
        type=str,
        help="The file (in workspace) of the partition config",
    )
    parser.add_argument(
        "--ip_config",
        type=str,
        help="The file (in workspace) of IP configuration for server processes",
    )
    parser.add_argument(
        "--num_server_threads",
        type=int,
        default=1,
        help="The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on \
                        the same machine. By default, it is 1.",
    )
    parser.add_argument(
        "--graph_format",
        type=str,
        default="csc",
        help='The format of the graph structure of each partition. \
                        The allowed formats are csr, csc and coo. A user can specify multiple \
                        formats, separated by ",". For example, the graph format is "csr,csc".',
    )
    parser.add_argument(
        "--extra_envs",
        nargs="+",
        type=str,
        default=[],
        help="Extra environment parameters need to be set. For example, \
                        you can set the LD_LIBRARY_PATH and NCCL_DEBUG by adding: \
                        --extra_envs LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH NCCL_DEBUG=INFO ",
    )
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, "Please provide user command line."
    assert (
        args.num_trainers is not None and args.num_trainers > 0
    ), "--num_trainers must be a positive number."
    assert (
        args.num_samplers is not None and args.num_samplers >= 0
    ), "--num_samplers must be a non-negative number."
    assert (
        args.num_servers is not None and args.num_servers > 0
    ), "--num_servers must be a positive number."
    assert (
        args.num_server_threads > 0
    ), "--num_server_threads must be a positive number."
    assert (
        args.workspace is not None
    ), "A user has to specify a workspace with --workspace."
    assert (
        args.part_config is not None
    ), "A user has to specify a partition configuration file with --part_config."
    assert (
        args.ip_config is not None
    ), "A user has to specify an IP configuration file with --ip_config."
    if args.num_omp_threads is None:
        # Here we assume all machines have the same number of CPU cores as the machine
        # where the launch script runs.
        args.num_omp_threads = max(
            multiprocessing.cpu_count() // 2 // args.num_trainers, 1
        )
        print(
            "The number of OMP threads per trainer is set to",
            args.num_omp_threads,
        )

    udf_command = str(udf_command[0])
    if "python" not in udf_command:
        raise RuntimeError(
            "DGL launching script can only support Python executable file."
        )
    submit_jobs(args, udf_command)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()