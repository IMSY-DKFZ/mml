# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#
import copy
import getpass
import os
import subprocess
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import orjson
import pandas as pd
from dacite import from_dict
from mml_lsf.requirements import LSFSubmissionRequirements
from tqdm.auto import tqdm

from mml.core.scripts.decorators import timeout
from mml.interactive import JobRunner, MMLJobDescription, default_file_manager


class LSFJobRunner(JobRunner):
    """
    A runner that submits jobs to the DKFZ LSF. Under the hood it uses "sshpass" to provide ssh with your password
    (install via "sudp apt install sshpass"). Also supports interactive jobs via
    :class:`~mml_lsf.requirements.LSFSubmissionRequirements`. A single prompt during runner init allows for multiple
    jobs being submitted without typing in password every time.

    To set everything up please read the ``mml_lsf`` README.md. Be aware that sshpass may be a potential security risk,
    although this implementation tries to hide the password via anonymous piping. See "SECURITY CONSIDERATIONS" in
    the manpage of "sshpass" for more details.

    Be also aware that sshpass execution will result in an endless loop if the ECDSA host key matching causes trouble.
    Please make sure the corresponding fingerprints match in advance.
    """

    def __init__(self, user_name: Optional[str] = None, host: Optional[str] = None):
        """
        Sets up the configuration for the runner. May rely either on env variables from "mml.env" or receive values
        directly.

        :param Optional[str] user_name: AD username, if not provided, will try to read MML_AD_USER which should be set
         in mml.env
        :param Optional[str] host: LSF submission host name, if not provided, will try to read MML_CLUSTER_HOST which
         should be set in mml.env
        """
        # check input variables and try to load from environment
        if user_name is None:
            if not os.getenv("MML_AD_USER", None):
                raise ValueError("Either provide user_name or environment variable MML_AD_USER must be set in mml.env")
            user_name = os.getenv("MML_AD_USER")
        self.user_name = user_name
        if host is None:
            if not os.getenv("MML_CLUSTER_HOST", None):
                raise ValueError("Either provide user or environment variable MML_CLUSTER_HOST must be set in mml.env")
            host = os.getenv("MML_CLUSTER_HOST")
        self.host = host
        # will store submitted jobs
        self._cache: Dict[int, MMLJobDescription] = {}
        # received job information
        self._info: Optional[pd.DataFrame] = None
        # read in password once
        self.__pw = getpass.getpass("AD Password:")
        # confirm settings and network availability
        self._test_connection()

    @property
    def all_projects(self) -> Set[str]:
        """
        Lists all unique projects that have been run by this runner. May be used in combination with the retrieve
        method.
        """
        return set(
            [job.config_options["proj"] if "proj" in job.config_options else "default" for job in self._cache.values()]
        )

    def _sshpass_execute(self, cmds: List[str], process_callback: Optional[Callable] = None) -> Any:
        """
        Run the commands via sshpass. Ensure cmds are split as expected by :class:`subprocess.Popen`.

        :param cmds: list of commands to execute, e.g. ["ssh", "USER@HOST", "ls"]
        :param process_callback: a callback while the process is running, will receive the running process as single
         arg and may return something that will be returned by this method in the end. If none, will print the output.
        :return: whatever is returned by process_callback
        """
        if process_callback is None:
            # default process callback
            def process_callback(process) -> None:
                # read in output
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")

        if "sshpass" in cmds:
            raise ValueError("Do not provide sshpass in commands.")
        if len(cmds) == 0:
            raise ValueError("No commands provided.")
        # set up pipe
        rpipe, wpipe = os.pipe()
        try:
            os.set_inheritable(rpipe, True)
            os.set_blocking(wpipe, False)
            try:
                # put password into pipe
                os.write(wpipe, self.__pw.encode())
            finally:
                # make sure to close pipe
                os.close(wpipe)
            # use sshpass and start process
            process = subprocess.Popen(
                ["sshpass"] + cmds,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                stdin=rpipe,
            )
            callback_output = process_callback(process)
            process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                # may be due to error in ssh connection, or failing command
                raise RuntimeError("sshpass execution failed.")
        finally:
            # make sure to close pipe
            os.close(rpipe)
        return callback_output

    def _test_connection(self) -> None:
        """
        Run a dummy command via sshpass. Raises an error in case connection is not established.
        """

        @timeout(seconds=15)
        def test_impl():
            """Timeout wrapped connection test execution."""
            try:
                self._sshpass_execute(["ssh", f"{self.user_name}@{self.host}", "pwd"], process_callback=lambda _: None)
            except RuntimeError:
                raise RuntimeError(
                    f"SSH connection failed. Username: {self.user_name}, Host: {self.host}. You may "
                    f"want to re-instantiate the runner to exclude a typo while entering the password."
                )

        try:
            test_impl()
        except TimeoutError:
            raise RuntimeError(
                "Connection timed out. Is the submission host reachable? Also ECDSA host key matching "
                "may causes trouble. Please make sure the corresponding fingerprints match in advance."
            )
        print("SSH connection established.")

    def run(self, job: MMLJobDescription, verbose: bool = True):
        """
        Submits a given MMLJob to the LSF Cluster. Caches the job and also catches the LSF JOB ID (for non-interactive
        jobs).

        :param MMLJobDescription job: job to be submitted
        :param bool verbose: if True prints the response from submission host
        :return: the LSF return message(s) are printed, no value is returned
        """
        if not isinstance(job.prefix_req, LSFSubmissionRequirements):
            raise TypeError("Job requirements must be of type LSFSubmissionRequirements.")

        cmds = ["ssh", f"{self.user_name}@{self.host}", job.render()]

        def extract_job_id(process):
            job_id = "0"  # backup job id (e.g. interactive jobs)
            # print job output
            for line in iter(process.stdout.readline, ""):
                # a successful submission is answered with a message
                # Job <XXXXXXXX> is submitted to queue <ABC>.
                if line.startswith("Job <") and "> is submitted to queue" in line:
                    job_id = line[line.find("<") + 1 : line.find(">")]
                if verbose:
                    print(line, end="")
            return job_id

        job_id = self._sshpass_execute(cmds=cmds, process_callback=extract_job_id)

        # now cache submission
        self._cache[int(job_id)] = copy.deepcopy(job)

    def kill(self, jobid: Optional[int] = None, pend_only: bool = False) -> List[int]:
        """
        Kills a given job, or all previously submitted jobs. Note that only jobs from this runner will ever be affected.
        For an emergency shutdown of all your jobs, send <bkill 0> manually to the cluster. May call :meth:`info`
        to update all current job status.

        :param jobid: a specific jobid, from a job submitted by this runner, or all jobs if None, the latter will
            require manual confirmation by typing "KILL" once requested
        :param pend_only: if True, will try to only kill jobs that are still pending and not affect running jobs
        :return: list of job ids for all jobs killed
        """
        if len(self._cache) == 0:
            raise RuntimeError("Cannot kill any job, since no jobs have been submitted with this runner before.")
        if jobid is None:
            response = getpass.getpass("Confirm killing all jobs by typing 'KILL':")
            if response.lower() != "kill":
                raise RuntimeError("Killing all jobs aborted.")
            print("Updating job status info.")
            current_info = self.info()
            if pend_only:
                current_info = current_info[current_info["status"] == "PEND"]
            if len(current_info) == 0:
                print("No jobs to kill after applying given filters. You may want to re-check with info method.")
                return
            jobid = current_info.index.tolist()
        elif jobid not in self._cache:
            raise ValueError(f"Job id {jobid} not found in cache. Any runner only handles its own jobs.")
        else:
            jobid = [jobid]
            if pend_only:
                warnings.warn("pend_only option is ignored when killing a single job.")

        for _id in jobid:
            cmds = ["ssh", f"{self.user_name}@{self.host}", "bkill", str(_id)]
            try:
                self._sshpass_execute(cmds=cmds)
            except RuntimeError:
                print(f"Could not kill job {_id}.")
                continue

        print(f"Killed {len(jobid)} job(s). It is recommended to wait ~20 seconds for shutdown before updating job "
              f"status info.")
        return jobid

    def info(self) -> pd.DataFrame:
        """
        Ask LSF about the status of submitted jobs. Shows some info about them. Be aware that the runner will only show
        info on jobs submitted by itself!

        Side effects: prints some general summary information, updates the internal :attr:`_info` dataframe

        :return: a reduced dataframe of self._info with only relevant entries
        """
        if len(self._cache) == 0:
            print("No jobs cached so far.")
            return pd.DataFrame()
        print(f"{len(self._cache)} jobs submitted with this runner. Retrieving updates...")

        cmds = ["ssh", f"{self.user_name}@{self.host}", "bjobs", "-a"]

        def parse_jobs(process):
            # read in job output
            all_jobs = []
            for line in iter(process.stdout.readline, ""):
                # the header should look like
                # JOBID      USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
                if line.startswith("JOBID"):
                    # header line
                    continue
                # parse through output
                parts = line.split(" ")
                entry = []
                for part in parts:
                    stripped = part.strip()
                    if len(stripped) == 0:
                        continue
                    entry.append(stripped)
                    if len(entry) == 6:
                        # we omit job name and submit time, they are harder to parse
                        break
                candidate = {
                    "job_id": int(entry[0]),
                    "user": entry[1],
                    "status": entry[2],
                    "queue": entry[3],
                    "from": entry[4],
                    "exec": entry[5],
                }
                if candidate["status"] == "PEND":
                    candidate["exec"] = ""  # since the string is empty parts of the job name mix in
                all_jobs.append(candidate)
            return all_jobs

        all_jobs = self._sshpass_execute(cmds=cmds, process_callback=parse_jobs)
        # now store information
        new_info = pd.DataFrame(all_jobs)
        new_info.set_index("job_id", inplace=True)
        if self._info is None:
            self._info = new_info
        else:
            # add rows for new entries
            self._info = self._info.reindex(new_info.index.union(self._info.index))
            # update
            self._info.update(new_info)
        # show feedback
        no_info = []
        has_info = []
        for job_id in self._cache:
            if job_id not in self._info.index:
                no_info.append(job_id)
            else:
                has_info.append(job_id)
        relevant_info = self._info.loc[has_info]
        if len(no_info) > 0:
            print(f"CAUTION: No job info for job IDs {no_info} ({len(no_info)} jobs in total).")
        if "EXIT" in relevant_info["status"].to_list():
            print(
                "CAUTION: There are failed jobs (marked as status EXIT). You might want to re-submit "
                "(use the resubmit method)."
            )
        for status, count in Counter(relevant_info["status"].to_list()).items():
            print(f" - {status}: {count} jobs")
        return relevant_info[["status", "queue", "exec"]]

    def resubmit(self, pbar: bool = True) -> None:
        """
        Allows to automatically re-submit failed jobs. The internal job cache is updated.

        :param bool pbar: activates a progress bar and deactivates verbosity of the :meth:`run` method
        :return: no return value
        """
        if self._info is None:
            print("No job info so far. Please call info() method first.")
            return
        failed_ids = []
        for job_id in self._cache:
            if job_id in self._info.index and self._info.loc[job_id]["status"] == "EXIT":
                failed_ids.append(job_id)
        if len(failed_ids) == 0:
            print("No failed jobs.")
            return
        print(f"{len(failed_ids)} jobs failed jobs found. Try resubmitting...")
        for job_id in tqdm(failed_ids, desc="Submitting jobs", disable=not pbar):
            self.run(self._cache[job_id], verbose=not pbar)
            # if this was successful, we remove the old entry
            self._cache.pop(job_id)
        print("All previously failed jobs re-submitted.")

    def peek(self, jobid: Optional[int] = None, watch: bool = False, max_length: Optional[int] = 10) -> None:
        """
        Runs the LSF `bpeek` command (see
        `LSF documentation <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=reference-bpeek>`_). And redirects the
        output to the interactive kernel.

        :param jobid: a specific LSF jobid to peek, if not provided uses the oldest currently running job, that was
         submitted by this runner (also make sure to run :meth:`info` before)
        :param watch: stay connected to the job until interrupted or the job ends, otherwise only a snapshot is sent
        :param max_length: if given an int, reduces the output the the most recent lines (like ``tail``), ignored when
         watch is used
        :return:
        """
        # jobid None maps to oldest running RUN job (make sure info has run before)
        if jobid is None:
            if self._info is None:
                raise RuntimeError("Please run info() first to use auto-jobid mechanism. Or provide a jobid.")
            run_ids = []
            for _job_id in self._cache:
                if _job_id in self._info.index and self._info.loc[_job_id]["status"] == "RUN":
                    run_ids.append(_job_id)
            if len(run_ids) == 0:
                print("No running jobs. Check job status by calling info().")
                return
            jobid = min(run_ids)

        cmds = ["ssh", f"{self.user_name}@{self.host}", "bpeek"]
        # watch adds -f flag and streams
        if watch:
            cmds += ["-f"]
        cmds += [str(jobid)]

        def peek_process_callback(process) -> Optional[deque]:
            # max_length (ignored when watch) cuts the history
            if not watch and max_length is not None:
                print_deque = deque(maxlen=max_length)
            else:
                print_deque = None
            # read in output
            try:
                for line in iter(process.stdout.readline, ""):
                    if print_deque is not None:
                        print_deque.append(line)
                    else:
                        print(line, end="")
            except KeyboardInterrupt:
                if watch:
                    print("Interrupted watching.")
                else:
                    raise
            return print_deque

        print(f"Peeking into job {jobid}:")
        print("-----------------------------")
        print(self._cache[jobid])
        print("-----------------------------")
        potential_deque = self._sshpass_execute(cmds=cmds, process_callback=peek_process_callback)
        if potential_deque is not None:
            while len(potential_deque) > 0:
                print(potential_deque.popleft(), end="")

    def dump(self, id: Optional[str] = None, overwrite: bool = False) -> str:
        """
        Dumps the cache of the runner, permanently storing submitted jobs and their respective LSF jobids. Can be fully
        reverted by the :meth:`load` method to create a compatible internal :attr:`_cache`.

        :param id: an identifier for the cache, will not overwrite existing cache for the same identifier (and raise an
         ValueError instead), if not provided (i.e., None) will use current datetime string
        :param overwrite: if True, will overwrite existing cache for the same identifier
        :return: returns the identifier
        """
        if id is None:
            id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with default_file_manager() as fm:
            path = fm.construct_saving_path(obj=None, key="lsf_jobs", file_name=id)
            if path.exists():
                if not overwrite:
                    raise ValueError(f"ID {id} already exists as LSF backup.")
                else:
                    print(f"Overwriting existing cache with ID {id}.")
                    path.unlink()
            # must ensure str keys for orjson
            reformatted = {str(job_id): job_desc for job_id, job_desc in self._cache.items()}
            with open(path, "wb") as f:
                f.write(orjson.dumps(reformatted))
        print(f"Stored cache with {len(self._cache)} jobs. Run `load(id={id})` to restore at a later point.")
        return path.name

    def load(self, id: str) -> None:
        """
        Loads the cache of a previous :meth:`dump` call to restore the runner cache.

        :param id: the identifier used to store the cache
        :return: no return value as :attr:`_cache` is restored in place
        """
        if self._cache:
            raise RuntimeError(
                "This runner has a non-empty cache. To avoid loss of information please create a new "
                "runner instance from scratch to load."
            )
        with default_file_manager() as fm:
            path = fm.construct_saving_path(obj=None, key="lsf_jobs", file_name=id)
            if not path.exists():
                raise ValueError(f"ID {id} does not exists as LSF backup.")
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        for k, v in data.items():
            v["prefix_req"] = from_dict(data_class=LSFSubmissionRequirements, data=v["prefix_req"])
            self._cache[int(k)] = from_dict(data_class=MMLJobDescription, data=v)
        print(f"Restored cache with {len(self._cache)} jobs. Run `info()` to get current job status.")

    def retrieve(self, project: str, excludes: Sequence[str] = ("PARAMETERS", "hpo", "runs")) -> None:
        """
        Retrieve results from the LSF cluster for a given project.

        :param project: name of the project to synchronize
        :param excludes: subfolders to be excluded, usually these comprise intermediates, log files or very large files
        :return: no return value
        """
        if " " in project or "/" in project:
            raise ValueError("Invalid project name.")
        # try to load env variables
        if not os.getenv("MML_CLUSTER_WORKER", None):
            raise ValueError("Environment variable MML_CLUSTER_WORKER must be set in mml.env")
        worker = os.getenv("MML_CLUSTER_WORKER")
        if not os.getenv("MML_CLUSTER_RESULTS_PATH", None):
            raise ValueError("Environment variable MML_CLUSTER_RESULTS_PATH must be set in mml.env")
        cluster_path = os.getenv("MML_CLUSTER_RESULTS_PATH")
        if not os.getenv("MML_RESULTS_PATH", None):
            raise ValueError("Environment variable MML_RESULTS_PATH must be set in mml.env")
        local_path = os.getenv("MML_RESULTS_PATH")
        # build bash string
        cmds = ["rsync", "-rtvu", "--stats"]
        for dir in excludes:
            cmds.append(f"--exclude={dir}")
        cmds.append(f"{self.user_name}@{worker}:{cluster_path}/{project}/")
        cmds.append(f"{local_path}/{project}")
        print(f"Executing {' '.join(cmds)}")
        self._sshpass_execute(cmds=cmds)
