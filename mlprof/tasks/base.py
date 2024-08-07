# coding: utf-8

"""
Generic tools and base tasks that are defined along typical objects in an analysis.
"""

from __future__ import annotations

import os

import luigi  # type: ignore[import-untyped]
import law  # type: ignore[import-untyped]

from collections import OrderedDict


class BaseTask(law.SandboxTask):

    version = luigi.Parameter(
        description="mandatory version that is encoded into output paths",
    )

    allow_empty_sandbox = True
    sandbox: str | None = None

    task_namespace = None
    local_workflow_require_branches = False
    output_collection_cls = law.SiblingFileCollection

    def store_parts(self):
        """
        Returns a :py:class:`law.util.InsertableDict` whose values are used to create a store path. For instance, the
        parts ``{"keyA": "a", "keyB": "b", 2: "c"}`` lead to the path "a/b/c". The keys can be used by subclassing tasks
        to overwrite values.
        """
        parts = law.util.InsertableDict()

        # in this base class, just add the task class name
        parts["task_family"] = self.task_family

        # add the version when set
        if self.version is not None:
            parts["version"] = self.version

        return parts

    def local_path(self, *path, **kwargs):
        """
        Joins path fragments from *store* (defaulting to :py:attr:`default_store`), :py:meth:`store_parts` and *path*,
        and returns the joined path.
        """
        # determine the main store directory
        store = "$MLP_STORE_LOCAL"

        # concatenate all parts that make up the path and join them
        parts = tuple(self.store_parts().values()) + path
        path = os.path.join(store, *(str(p) for p in parts))

        return path

    def local_target(self, *path, **kwargs):
        """ local_target(*path, dir=False, **kwargs)
        Creates either a local file or directory target, depending on *dir*, forwarding all *path* fragments and *store*
        to :py:meth:`local_path` and all *kwargs* the respective target class.
        """
        # select the target class
        cls = law.LocalDirectoryTarget if kwargs.pop("dir", False) else law.LocalFileTarget

        # create the local path
        path = self.local_path(*path)

        # create the target instance and return it
        return cls(path, **kwargs)


class CommandTask(BaseTask):
    """
    A task that provides convenience methods to work with shell commands, i.e., printing them on the command line and
    executing them with error handling.
    """

    print_command = law.CSVParameter(
        default=(),
        significant=False,
        description="print the command that this task would execute but do not run any task; this "
        "CSV parameter accepts a single integer value which sets the task recursion depth to also "
        "print the commands of required tasks (0 means non-recursive)",
    )
    custom_args = luigi.Parameter(
        default="",
        description="custom arguments that are forwarded to the underlying command; they might not "
        "be encoded into output file paths; empty default",
    )

    exclude_index = True
    exclude_params_req = {"custom_args"}
    interactive_params = BaseTask.interactive_params + ["print_command"]

    run_command_in_tmp = False

    def _print_command(self, args):
        from law.task.interactive import fmt_chars, _print_wrapped  # type: ignore[import-untyped]
        from law.util import colored, get_terminal_width  # type: ignore[import-untyped]

        max_depth = int(args[0])

        print(f"print task commands with max_depth {max_depth}")
        print("")

        # get the format chars
        fmt_name = law.config.get_expanded("task", "interactive_format")
        fmt = fmt_chars.get(fmt_name, fmt_chars["fancy"])

        # get the line break setting
        break_lines = law.config.get_expanded_bool("task", "interactive_line_breaks")
        out_width = law.config.get_expanded_int("task", "interactive_line_width")
        print_width = (out_width if out_width > 0 else get_terminal_width()) if break_lines else None
        _print = lambda line, offset, br=1: _print_wrapped(line, print_width if br else None, offset)

        # walk through deps
        parents_last_flags = []
        for dep, next_deps, depth, is_last in self.walk_deps(
            max_depth=max_depth,
            order="pre",
            yield_last_flag=True,
        ):
            del parents_last_flags[depth:]
            next_deps_shown = bool(next_deps) and (max_depth < 0 or depth < max_depth)

            # determine the print common offset
            offset = [(" " if f else fmt["|"]) + fmt["ind"] * " " for f in parents_last_flags[1:]]
            offset = "".join(offset)
            parents_last_flags.append(is_last)

            # print free space
            free_offset = offset + fmt["|"]
            free_lines = "\n".join(fmt["free"] * [free_offset])
            if depth > 0 and free_lines:
                print(free_lines)

            # determine task offset and prefix
            task_offset = offset
            if depth > 0:
                task_offset += fmt["l" if is_last else "t"] + fmt["ind"] * fmt["-"]
            task_prefix = f"{depth} {fmt['>']} "

            # determine text offset and prefix
            text_offset = offset
            if depth > 0:
                text_offset += (" " if is_last else fmt["|"]) + fmt["ind"] * " "
            text_prefix = (len(task_prefix) - 1) * " "
            text_offset += (fmt["|"] if next_deps_shown else " ") + text_prefix
            text_offset_ind = text_offset + fmt["ind"] * " "

            # print the task line
            _print(task_offset + task_prefix + dep.repr(color=True), text_offset)

            # stop when dep has no command
            if not isinstance(dep, CommandTask):
                _print(text_offset_ind + colored("not a CommandTask", "yellow"), text_offset_ind)
                continue

            # when dep is a workflow, take the first branch
            text = law.util.colored("command", style="bright")
            if isinstance(dep, law.BaseWorkflow) and dep.is_workflow():
                dep = dep.as_branch(0)
                text += f" (from branch {law.util.colored('0', 'red')})"
            text += ": "

            cmd = dep.build_command()
            if cmd:
                # when cmd is a 2-tuple, i.e. the real command and a representation for printing
                # pick the second one
                if isinstance(cmd, tuple) and len(cmd) == 2:
                    cmd = cmd[1]
                else:
                    if isinstance(cmd, list):
                        cmd = law.util.quote_cmd(cmd)
                    # defaut highlighting
                    cmd = law.util.colored(cmd, "cyan")
                text += cmd
            else:
                text += law.util.colored("empty", "red")
            _print(text_offset_ind + text, text_offset_ind, br=False)

    def build_command(self):
        # this method should build and return the command to run
        raise NotImplementedError

    def touch_output_dirs(self):
        # keep track of created uris so we can avoid creating them twice
        handled_parent_uris = set()

        for outp in law.util.flatten(self.output()):
            # get the parent directory target
            parent = None
            if isinstance(outp, law.SiblingFileCollection):
                parent = outp.dir
            elif isinstance(outp, law.FileSystemFileTarget):
                parent = outp.parent

            # create it
            if parent and parent.uri() not in handled_parent_uris:
                parent.touch()
                handled_parent_uris.add(parent.uri())

    def run_command(self, cmd, cmd_repr=None, optional=False, **kwargs):
        # proper command encoding
        cmd = (law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd).strip()

        # default command representation
        if not cmd_repr:
            cmd_repr = law.util.colored(cmd, "cyan")

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path
        self.publish_message(f"cwd: {kwargs.get('cwd', os.getcwd())}")

        # call it
        with self.publish_step(f"running '{cmd_repr}' ..."):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            for line in lines:
                print(line)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            raise Exception(f"command failed with exit code {p.returncode}: {cmd}")

        return p

    @law.decorator.log
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self, **kwargs):
        self.pre_run_command()

        # default run implementation
        # first, create all output directories
        self.touch_output_dirs()

        # build the command
        cmd = self.build_command()
        cmd_repr = None
        if isinstance(cmd, (list, tuple)) and len(cmd) == 2:
            cmd, cmd_repr = cmd

        # run it
        self.run_command(cmd, cmd_repr=cmd_repr, **kwargs)

        self.post_run_command()

    def pre_run_command(self):
        return

    def post_run_command(self):
        return


class CMSRunCommandTask(CommandTask):

    def build_cmsrun_command(self, cfg_file, options=None):
        # highlighting helpers
        hl1 = lambda s: law.util.colored(s, color="cyan", style="bright")
        hl2 = lambda s: law.util.colored(s, color="cyan")

        # build the command and its representation
        cmd = ["cmsRun", cfg_file]
        cmd_repr = f"{hl1('cmsRun')} {hl2(cfg_file)}"
        for key, values in (options or {}).items():
            value = ",".join(map(str, law.util.make_list(values)))
            cmd.append(f"{key}={value}")
            cmd_repr += f" {hl1(key + '=')}{hl2(value)}"

        return cmd, cmd_repr


class PlotTask(BaseTask):

    file_types = law.CSVParameter(
        default=("pdf",),
        description="comma-separated types of the output plot files; default: pdf",
    )
    plot_postfix = luigi.Parameter(
        default=law.NO_STR,
        description="an arbitrary postfix that is added with two underscores to all paths of produced plots; "
        "default: empty",
    )
    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots right in the terminal; "
        "default: empty",
    )

    def create_plot_names(self, parts):
        plot_file_types = ["pdf", "png", "root", "c", "eps"]
        if any(t not in plot_file_types for t in self.file_types):
            raise Exception(
                f"plot names only allowed for file types {','.join(plot_file_types)}, got {','.join(self.file_types)}",
            )

        if self.plot_postfix and self.plot_postfix != law.NO_STR:
            parts.append((self.plot_postfix,))

        name = "__".join(map(str, parts))
        return [f"{name}.{t}" for t in self.file_types]


@law.decorator.factory(accept_generator=True)
def view_output_plots(fn, opts, task, *args, **kwargs):
    def before_call():
        return None

    def call(state):
        return fn(task, *args, **kwargs)

    def after_call(state):
        view_cmd = getattr(task, "view_cmd", None)
        if not view_cmd or view_cmd == law.NO_STR:
            return

        # prepare the view command
        if "{}" not in view_cmd:
            view_cmd += " {}"

        # collect all paths to view
        view_targets = OrderedDict()
        outputs = law.util.flatten(task.output())
        while outputs:
            output = outputs.pop(0)
            if isinstance(output, law.TargetCollection):
                outputs.extend(output._flat_target_list)
                continue
            if not getattr(output, "path", None):
                continue
            if output.path.endswith((".pdf", ".png")) and output.uri() not in view_targets:
                view_targets[output.uri()] = output

        # loop through targets and view them
        for target in view_targets.values():
            task.publish_message(f"showing {target.path}")
            with target.localize("r") as tmp:
                law.util.interruptable_popen(
                    view_cmd.format(tmp.path),
                    shell=True,
                    executable="/bin/bash",
                )

    return before_call, call, after_call
