# coding: utf-8

"""
Collection of test tasks.
"""

import os

import luigi
import law

from mlprof.tasks.base import BaseTask


class TestTask(BaseTask):

    i = luigi.IntParameter(default=1)

    def output(self):
        return self.local_target(f"the_test_file_{self.i}.json")

    def run(self):
        j = self.i * 3
        self.output().dump({"j": j}, indent=4, formatter="json")


class CMSSWTestTask(BaseTask, law.tasks.RunOnceTask):

    sandbox = "bash::$MLP_BASE/sandboxes/cmssw_default.sh"

    def requires(self):
        return TestTask.req(self, i=99)

    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        # print the content of the input
        print(self.input().load(formatter="json"))

        # print the cmssw version
        print(os.getenv("CMSSW_VERSION"))
