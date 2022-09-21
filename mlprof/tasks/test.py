# coding: utf-8

import luigi

from mlprof.tasks.base import BaseTask


class TestTask(BaseTask):

    i = luigi.IntParameter(default=1)

    def output(self):
        return self.local_target(f"the_test_file_{self.i}.json")

    def run(self):
        j = self.i * 3
        self.output().dump({"j": j}, indent=4, formatter="json")
