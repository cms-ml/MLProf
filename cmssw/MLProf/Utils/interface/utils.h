#ifndef MLPROF_UTILS_H
#define MLPROF_UTILS_H

/*
 * Helper functions.
 */

#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <random>

namespace mlprof {

  enum InputType {
    Incremental,
    Random,
    Zeros,
    Ones,
  };

  void writeRuntimes(const std::string& path, float batchSize, std::vector<float> runtimes) {
    std::ofstream file(path, std::ios::out | std::ios::app);
    for (int i = 0; i < (int)runtimes.size(); i++) {
      file << batchSize << "," << runtimes[i] << std::endl;
    }
    file.close();
  }

}  // namespace mlprof

#endif  // MLPROF_UTILS_H
