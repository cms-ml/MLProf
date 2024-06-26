/*
 * Plugin to measure the inference runtime of an onnx model.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "MLProf/Utils/interface/utils.h"

using namespace cms::Ort;

class ONNXInference : public edm::stream::EDAnalyzer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit ONNXInference(const edm::ParameterSet&, const ONNXRuntime*);
  ~ONNXInference(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  void beginJob(){};
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob(){};

  inline float drawNormal() { return normalPdf_(rndGen_); }

  // parameters
  std::vector<std::string> inputTensorNames_;
  std::vector<std::string> outputTensorNames_;
  std::string outputFile_;
  std::string inputTypeStr_;
  std::vector<int> inputRanks_;
  std::vector<int> flatInputSizes_;
  int batchSize_;
  int nCalls_;

  // other members
  int nInputs_;
  int nPreCalls_;
  mlprof::InputType inputType_;
  std::random_device rnd_;
  std::default_random_engine rndGen_;
  std::normal_distribution<float> normalPdf_;

  std::vector<std::vector<int64_t>> input_shapes_;
  FloatArrays inputArrays_;  // each stream hosts its own data
};

void ONNXInference::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when
  // compiling
  edm::ParameterSetDescription desc;
  // the path to the file containing the graph
  desc.add<std::string>("graphPath");
  // the names of the input tensors
  desc.add<std::vector<std::string>>("inputTensorNames");
  // the names of the output tensors
  desc.add<std::vector<std::string>>("outputTensorNames");
  // the name of the output csv file
  desc.add<std::string>("outputFile");
  // the type of input values, either "incremental" or "random"
  desc.add<std::string>("inputType", "random");
  // the rank (number of dimensions) of each input tensor
  desc.add<std::vector<int>>("inputRanks");
  // flat list of sizes of each dimension of each input tensor
  // (for a graph with a 1D and a 2D input tensor, this would be a vector of three values)
  desc.add<std::vector<int>>("flatInputSizes");
  // batch size to test
  desc.add<int>("batchSize");
  // the number of calls to the graph to measure the runtime
  desc.add<int>("nCalls");

  descriptions.addWithDefaultLabel(desc);
}

ONNXInference::ONNXInference(const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : inputTensorNames_(iConfig.getParameter<std::vector<std::string>>("inputTensorNames")),
      outputTensorNames_(iConfig.getParameter<std::vector<std::string>>("outputTensorNames")),
      outputFile_(iConfig.getParameter<std::string>("outputFile")),
      inputTypeStr_(iConfig.getParameter<std::string>("inputType")),
      inputRanks_(iConfig.getParameter<std::vector<int>>("inputRanks")),
      flatInputSizes_(iConfig.getParameter<std::vector<int>>("flatInputSizes")),
      batchSize_(iConfig.getParameter<int>("batchSize")),
      nCalls_(iConfig.getParameter<int>("nCalls")),
      nInputs_(inputTensorNames_.size()),
      nPreCalls_(10),
      rndGen_(rnd_()),
      normalPdf_(0.0, 1.0) {
  // the number of input ranks must match the number of input tensors
  if ((int)inputRanks_.size() != nInputs_) {
    throw cms::Exception("InvalidInputRanks") << "number of input ranks must match number of input tensors";
  }
  // the input must be at least 1 dimensional
  for (auto rank : inputRanks_) {
    if (rank < 1) {
      throw cms::Exception("InvalidRank") << "only ranks above 0 are supported, got " << rank;
    }
  }
  // the sum of ranks must match the number of flat input sizes
  if (std::accumulate(inputRanks_.begin(), inputRanks_.end(), 0) != (int)flatInputSizes_.size()) {
    throw cms::Exception("InvalidFlatInputSizes")
        << "sum of input ranks must match number of flat input sizes, got " << flatInputSizes_.size();
  }
  // batch size must be positive
  if (batchSize_ < 1) {
    throw cms::Exception("InvalidBatchSize") << "batch size must be positive, got " << batchSize_;
  }

  // input sizes must be positive
  for (auto size : flatInputSizes_) {
    if (size < 1) {
      throw cms::Exception("InvalidInputSize") << "input sizes must be positive, got " << size;
    }
  }
  // check the input type
  if (inputTypeStr_ == "incremental") {
    inputType_ = mlprof::InputType::Incremental;
  } else if (inputTypeStr_ == "random") {
    inputType_ = mlprof::InputType::Random;
  } else if (inputTypeStr_ == "zeros") {
    inputType_ = mlprof::InputType::Zeros;
  } else if (inputTypeStr_ == "ones") {
    inputType_ = mlprof::InputType::Ones;
  } else {
    throw cms::Exception("InvalidInputType") << "input type unknown: " << inputTypeStr_;
  }

  // initialize the input_shapes array with inputRanks_ and flatInputSizes_
  int i = 0;
  for (auto rank : inputRanks_) {
    std::vector<int64_t> input_shape(flatInputSizes_.begin() + i, flatInputSizes_.begin() + i + rank);
    input_shape.insert(input_shape.begin(), batchSize_);
    input_shapes_.push_back(input_shape);
    i += rank;
  }
  // initialize the input data arrays
  // note there is only one element in the FloatArrays type (i.e. vector<vector<float>>) variable
  for (int i = 0; i < nInputs_; i++) {
    // multiply the size of all dimensions in an input
    int full_size_input = std::accumulate(begin(input_shapes_[i]), end(input_shapes_[i]), 1, std::multiplies<int>());
    // initialize inputArrays_ with 0s at first
    inputArrays_.emplace_back(full_size_input, 0);
  }
}

std::unique_ptr<ONNXRuntime> ONNXInference::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(edm::FileInPath(iConfig.getParameter<std::string>("graphPath")).fullPath());
}

void ONNXInference::globalEndJob(const ONNXRuntime* cache) {}

void ONNXInference::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  for (int i = 0; i < nInputs_; i++) {
    std::vector<float>& group_data = inputArrays_[i];
    // fill the input
    for (int i = 0; i < (int)group_data.size(); i++) {
      group_data[i] = inputType_ == mlprof::InputType::Incremental
                          ? float(i)
                          : float(inputType_ == mlprof::InputType::Zeros ? 0 : drawNormal());
    }
  }

  // run prediction and get outputs
  std::vector<std::vector<float>> outputs;

  // pre calls to "warm up"
  for (int r = 0; r < nPreCalls_; r++) {
    outputs = globalCache()->run(inputTensorNames_, inputArrays_, input_shapes_, outputTensorNames_, batchSize_);
  }

  // actual calls to measure runtimes
  std::vector<float> runtimes;
  for (int r = 0; r < nCalls_; r++) {
    auto start = std::chrono::high_resolution_clock::now();

    // inference
    outputs = globalCache()->run(inputTensorNames_, inputArrays_, input_shapes_, outputTensorNames_, batchSize_);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> runtime_in_seconds = (end - start);
    runtimes.push_back(runtime_in_seconds.count() * 1000);
  }

  // save them
  mlprof::writeRuntimes(outputFile_, batchSize_, runtimes);
}

DEFINE_FWK_MODULE(ONNXInference);
