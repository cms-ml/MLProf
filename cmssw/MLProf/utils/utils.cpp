#include <list>
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>

//#include <vector>?

//Protected header?

std::tuple<float,float> mean_and_std(std::vector<float> data_vector)
{
    float runs = data_vector.size();

    // calculate mean runtime
    float mean_runtime = 0;
    for (float time : data_vector){mean_runtime += time;}

    // calculate std of mean runtime
    mean_runtime /= runs;
    float std_runtime = 0;

    for (float time : data_vector){std_runtime += pow(time - mean_runtime, 2);}
    std_runtime /= runs;
    std_runtime = pow(std_runtime, 0.5);

    // wrapp results in tuple
    auto results = std::make_tuple(mean_runtime, std_runtime);
    return results;
}

void deletePreviousFileContent(std::string csv_file)
{
  std::ofstream file(csv_file, std::ios::out | std::ios::trunc);
  file.close();
}

void writeFile(float batch_size, float mean, float std, std::string csv_file)
{
  std::ofstream file(csv_file, std::ios::out| std::ios::app);
  file << batch_size <<","<<mean<<","<<std <<std::endl;
  file.close();
}

void print_list(std::list<int> const &list)
{
    for (auto const &i: list) {
        std::cout << i << std::endl;
    }
}

void print_vector(std::vector<int> const &vector)
{
    for (auto const &i: vector) {
        std::cout << i << std::endl;
    }
}
