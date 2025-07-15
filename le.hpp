
#include <aiCompute.h>
static double d = 0;//length
static std::vector<std::vector<double>> tsp=sampleTSP();
static std::vector<std::vector<std::vector<double>>> tsps(4, tsp);
static aiCompute e=aiCompute();
double upd(const std::vector<std::vector<std::vector<double>>> &v3d) {
    for (auto dist:v3d) {
        d+=length(dist);
    }
}
double length(const std::vector<std::vector<double>> &dist){
    std::vector<int> aco_path = e.antColonyOptimizeTSP(dist, 20, 100);
    for (size_t i = 0; i < dist.size(); ++i) {
        int from = aco_path[i];
        int to = aco_path[(i + 1) % aco_path.size()];
        sum += dist[from][to];
    }
    return sum;
}
static auto arr = e.linspace(0, 1, 11);
static double mean = e.mean(arr);
static double dev = e.stddev(arr);
