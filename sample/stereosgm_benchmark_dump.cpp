#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libsgm.h>

#include "sample_common.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal flat-JSON parser for params.json
// ---------------------------------------------------------------------------
struct DumpParams
{
    int width;
    int height;
    int disparity_size;
    int src_depth;
    int output_depth;
    int P1;
    int P2;
    float uniqueness;
    int subpixel;
    int num_paths;
    int min_disp;
    int lr_max_diff;
    int census_type;
};

static std::string trim(const std::string &s)
{
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static DumpParams parse_params_json(const fs::path &path)
{
    std::ifstream ifs(path);
    ASSERT_MSG(ifs.is_open(), "Cannot open " + path.string());

    DumpParams p{};
    std::string line;
    while (std::getline(ifs, line))
    {
        // Each line is like:   "key": value,
        auto colon = line.find(':');
        if (colon == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));

        // remove surrounding quotes from key
        if (key.size() >= 2 && key.front() == '"' && key.back() == '"')
            key = key.substr(1, key.size() - 2);

        // remove trailing comma from value
        if (!val.empty() && val.back() == ',')
            val.pop_back();

        if (key == "width")
            p.width = std::stoi(val);
        else if (key == "height")
            p.height = std::stoi(val);
        else if (key == "disparity_size")
            p.disparity_size = std::stoi(val);
        else if (key == "src_depth")
            p.src_depth = std::stoi(val);
        else if (key == "output_depth")
            p.output_depth = std::stoi(val);
        else if (key == "P1")
            p.P1 = std::stoi(val);
        else if (key == "P2")
            p.P2 = std::stoi(val);
        else if (key == "uniqueness")
            p.uniqueness = std::stof(val);
        else if (key == "subpixel")
            p.subpixel = std::stoi(val);
        else if (key == "num_paths")
            p.num_paths = std::stoi(val);
        else if (key == "min_disp")
            p.min_disp = std::stoi(val);
        else if (key == "lr_max_diff")
            p.lr_max_diff = std::stoi(val);
        else if (key == "census_type")
            p.census_type = std::stoi(val);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct DumpEntry
{
    std::string name; // e.g. "sgm_dump_000"
    fs::path dir;     // full path to the dump folder
    DumpParams params;
    cv::Mat left;
    cv::Mat right;
};

// ---------------------------------------------------------------------------
// Discover and sort all sgm_dump_* directories
// ---------------------------------------------------------------------------
static std::vector<DumpEntry> discover_dumps(const fs::path &input_dir)
{
    std::vector<DumpEntry> entries;
    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (!entry.is_directory())
            continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("sgm_dump_", 0) != 0)
            continue;

        DumpEntry d;
        d.name = name;
        d.dir = entry.path();
        entries.push_back(std::move(d));
    }

    std::sort(entries.begin(), entries.end(), [](const DumpEntry &a, const DumpEntry &b) { return a.name < b.name; });

    return entries;
}

// ---------------------------------------------------------------------------
// Load params + images for a single dump entry
// ---------------------------------------------------------------------------
static void load_dump(DumpEntry &d)
{
    d.params = parse_params_json(d.dir / "params.json");
    d.left = cv::imread((d.dir / "left.png").string(), cv::IMREAD_UNCHANGED);
    d.right = cv::imread((d.dir / "right.png").string(), cv::IMREAD_UNCHANGED);
    ASSERT_MSG(!d.left.empty(), "Failed to load " + (d.dir / "left.png").string());
    ASSERT_MSG(!d.right.empty(), "Failed to load " + (d.dir / "right.png").string());
    ASSERT_MSG(d.left.size() == d.right.size() && d.left.type() == d.right.type(), d.name + ": left/right images must have same size and type");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static const std::string keys = "{ @input_dir      | <none> | path to folder containing sgm_dump_* directories         }"
                                "{ csv              |        | optional path to write CSV results                       }"
                                "{ save_disparity   |        | optional output folder for colorized disparity images    }"
                                "{ help h           |        | display this help and exit                               }";

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const fs::path input_dir = parser.get<cv::String>("@input_dir");
    const std::string csv_path = parser.get<cv::String>("csv");
    const std::string save_disp_dir = parser.get<cv::String>("save_disparity");

    if (!parser.check())
    {
        parser.printErrors();
        parser.printMessage();
        std::exit(EXIT_FAILURE);
    }

    ASSERT_MSG(fs::is_directory(input_dir), "Input directory does not exist: " + input_dir.string());

    // create output folder for disparity images if requested
    if (!save_disp_dir.empty())
    {
        fs::create_directories(save_disp_dir);
    }

    // -----------------------------------------------------------------------
    // 1. Discover all dump directories
    // -----------------------------------------------------------------------
    std::cout << "Discovering dump directories in " << input_dir << " ..." << std::endl;
    std::vector<DumpEntry> dumps = discover_dumps(input_dir);
    ASSERT_MSG(!dumps.empty(), "No sgm_dump_* directories found in " + input_dir.string());
    std::cout << "Found " << dumps.size() << " dump directories." << std::endl;

    // -----------------------------------------------------------------------
    // 2. Load all data (images + params) upfront
    // -----------------------------------------------------------------------
    std::cout << "Loading all images and parameters..." << std::endl;
    for (auto &d : dumps)
    {
        load_dump(d);
    }
    std::cout << "All data loaded." << std::endl;

    // -----------------------------------------------------------------------
    // 3. CUDA warm-up: run the first image once, untimed
    // -----------------------------------------------------------------------
    {
        const auto &d = dumps.front();
        const auto &p = d.params;
        const auto path_type = p.num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
        const auto census = static_cast<sgm::CensusType>(p.census_type);

        sgm::LibSGMWrapper warmup(p.disparity_size, p.P1, p.P2, p.uniqueness, p.subpixel != 0, path_type, p.min_disp, p.lr_max_diff, census);
        cv::Mat disp;
        warmup.execute(d.left, d.right, disp);
        cudaDeviceSynchronize();
        std::cout << "CUDA warm-up complete." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4. Benchmark loop
    // -----------------------------------------------------------------------
    cudaDeviceProp prop;
    int version;
    cudaGetDeviceProperties(&prop, 0);
    cudaRuntimeGetVersion(&version);

    std::cout << std::endl;
    std::cout << "# Settings" << std::endl;
    std::cout << "Device        : " << prop.name << std::endl;
    std::cout << "CUDA runtime  : " << version << std::endl;
    std::cout << "Dump count    : " << dumps.size() << std::endl;
    std::cout << std::endl;

    struct Result
    {
        std::string name;
        int width;
        int height;
        int disparity_size;
        int min_disp;
        double time_ms;
    };
    std::vector<Result> results;
    results.reserve(dumps.size());

    std::cout << "Running benchmark..." << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto &d : dumps)
    {
        const auto &p = d.params;
        const auto path_type = p.num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
        const auto census = static_cast<sgm::CensusType>(p.census_type);
        cv::Mat disparity;

        // Create a fresh SGM instance for each image

        // --- timed section ---
        const auto t1 = std::chrono::high_resolution_clock::now();
        sgm::LibSGMWrapper sgm(p.disparity_size, p.P1, p.P2, p.uniqueness, p.subpixel != 0, path_type, p.min_disp, p.lr_max_diff, census);
        sgm.execute(d.left, d.right, disparity);
        cudaDeviceSynchronize();
        const auto t2 = std::chrono::high_resolution_clock::now();
        // --- end timed section ---

        const double elapsed_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        results.push_back({d.name, p.width, p.height, p.disparity_size, p.min_disp, elapsed_ms});

        std::cout << std::setw(16) << std::left << d.name << std::setw(5) << std::right << p.width << "x" << std::setw(5) << std::left << p.height << "  disp=" << std::setw(4)
                  << std::left << p.disparity_size << std::fixed << std::setprecision(2) << std::setw(9) << std::right << elapsed_ms << " ms" << std::endl;

        // optionally save colorized disparity
        if (!save_disp_dir.empty())
        {
            const int disp_scale = (p.subpixel != 0) ? sgm::StereoSGM::SUBPIXEL_SCALE : 1;
            cv::Mat colorized;
            colorize_disparity(disparity, colorized, disp_scale * p.disparity_size, disparity == sgm.getInvalidDisparity());
            const fs::path out_path = fs::path(save_disp_dir) / (d.name + "_disparity.png");
            cv::imwrite(out_path.string(), colorized);
        }
    }
    std::cout << std::string(70, '-') << std::endl;

    // -----------------------------------------------------------------------
    // 5. Aggregate statistics
    // -----------------------------------------------------------------------
    std::vector<double> times;
    times.reserve(results.size());
    for (const auto &r : results)
        times.push_back(r.time_ms);

    const double total_ms = std::accumulate(times.begin(), times.end(), 0.0);
    const double avg_ms = total_ms / static_cast<double>(times.size());
    const double min_ms = *std::min_element(times.begin(), times.end());
    const double max_ms = *std::max_element(times.begin(), times.end());

    std::sort(times.begin(), times.end());
    const double median_ms = (times.size() % 2 == 0) ? (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0 : times[times.size() / 2];

    std::cout << std::endl;
    std::cout << "# Results (" << results.size() << " images)" << std::endl;
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(2) << "Total time : " << total_ms << " ms" << std::endl
              << "Average    : " << avg_ms << " ms" << std::endl
              << "Median     : " << median_ms << " ms" << std::endl
              << "Min        : " << min_ms << " ms" << std::endl
              << "Max        : " << max_ms << " ms" << std::endl
              << "Throughput : " << std::setprecision(1) << (1e3 * results.size() / total_ms) << " images/sec" << std::endl;

    // per-disparity-size breakdown
    std::map<int, std::vector<double>> times_by_disp;
    for (const auto &r : results)
        times_by_disp[r.disparity_size].push_back(r.time_ms);

    std::cout << std::endl;
    std::cout << "# Average by disparity size" << std::endl;
    for (const auto &[disp_size, dtimes] : times_by_disp)
    {
        const double sum = std::accumulate(dtimes.begin(), dtimes.end(), 0.0);
        const double avg = sum / static_cast<double>(dtimes.size());
        const double dmin = *std::min_element(dtimes.begin(), dtimes.end());
        const double dmax = *std::max_element(dtimes.begin(), dtimes.end());
        std::cout << std::setprecision(2) << "  disp=" << std::setw(4) << std::left << disp_size << " :  n=" << std::setw(4) << std::right << dtimes.size()
                  << "  avg=" << std::setw(8) << std::right << avg << " ms"
                  << "  min=" << std::setw(8) << std::right << dmin << " ms"
                  << "  max=" << std::setw(8) << std::right << dmax << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 6. CSV output
    // -----------------------------------------------------------------------
    if (!csv_path.empty())
    {
        std::ofstream csv(csv_path);
        ASSERT_MSG(csv.is_open(), "Cannot open CSV file for writing: " + csv_path);
        csv << "dump_name,width,height,disparity_size,min_disp,time_ms\n";
        for (const auto &r : results)
        {
            csv << r.name << "," << r.width << "," << r.height << "," << r.disparity_size << "," << r.min_disp << "," << std::fixed << std::setprecision(3) << r.time_ms << "\n";
        }
        std::cout << std::endl << "CSV results written to: " << csv_path << std::endl;
    }

    return 0;
}
