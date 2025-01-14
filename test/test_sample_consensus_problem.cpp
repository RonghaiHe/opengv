/*
 * @Author: RonghaiHe hrhkjys@qq.com
 * @Date: 2025-01-13 11:02:54
 * @LastEditors: RonghaiHe hrhkjys@qq.com
 * @LastEditTime: 2025-01-13 16:29:14
 * @FilePath: /opengv/test/test_sample_consensus_problem.cpp
 * @Description: 测试SampleConsensusProblem类的进程安全性
 */
#include <gtest/gtest.h>
#include <opengv/sac/SampleConsensusProblem.hpp>
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>

// Simple model type for testing
struct DummyModel {};

class SampleConsensusProblemTest : public ::testing::Test {
protected:
    class TestProblem : public opengv::sac::SampleConsensusProblem<DummyModel> {
    public:
        TestProblem(bool randomSeed = false) 
            : opengv::sac::SampleConsensusProblem<DummyModel>(randomSeed) {}

        // Minimal implementation of pure virtual functions
        int getSampleSize() const override { return 1; }
        bool computeModelCoefficients(
            const std::vector<int>& indices,
            DummyModel& model) const override { return true; }
        void optimizeModelCoefficients(
            const std::vector<int>& inliers,
            const DummyModel& model,
            DummyModel& optimized_model) override {}
        void getSelectedDistancesToModel(
            const DummyModel& model,
            const std::vector<int>& indices,
            std::vector<double>& scores) const override {}
    };

    static const int SEQUENCE_LENGTH = 10000;
    static const int NUM_THREADS = 8;
    
    void printSequenceStats(const std::vector<int>& sequence, int threadId) {
        if(sequence.empty()) return;
                
        std::cout << "\nThread " << threadId << " statistics:"
                  << "\n  Sequence length: " << sequence.size() << std::endl;
    }

    void printFirstNumbers(const std::vector<int>& sequence, int threadId, int count = 10) {
        std::cout << "\nThread " << threadId << " first " << count << " numbers:";
        for(int i = 0; i < std::min(count, static_cast<int>(sequence.size())); ++i) {
            if(i % 5 == 0) std::cout << "\n  ";
            std::cout << sequence[i] << " ";
        }
        std::cout << std::endl;
    }

    void compareSequences(const std::vector<int>& seq1, const std::vector<int>& seq2, int thread1, int thread2) {
        std::cout << "\nComparing first 10 numbers between thread " << thread1 << " and " << thread2 << ":\n";
        for(int i = 0; i < 10; ++i) {
            std::cout << seq1[i] << " vs " << seq2[i] << " : " 
                     << (seq1[i] == seq2[i] ? "SAME" : "DIFFERENT") << "\n";
        }
    }
};

TEST_F(SampleConsensusProblemTest, RandomGenerationTest) {
    const int ITERATIONS = 100000;
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> results(NUM_THREADS);
    
    std::cout << "\nStarting random number generation test with "
              << NUM_THREADS << " threads, "
              << ITERATIONS << " iterations per thread" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch threads
    for(int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([i, ITERATIONS, &results]() {
            TestProblem problem(false);
            std::vector<int>& sequence = results[i];
            sequence.reserve(ITERATIONS);
            
            for(int j = 0; j < ITERATIONS; ++j) {
                sequence.push_back(problem.rnd());
            }
        });
    }
    
    // Wait for completion
    for(auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== First 10 Numbers from Each Thread ===" << std::endl;
    for(int i = 0; i < NUM_THREADS; ++i) {
        printFirstNumbers(results[i], i);
    }
    
    std::cout << "\n=== Test Results ===" << std::endl;
    
    // Verify results and print correlation analysis
    std::cout << "\n=== Correlation Analysis ===" << std::endl;
    for(int i = 0; i < NUM_THREADS; ++i) {
        for(int j = i + 1; j < NUM_THREADS; ++j) {
            int matches = 0;
            for(int k = 0; k < ITERATIONS; ++k) {
                if(results[i][k] == results[j][k]) {
                    matches++;
                }
            }
            float match_ratio = static_cast<float>(matches) / ITERATIONS;
            std::cout << "Threads " << i << " and " << j 
                     << ": Matching numbers: " << matches 
                     << " (" << (match_ratio * 100) << "%)" << std::endl;
            
            EXPECT_LT(match_ratio, 0.001) 
                << "Suspicious similarity between threads " << i << " and " << j;
        }
    }
    
    // Performance metrics
    double numbers_per_second = (NUM_THREADS * ITERATIONS * 1000.0) / duration.count();
    std::cout << "\n=== Performance Metrics ==="
              << "\nTotal random numbers generated: " << (NUM_THREADS * ITERATIONS)
              << "\nTotal time: " << duration.count() << "ms"
              << "\nNumbers per second: " << numbers_per_second
              << "\nNumbers per second per thread: " << (numbers_per_second / NUM_THREADS)
              << std::endl;
}

// TEST_F(SampleConsensusProblemTest, MultiInstanceRandomTest) {
//     const int NUM_NUMBERS = 1000;  // Generate fewer numbers for clearer output
//     std::vector<std::unique_ptr<TestProblem>> problems;
//     std::vector<std::vector<int>> results(NUM_THREADS);
//     std::vector<std::thread> threads;
    
//     // Create problem instances (C++11 compatible)
//     for(int i = 0; i < NUM_THREADS; ++i) {
//         problems.push_back(std::unique_ptr<TestProblem>(new TestProblem(false)));
//     }
    
//     std::cout << "\nTesting multiple instances with " << NUM_THREADS 
//               << " threads, each generating " << NUM_NUMBERS << " numbers\n";
    
//     // Launch threads with separate instances
//     for(int i = 0; i < NUM_THREADS; ++i) {
//         threads.emplace_back([i, NUM_NUMBERS, &problems, &results]() {
//             std::vector<int>& sequence = results[i];
//             sequence.reserve(NUM_NUMBERS);
            
//             for(int j = 0; j < NUM_NUMBERS; ++j) {
//                 sequence.push_back(problems[i]->rnd());
//             }
//         });
//     }
    
//     // Wait for completion
//     for(auto& thread : threads) {
//         thread.join();
//     }
    
//     // Print first 10 numbers from each instance
//     std::cout << "\n=== First 10 Numbers from Each Instance ===\n";
//     for(int i = 0; i < NUM_THREADS; ++i) {
//         printFirstNumbers(results[i], i);
//     }
    
//     // Compare sequences between instances
//     std::cout << "\n=== Sequence Comparisons ===\n";
//     for(int i = 0; i < NUM_THREADS; ++i) {
//         for(int j = i + 1; j < NUM_THREADS; ++j) {
//             compareSequences(results[i], results[j], i, j);
//         }
//     }
    
//     // Verify all sequences are identical (since we used fixed seed)
//     for(int i = 1; i < NUM_THREADS; ++i) {
//         ASSERT_EQ(results[0], results[i]) 
//             << "Sequences from instance 0 and " << i << " differ!";
//     }
// }
