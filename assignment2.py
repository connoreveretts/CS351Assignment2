# assignment2.py
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# ThreeSum analysis
class ThreeSumAnalyzer:
    # Starting point for operation counting
    def __init__(self):
        self.operation_count = 0
    
    # ThreeSum implementation
    def threesum_bruteforce(self, arr, target):
        self.operation_count = 0
        
        # Check if array is valid wth at least 3 elements
        if arr is None or len(arr) < 3:
            return False, None, 0
        
        n = len(arr)
        
        # First loop is to pick the first element
        for i in range(n - 2):
            # Second loop is to pick the second element
            for j in range(i + 1, n - 1):
                # Third loop is to pick the third element
                for k in range(j + 1, n):
                    self.operation_count += 1
                    # Check if the sum matches the target
                    if arr[i] + arr[j] + arr[k] == target:
                        # If then return True and the triplet
                        return True, (arr[i], arr[j], arr[k]), self.operation_count
        # If no triplet found, return False
        return False, None, self.operation_count
    
    # Extra Credit: ThreeSum with early termination
    def threesum_early_termination(self, arr, target):
        self.operation_count = 0
        
        # Check if array is valid
        if arr is None or len(arr) < 3:
            return False, None, 0
        sorted_arr = sorted(arr)
        n = len(sorted_arr)
        
        # First loop is to pick the first element 
        for i in range(n - 2):
            # Early termination if the smallest possible sum is larger then target number
            if sorted_arr[i] + sorted_arr[i+1] + sorted_arr[i+2] > target:
                break
            # Second loop is to pick the second element
            for j in range(i + 1, n - 1):
                # Early termination if the smallest possible sum is larger then target number as above
                if sorted_arr[i] + sorted_arr[j] + sorted_arr[j+1] > target:
                    break
                # Third loop is to pick the third element
                for k in range(j + 1, n):
                    # Count each operation and check if the sum matches the target
                    self.operation_count += 1
                    current_sum = sorted_arr[i] + sorted_arr[j] + sorted_arr[k]
                    # If then return True and the triplet
                    if current_sum == target:
                        return True, (sorted_arr[i], sorted_arr[j], sorted_arr[k]), self.operation_count
                    # If else then break to next iteration of j loop
                    elif current_sum > target:
                        break
        # If no triplet found, return False
        return False, None, self.operation_count
    
    # Generate the random test data
    def generate_test_data(self, size, value_range=1000):
        return [random.randint(-value_range, value_range) for _ in range(size)]
    
    # Find the maximum value in an array
    def run_performance_test(self, sizes, num_trials=10):
        results = {
            'sizes': [],
            'average_runtime': [],
            'average_operations': [],
            'runtime_ratios': [],
            'operation_ratios': []
        }
        prev_runtime = None
        prev_operations = None
        
        # Run tests for each size in size list 
        for size in sizes:
            runtimes = []
            operations = []
            # Run multiple trials for averaging
            for trial in range(num_trials):
                test_array = self.generate_test_data(size)
                target = 999999
                start_time = time.perf_counter()
                found, triplet, ops = self.threesum_bruteforce(test_array, target)
                end_time = time.perf_counter()
                runtime_ms = (end_time - start_time) * 1000
                runtimes.append(runtime_ms)
                operations.append(ops)
            # Calculate averages
            average_runtime = sum(runtimes) / len(runtimes)
            average_ops = sum(operations) / len(operations)
            runtime_ratio = average_runtime / prev_runtime if prev_runtime else 1.0
            ops_ratio = average_ops / prev_operations if prev_operations else 1.0
            # Store results
            results['sizes'].append(size)
            results['average_runtime'].append(average_runtime)
            results['average_operations'].append(average_ops)
            results['runtime_ratios'].append(runtime_ratio)
            results['operation_ratios'].append(ops_ratio)
            prev_runtime = average_runtime
            prev_operations = average_ops
        return results
    
    # Run the next 5 test cases
    def run_test_cases(self):
        test_cases = [
            ([3, 7, 1, 2, 8, 4, 5], 13, True),
            ([1, 2, 3, 4], 15, False),
            ([1, 2, 3], 6, True),
            ([-1, -2, -3, 4, 5, 6], 7, True),
            ([1, 2], 3, False)]
        results = []
        
        # Run each test case
        for arr, target, expected in test_cases:
            found, triplet, operations = self.threesum_bruteforce(arr, target)
            results.append({
                'array': arr,
                'target': target,
                'expected': expected,
                'actual': found,
                'triplet': triplet,
                'operations': operations,
                'passed': found == expected
            })
        return results

# Set up plotting functions
def create_performance_graphs(results):
    sizes = results['sizes']
    runtimes = results['average_runtime']
    operations = results['average_operations']
    
    # The O(n^3) curve
    theoretical_ops = [n * (n-1) * (n-2) / 6 for n in sizes]
    
    # Runtime graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, runtimes, 'bo-', label='Measured Runtime', linewidth=2)
    plt.xlabel('Array Size (n)')
    plt.ylabel('Average Runtime (ms)')
    plt.title('Three Sum Runtime Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Operations graph
    plt.subplot(1, 2, 2)
    plt.plot(sizes, operations, 'ro-', label='Operations', linewidth=2)
    plt.plot(sizes, theoretical_ops, 'g--', label='Theoretical O(n³)', linewidth=2)
    plt.xlabel('Array Size (n)')
    plt.ylabel('Operations')
    plt.title('Measured vs Theoretical')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Layouts
    plt.tight_layout()
    plt.savefig('threesum_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    return theoretical_ops

# Extra Credit: Comparing standard vs the optimized algorithm
# Compare the two algorithms
def compare_optimization(analyzer, test_size=200):
    test_array = analyzer.generate_test_data(test_size)
    target = 50
    
    # Standard algorithm
    start = time.perf_counter()
    found1, triplet1, ops1 = analyzer.threesum_bruteforce(test_array, target)
    time1 = time.perf_counter() - start
    
    # Optimized algorithm
    start = time.perf_counter()
    found2, triplet2, ops2 = analyzer.threesum_early_termination(test_array, target)
    time2 = time.perf_counter() - start
    
    # Calculate improvement
    improvement = (ops1 - ops2) / ops1 * 100 if ops1 > 0 else 0
    
    # Return results
    return {
        'standard_ops': ops1,
        'standard_time': time1 * 1000,
        'optimized_ops': ops2,
        'optimized_time': time2 * 1000,
        'improvement_percent': improvement
    }

# Main execution functions
def main():
    analyzer = ThreeSumAnalyzer()
    
    # Run test cases
    print("---------Test Cases----------")
    test_results = analyzer.run_test_cases()
    # For each test case, print the results
    for i, result in enumerate(test_results, 1):
        print(f"Test {i}: Expected={result['expected']}, Actual={result['actual']}, "
              f"Operations={result['operations']}, Passed={result['passed']}")
        if result['triplet']:
            print(f"  Triplet found: {result['triplet']}")
    
    # Run performance tests
    print("---------Performance Tests----------")
    test_sizes = [50, 100, 200, 400, 800]
    perf_results = analyzer.run_performance_test(test_sizes, num_trials=5)
    print(f"{'Size':<8} {'Runtime(ms)':<12} {'Operations':<12} {'Ratio':<8}")
    print("-" * 45)
    # Print performance results
    for i, size in enumerate(perf_results['sizes']):
        runtime = perf_results['average_runtime'][i]
        ops = perf_results['average_operations'][i]
        ratio = perf_results['runtime_ratios'][i]
        print(f"{size:<8} {runtime:<12.2f} {ops:<12.0f} {ratio:<8.2f}")
    
    # Create graphs
    theoretical_ops = create_performance_graphs(perf_results)
    
    # Extra credit optimization comparison
    print("---------Optimization Comparison-----------")
    opt_results = compare_optimization(analyzer)
    print(f"Standard: {opt_results['standard_ops']} ops, {opt_results['standard_time']:.2f} ms")
    print(f"Optimized: {opt_results['optimized_ops']} ops, {opt_results['optimized_time']:.2f} ms")
    print(f"Improvement: {opt_results['improvement_percent']:.1f}%")
    return {
        'test_results': test_results,
        'performance_results': perf_results,
        'theoretical_operations': theoretical_ops,
        'optimization_results': opt_results
    }

# Run the main function
if __name__ == "__main__":
    results = main()