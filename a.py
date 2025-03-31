import numpy as np
from scipy.stats import truncnorm

# Constants
WORK_UNITS_PER_HOUR = 10
WORK_UNIT_TIME = 60 / WORK_UNITS_PER_HOUR  # minutes per work-unit
OPERATING_TIME = 8 * 60  # 480 minutes
NUM_CUSTOMERS = 160

# Generate shared customer data
def generate_customers(seed=42):
    np.random.seed(seed)
    arrival_times = np.sort(np.random.uniform(0, OPERATING_TIME, NUM_CUSTOMERS))
    work_units = truncnorm((5 - 5)/0.5, (15 - 5)/0.5, loc=5, scale=0.5).rvs(NUM_CUSTOMERS)
    return [{'arrival_time': a, 'work_units': w} for a, w in zip(arrival_times, work_units)]

# Simulation without priority queue
def simulate_bank(customers, num_windows):
    window_free_at = [0] * num_windows
    results = []

    for cust in customers:
        arrival = cust['arrival_time']
        work = cust['work_units']
        service_duration = work * WORK_UNIT_TIME

        best_window = 0
        earliest_start_time = max(arrival, window_free_at[0])

        for w in range(1, num_windows):
            possible_start = max(arrival, window_free_at[w])
            if possible_start < earliest_start_time:
                earliest_start_time = possible_start
                best_window = w

        start_time = earliest_start_time
        end_time = start_time + service_duration

        if end_time <= OPERATING_TIME:
            window_free_at[best_window] = end_time
            wait_time = start_time - arrival
            served = True
        else:
            wait_time = None
            served = False

        results.append({'wait_time': wait_time, 'served': served})

    wait_times = [r['wait_time'] for r in results if r['served']]
    num_not_served = sum(1 for r in results if not r['served'])
    avg_wait = sum(wait_times) / len(wait_times)

    print(f"With {num_windows} windows (no priority queue):")
    print(f"- Average wait time: {avg_wait:.2f} minutes")
    print(f"- Number of people not served: {num_not_served}")
    print()

# Simulation with a priority queue for light work
def simulate_with_priority_queue(customers, num_windows=10, light_window_share=0.3, threshold=7):
    # Separate customers
    light_customers = [c for c in customers if c['work_units'] <= threshold]
    regular_customers = [c for c in customers if c['work_units'] > threshold]

    # Assign window count
    light_windows = int(num_windows * light_window_share)
    regular_windows = num_windows - light_windows

    def run_queue(queue_customers, num_windows):
        window_free_at = [0] * num_windows
        results = []
        for cust in queue_customers:
            arrival = cust['arrival_time']
            work = cust['work_units']
            service_duration = work * WORK_UNIT_TIME

            best_window = 0
            earliest_start = max(arrival, window_free_at[0])

            for w in range(1, num_windows):
                possible_start = max(arrival, window_free_at[w])
                if possible_start < earliest_start:
                    earliest_start = possible_start
                    best_window = w

            end_time = earliest_start + service_duration

            if end_time <= OPERATING_TIME:
                window_free_at[best_window] = end_time
                wait_time = earliest_start - arrival
                results.append({'wait_time': wait_time, 'served': True})
            else:
                results.append({'wait_time': None, 'served': False})
        return results

    # Run both queues
    light_results = run_queue(light_customers, light_windows)
    regular_results = run_queue(regular_customers, regular_windows)

    all_results = light_results + regular_results
    wait_times = [r['wait_time'] for r in all_results if r['served']]
    not_served = sum(1 for r in all_results if not r['served'])
    avg_wait = sum(wait_times) / len(wait_times)

    print(f"Priority Queue ({light_windows} light, {regular_windows} regular windows):, threshold = {threshold} (light WU)")
    print(f"- Average wait time: {avg_wait:.2f} minutes")
    print(f"- Number of people not served: {not_served}")
    print()

# Generate shared customer list
customers = generate_customers(seed=42) # use seed to keep the data consistent for debugging

#percentage of customers per work-unit range
def print_work_unit_percentages(customers, bin_size=0.2, min_wu=5.0, max_wu=6.5):
    work_units = np.array([c['work_units'] for c in customers])
    bins = np.arange(min_wu, max_wu + bin_size, bin_size)
    hist, bin_edges = np.histogram(work_units, bins=bins)

    total = len(work_units)
    print("Work Unit Range\t|\t% of Customers")
    print("-" * 40)
    for count, low, high in zip(hist, bin_edges[:-1], bin_edges[1:]):
        percentage = (count / total) * 100
        print(f"{low:.1f} – {high:.1f}\t|\t{percentage:5.2f}%")

print_work_unit_percentages(customers)

# Run standard simulation
simulate_bank(customers, 9)
simulate_bank(customers, 10)
simulate_bank(customers, 11)

# Run priority queue version
#simulate_with_priority_queue(customers, num_windows=10, light_window_share=0.3, threshold=6)

simulate_with_priority_queue(customers, num_windows=10, light_window_share=0.1, threshold=5.4)

#customer work per unit demand graph

import matplotlib.pyplot as plt

def plot_work_unit_distribution(customers, bins=160):
    work_units = [c['work_units'] for c in customers]

    plt.figure(figsize=(8, 5))
    plt.hist(work_units, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Distribution of Customer Work-Unit Demand")
    plt.xlabel("Work Units")
    plt.ylabel("Number of Customers")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    avg = sum(work_units) / len(work_units)
    plt.axvline(avg, color='red', linestyle='dashed', linewidth=1.5, label=f'Average = {avg:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_work_unit_distribution(customers,20)

# each teller handle 11 work unit per hour

def simulate_variable_teller_speeds(customers, teller_speeds):
    num_tellers = len(teller_speeds)
    window_free_at = [0] * num_tellers
    results = []

    for cust in customers:
        arrival = cust['arrival_time']
        work = cust['work_units']

        # Choose the best teller (earliest possible finish time)
        best_window = 0
        best_start = max(arrival, window_free_at[0])
        best_end = best_start + (work * 60 / teller_speeds[0])

        for w in range(1, num_tellers):
            start = max(arrival, window_free_at[w])
            end = start + (work * 60 / teller_speeds[w])
            if end < best_end:
                best_window = w
                best_start = start
                best_end = end

        if best_end <= OPERATING_TIME:
            window_free_at[best_window] = best_end
            wait_time = best_start - arrival
            served = True
        else:
            wait_time = None
            served = False

        results.append({'wait_time': wait_time, 'served': served})

    wait_times = [r['wait_time'] for r in results if r['served']]
    not_served = sum(1 for r in results if not r['served'])
    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

    print(f"Teller Speeds: {teller_speeds}")
    print(f"- Average wait time: {avg_wait:.2f} minutes")
    print(f"- Number of people not served: {not_served}")
    print()

    
simulate_variable_teller_speeds(customers, teller_speeds=[11]*10)




