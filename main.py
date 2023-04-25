import random
from collections import deque
import numpy


def parse_data_file(filename):
    with open(filename) as f:
        k = f.readline().strip()
        lines = f.readlines()
        max_capacity = lines[-1].strip()
        lines = lines[:-1]
        items = []
        for line in lines[0:]:
            line = line.split()
            items.append((int(line[1]), int(line[2])))
    return int(max_capacity), items


def generate_solution(k):
    # Generating a list of 0s and 1s.
    solution = [random.randint(0, 1) for _ in range(k)]
    return solution


def determine_quality(solution):
    # Determining the quality of a solution by iterating through the solution and value lists, multiplying the
    # elements with corresponding indexes and adding the result to quality.
    quality = 0
    for i in range(0, len(solution)):
        quality = quality + solution[i] * items[i][1]
    return quality


def is_valid(solution):
    # Checking if a solution is valid by computing the total weight(same logic as above, this time using weights list
    # instead of value), and comparing it with the max_capacity.
    total_weight = 0
    for i in range(0, len(solution)):
        total_weight = solution[i] * items[i][0] + total_weight
    if total_weight <= max_capacity:
        return True
    else:
        return False


def generate_neighbor(solution, n):
    # This code determines a neighbour of a solution by 'bit flipping'.
    # For example:
    #   n = 6
    #   solution = 0 1 1 1 0 1 =>
    #   neighbour = 0 1 1 1 0 0
    neighbor = solution[:]  # shallow copy with new reference
    index = random.randint(0, n - 1)
    if neighbor[index] == 1:
        neighbor[index] = 0
    else:
        neighbor[index] = 1
    return neighbor


def determine_best_solution_taboo_search_knapsack(solutions):
    # This code determines the best value out of a list of solutions.
    best_value = solutions[0][1]
    for i in range(len(solutions)):
        if solutions[i][1] > best_value:
            best_value = solutions[i][1]
    return best_value


def determine_worst_solution_taboo_search_knapsack(solutions):
    # The exact same logic as above.
    worst_value = solutions[0][1]
    for i in range(len(solutions)):
        if solutions[i][1] < worst_value:
            worst_value = solutions[i][1]
    return worst_value


def taboo_search(items, taboo_size=200):
    # This code implements the taboo search algorithm.
    #
    #   1. Generate a random solution.
    #   2. Generate a list of neighbours of the current solution.
    #   3. Determine the best neighbour that is not in the taboo list.
    #   4. If the best neighbour is better than the current solution, then the best neighbour becomes the current
    #   solution.
    #   5. Add the current solution to the taboo list.
    #   6. If the taboo list is longer than the taboo size, then remove the first element of the taboo list.
    #   7. Repeat steps 2-6 for a given number of iterations.
    #   8. Return the best solution and its quality.

    number_of_items = len(items)
    current_solution = generate_solution(number_of_items)
    while not is_valid(current_solution):
        current_solution = generate_solution(number_of_items)
    best_solution = current_solution
    taboo_list = []

    for i in range(10):
        best_neighbor = None
        best_neighbor_value = -1

        for j in range(number_of_items):  # generate all neighbours
            neighbor = generate_neighbor(current_solution, number_of_items)
            while not is_valid(neighbor):  # if the neighbor is not valid, generate a new one
                neighbor = generate_neighbor(current_solution, number_of_items)

            if (neighbor not in taboo_list) and (determine_quality(neighbor) > best_neighbor_value):
                best_neighbor = neighbor
                best_neighbor_value = determine_quality(neighbor)

        if best_neighbor_value > determine_quality(best_solution):
            best_solution = best_neighbor

        taboo_list.append(current_solution)
        if len(taboo_list) > taboo_size:  # if the taboo list is longer than the taboo size, remove the first element
            taboo_list.pop(0)

        current_solution = best_neighbor

    return best_solution, determine_quality(best_solution)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# ------------------------------------ Taboo Search ------------------------------------
# ------------------------------------ TSP kroE100 ------------------------------------
def parse_tsp_instance(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    cities = []
    for line in lines:
        if line.strip().split()[0].isdigit():  # if the first column is a number (city number) then it is a city line
            _, x, y = line.strip().split()  # ignore the first column (city number) and split the rest into x and y
            cities.append((float(x), float(y)))
    return cities  # list of tuples


def distance_matrix(cities):
    # Example:
    # cities = [(0, 0), (1, 0), (0, 1)]
    # distances = [[0, 1, 1],
    #              [1, 0, 1],
    #              [1, 1, 0]]

    n = len(cities)
    distances = numpy.zeros((n, n))  # create a matrix of zeros
    for i in range(n):
        for j in range(n):
            distances[i, j] = numpy.linalg.norm(numpy.array(cities[i]) - numpy.array(cities[j]))
    return distances


def tsp_cost(solution, distances):
    # Given a solution (a list of cities) and a distance matrix, this function computes the total cost of the solution.
    # (cost = total distance)
    cost = 0
    for i in range(len(solution) - 1):
        cost += distances[solution[i], solution[i + 1]]  # add the distance between city i and city i+1 (indexes)
    cost += distances[solution[-1], solution[0]]  # add the distance between the last city and the first city
    return cost


def two_opt(solution, i, j):
    # 2-opt swap (helps to quickly find a good solution)
    # It aims to eliminate crossing edges by swapping two edges
    # The edges are defined by the indices i and j
    # The edges are (i-1, i) and (j, j+1)
    # The new edges are (i-1, j) and (i, j+1)

    # Example:
    #   solution = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #   i = 3
    #   j = 7
    #   new_solution = [0, 1, 2, 7, 6, 5, 4, 3, 8, 9]

    # solution[:i] contains the elements from 0 to i-1
    # solution[i:j + 1][::-1] contains the elements from i to j in reverse order ([::-1] reverses the list)
    # solution[j + 1:] contains the elements from j+1 to the end of the list
    new_solution = solution[:i] + solution[i:j + 1][::-1] + solution[j + 1:]
    return new_solution


def delta_cost_two_opt(solution, distances, i, j):
    # This function calculates the change in cost if the 2-opt swap is applied
    # Instead of calculating the whole cost of a new solution, it calculates the difference in cost
    n = len(solution)
    a, b = solution[i - 1], solution[i]  # a and b are the cities(nodes) that form the first edge
    c, d = solution[j], solution[(j + 1) % n]  # c and d are the cities(nodes) that form the second edge; %n is used to
    # make sure that d is not out of bounds (j+1 can be n)
    old_cost = distances[a, b] + distances[c, d]  # the sum of the costs of the original edges
    new_cost = distances[a, c] + distances[
        b, d]  # the sum of the costs of the new edges that form after applying the 2-opt swap
    return new_cost - old_cost


def tsp_taboo_search(cities, iterations=10, taboo_tenure=30, aspiration=0.9):
    # Aspiration criteria: if the new solution is better than the best solution by a certain amount, it is accepted
    # even if it is in the tabu list
    distances = distance_matrix(cities)  # calculate the distance matrix
    n = len(cities)
    best_solution = list(range(n))
    random.shuffle(best_solution)  # shuffle the list of cities (basically, generate a random initial solution)
    best_cost = tsp_cost(best_solution, distances)  # calculate the cost of the initial solution

    tabu_list = {tuple(best_solution)}  # the tabu list is a set of tuples (the solutions are converted to tuples)
    tabu_queue = deque(
        [tuple(best_solution)])  # the tabu queue is a queue of tuples (the solutions are converted to tuples)
    # The queue helps maintain the size of the tabu list

    for _ in range(iterations):  # iterate for a given number of iterations
        candidates = []
        delta_costs = []
        for __ in range(n):
            # generate a list of candidates by applying the 2-opt swap to the best solution
            # for each candidate, calculate the change in cost and store it in a list
            i, j = sorted(random.sample(range(1, n), 2))
            candidate = two_opt(best_solution, i, j)
            candidates.append(candidate)
            delta_costs.append(delta_cost_two_opt(best_solution, distances, i, j))

        #  Find the best candidate solution (with the lowest delta cost) and calculate its cost
        best_candidate_index = numpy.argmin(delta_costs)
        best_candidate = candidates[best_candidate_index]
        best_candidate_cost = best_cost + delta_costs[best_candidate_index]

        #  Apply the aspiration criterion to decide whether to accept the best candidate solution. If the candidate
        #  solution is better than the current best solution or if it meets the aspiration threshold and is not in the
        #  Tabu list, accept the candidate solution.
        if (best_candidate_cost < best_cost) or (
                tuple(best_candidate) not in tabu_list and best_candidate_cost < aspiration * best_cost):
            best_solution, best_cost = best_candidate, best_candidate_cost
            tabu_list.add(tuple(best_candidate))
            tabu_queue.append(tuple(best_candidate))
            if len(tabu_queue) > taboo_tenure:
                #  Update the Tabu list and queue. If the candidate solution is accepted, add it to the Tabu list and
                #  queue. If the queue exceeds the maximum size, remove the oldest item.
                removed_item = tabu_queue.popleft()
                tabu_list.remove(removed_item)

    return best_solution, best_cost


def determine_best_solution_taboo_search_tsp(solutions):
    best_cost = solutions[0][1]
    for i in range(len(solutions)):
        if solutions[i][1] < best_cost:
            best_cost = solutions[i][1]
    return best_cost


def determine_worst_solution_taboo_search_tsp(solutions):
    worst_cost = solutions[0][1]
    for i in range(len(solutions)):
        if solutions[i][1] > worst_cost:
            worst_cost = solutions[i][1]
    return worst_cost


if __name__ == '__main__':
    # max_capacity = 3000
    # items = [(56, 333), (121, 34), (200, 1231), (5, 6), (343, 44), (65, 1222), (23, 543), (434, 522), (150, 999),
    #          (90, 10000)]

    # number_of_cities = 20
    # cities = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
    #           (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19)]
    tsp_taboo_search_solutions = []
    knapsack_taboo_search_solutions = []
    with open('output.txt', 'w') as f:
        max_capacity, items = parse_data_file('rucsac-200.txt')
        cities = parse_tsp_instance('kroE100.tsp')

        for i in range(10):
            print(f"Iteration ", i + 1, file=f)

            knapsack_taboo_search_solutions.append(taboo_search(items, max_capacity))
            print(f"***Taboo search for the knapsack problem found the solution: ",
                  knapsack_taboo_search_solutions[i][0],
                  "\nwith value: ", knapsack_taboo_search_solutions[i][1], file=f)

            tsp_taboo_search_solutions.append(tsp_taboo_search(cities))
            print(f"***Taboo search for the TSP problem found the solution: ", tsp_taboo_search_solutions[i][0],
                  "\nwith distance units: ", tsp_taboo_search_solutions[i][1], file=f)
            print(f"\n\n\n", file=f)

        print(f"***Best value for the knapsack problem found by taboo search: ",
              determine_best_solution_taboo_search_knapsack(knapsack_taboo_search_solutions), file=f)
        print(f"***Worst value for the knapsack problem found by taboo search: ",
              determine_worst_solution_taboo_search_knapsack(knapsack_taboo_search_solutions), file=f)

        print(f"***Best distance for the tsp problem found by taboo search: ",
              determine_best_solution_taboo_search_tsp(tsp_taboo_search_solutions), file=f)
        print(f"***Worst distance for the tsp problem found by taboo search: ",
              determine_worst_solution_taboo_search_tsp(tsp_taboo_search_solutions), file=f)
