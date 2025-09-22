#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

#define DOMAIN_LOWER -5.0
#define DOMAIN_UPPER 5.0
#define MAX_ITER 100000000

enum CoolingSchedule { LINEAR, EXPONENTIAL, LOGARITHMIC };

double himmelblau(double x, double y) {
  return std::pow(x * x + y - 11, 2) + std::pow(x + y * y - 7, 2);
}

class State {
public:
  double x;
  double y;
  double energy;

  State(std::mt19937 &gen) {
    std::uniform_real_distribution<> dist(DOMAIN_LOWER, DOMAIN_UPPER);
    x = dist(gen);
    y = dist(gen);
    energy = himmelblau(x, y);
  }

  State(double new_x, double new_y) {
    x = new_x;
    y = new_y;
    energy = himmelblau(x, y);
  }

  State get_neighbour(std::mt19937 &gen, double temperature) const {
    std::uniform_real_distribution<> dist_modifier(-temperature / 100.0,
                                                   temperature / 100.0);
    double new_x = x + dist_modifier(gen);
    double new_y = y + dist_modifier(gen);
    new_x = std::max(DOMAIN_LOWER, std::min(DOMAIN_UPPER, new_x));
    new_y = std::max(DOMAIN_LOWER, std::min(DOMAIN_UPPER, new_y));

    return State(new_x, new_y);
  }
};

double stateAcceptanceProb(State current, State neighbour, double temperature) {
  return neighbour.energy < current.energy
             ? 1.0
             : std::exp((current.energy - neighbour.energy) / temperature);
}

double linearCooling(double s_temp, float alpha, int iteration) {
  return std::max(0.0, s_temp - alpha * iteration);
}

double exponentialCooling(double s_temp, float alpha, int iteration) {
  return s_temp * std::pow(alpha, iteration);
}

double logarithmicCooling(double s_temp, int iteration) {
  return s_temp / (std::log(1.0 + iteration));
}

void simulated_annealing(CoolingSchedule cooling_type, double s_temp,
                         double alpha, int ations) {
  std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<> dist_prob(0.0, 1.0);

  State current_state(gen);
  State best_state = current_state;

  double temp = s_temp;

  std::cout << "SA with initial state: (" << current_state.x << ", "
            << current_state.y << ") and energy: " << current_state.energy
            << std::endl;

  for (int i = 0; i < ations; ++i) {
    State neighbour_state = current_state.get_neighbour(gen, temp);
    double acceptance_prob =
        stateAcceptanceProb(current_state, neighbour_state, temp);

    if (acceptance_prob > dist_prob(gen))
      current_state = neighbour_state;

    if (current_state.energy < best_state.energy)
      best_state = current_state;

    switch (cooling_type) {
    case LINEAR:
      temp = linearCooling(s_temp, alpha, i);
      break;
    case EXPONENTIAL:
      temp = exponentialCooling(s_temp, alpha, i);
      break;
    case LOGARITHMIC:
      temp = logarithmicCooling(s_temp, i);
      break;
    }
  }

  std::cout << "Final result after " << ations << " iterations:" << std::endl;
  std::cout << "Optimal state found at: (" << best_state.x << ", "
            << best_state.y << ")" << std::endl;
  std::cout << "Minimum energy (function value): " << best_state.energy
            << std::endl
            << std::endl;
}

int main() {
  double temp = 1000.0;
  std::cout << "Linear cooling" << std::endl;
  simulated_annealing(LINEAR, temp, 100.0 / MAX_ITER, MAX_ITER);
  std::cout << "Exponential cooling" << std::endl;
  simulated_annealing(EXPONENTIAL, temp, 0.995, MAX_ITER);
  std::cout << "Logarithmic cooling" << std::endl;
  simulated_annealing(LOGARITHMIC, temp, 0.0, MAX_ITER);
}
