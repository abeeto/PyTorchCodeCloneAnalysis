from occupancy_calculator import CudaOccupancyCalculator
calculator = CudaOccupancyCalculator("8.0")
calculator.set_inputs(32*16, 32,
                      "8.0", 16)
occupancy = calculator.occupancyOfMultiprocessor()
print(occupancy)