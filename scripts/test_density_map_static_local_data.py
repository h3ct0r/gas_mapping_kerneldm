import csv
import matplotlib.pyplot as plt
from td_kernel_dmvw import TDKernelDMVW

filename = '../data/gas_data_spot2.txt'
positions_x = []
positions_y = []
concentrations = []
wind_directions = []
wind_speeds = []
timestamps = []

# quoting=csv.QUOTE_NONNUMERIC to automatically converto to float
timestamp_count = 0
for row in csv.reader(open(filename), delimiter=' ', quoting=csv.QUOTE_NONNUMERIC):
    positions_x.append(row[0])
    positions_y.append(row[1])
    concentrations.append(row[2])
    wind_directions.append(0)
    wind_speeds.append(0)

    timestamps.append(timestamp_count)
    timestamp_count += 1

# plt.scatter(positions_x, positions_y)
# plt.show()

# Set parameters
min_x = min(positions_x)
min_y = min(positions_y)
max_x = max(positions_x)
max_y = max(positions_y)

print("min_x:", min_x, "max_x:", max_x, "min_y:", min_y, "max_y:", max_y)

cell_size = 0.1
kernel_size = 8 * cell_size
wind_scale = 0.05
time_scale = 0.001
evaluation_radius = 8 * kernel_size

# call Kernel
kernel = TDKernelDMVW(min_x, min_y, max_x, max_y, cell_size, kernel_size, wind_scale, time_scale,
                      low_confidence_calculation_zero=True, evaluation_radius=evaluation_radius)
                      #cell_interconnections=None, target_grid_data=None)

kernel.set_measurements(positions_x, positions_y, concentrations, timestamps, wind_speeds, wind_directions)
kernel.calculate_maps()

# Show result as map
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Kernel DM+V')

ax1.set_aspect(1.0)
ax1.title.set_text("mean map")
ax1.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.mean_map)

ax2.set_aspect(1.0)
ax2.title.set_text("variance map")
ax2.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.variance_map)
#ax2.imshow(kernel.variance_map)

ax3.set_aspect(1.0)
ax3.title.set_text("confidence map")
ax3.contourf(kernel.cell_grid_x, kernel.cell_grid_y, kernel.confidence_map)

plt.show()
plt.savefig('/tmp/kernel_dmv_local.png')
plt.close()
