import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

column = 1

font = {'family' : 'Times New Roman',
        #'weight' : 'bold',
        'size'   : 9}

plt.rc('font', **font)

def square_elements(arr):
    result = []
    for num in arr:
        result.append(num ** 2)
    return result

fs         = 300  # Sampling frequency
fs_imu     = 150
fs_control = 100
cutoff     = 30  # Cutoff frequency in Hz
order      = 5  # Order of the filter

# Normalize the frequency

nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

# Get the filter coefficients

b, a = signal.butter(order, normal_cutoff, btype='low', analog=False) # for pose data
b2, a2 = signal.butter(order, cutoff / (0.5 * fs_imu), btype='low', analog=False) # for imu data
b3, a3 = signal.butter(order, cutoff / (0.5 * fs_control), btype='low', analog=False) # for control data

weight = 1.
g      = 9.81


# Display
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
plt.rcParams['axes.labelsize'] = 8  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Define a color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bottom_colors = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown

# Function to set dense grid
def set_dense_grid(ax):
    ax.grid(True, which='major', linestyle='-', color='lightgray', linewidth='2', alpha=0.8)
    # ax.grid(True, which='minor', linestyle=':', color='gray', linewidth='1', alpha=0.8)
    ax.minorticks_on()  # Enable minor ticks

# Function to set y-axis limits with a margin
def set_y_limits(ax, data_list, margin=0.1):
    data_min = min([np.min(data) for data in data_list])
    data_max = max([np.max(data) for data in data_list])
    range_margin = (data_max - data_min) * margin
    ax.ylim(data_min - range_margin, data_max + range_margin)

def plot_vertical_lines(plt, x_coords): 
    for x in x_coords: 
        plt.axvline(x=x, color='lightcoral', linestyle='--', zorder=0)

def plot_vertical_lines(plt, x_coords, col_type): 
    for x, type in zip(x_coords, col_type):
        if type == 1:
            plt.axvline(x=x, color='lightcoral', linestyle='--', zorder=0)
        elif type == 2:
            plt.axvline(x=x, color='cornflowerblue', linestyle='--', zorder=0)
        else:
            plt.axvline(x=x, color='plum', linestyle='--', zorder=0)

def plot_vertical_line(plt, x_coord, col_type): 
    if col_type == 1:
        plt.axvline(x=x_coord, color='lightcoral', linestyle='--', zorder=0)
    elif col_type == 2:
        plt.axvline(x=x_coord, color='cornflowerblue', linestyle='--', zorder=0)
    else:
        plt.axvline(x=x_coord, color='plum', linestyle='--', zorder=0)

# file = 'Test_Data/drive-download-20230927T134757Z-001/actuator.txt'
def find_index(time_sequence, time0):
    for index, time in enumerate(time_sequence):
        if time >= time0:
            return index
    return -1  # Return -1 if no such time is found

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    return r.as_matrix()

# data = np.loadtxt(file)

# time = data[:,0] 
# pwm1 = data[:,1] 
# pwm2 = data[:,2] 
# pwm3 = data[:,3]
# pwm4 = data[:,4]

# time_duration = (time[1:-1] - time[0:-2])/1e6;

# plt.figure(1)
# plt.plot(time - time[0], pwm1, '-r', time- time[0], pwm2, '-b', time- time[0], pwm3, '-g', time- time[0], pwm4, '-m')
# plt.legend(['pwm1', 'pwm2', 'pwm3', 'pwm4'])
# plt.xlabel('Time') #注意后面的字体属性
# plt.ylabel('PWM')
# plt.title('PWM')  

# plt.figure(2)
# plt.plot(time_duration)
# plt.ylabel('duration (ms)')
# plt.title('Actuator time duration')  

# plt.savefig('PWM.jpg')

file      = 'Data/control_2024-12-04-15-29-53.txt'
pose_file = 'Data/pose_2024-12-04-15-29-53.txt'
imu_file  = 'Data/imu_2024-12-04-15-29-53.txt'
# file = 'Test_Data/rsm_2023-10-19-11-31-36.txt'

actuator_t_delay = 0.05
data = np.loadtxt(file)
time = data[:,0] 
time_begin = time[0]
time  = (time - time_begin)/1e9 + actuator_t_delay

bodyrate_x = data[:,1]
bodyrate_y = data[:,2]
bodyrate_z = data[:,3]
thrust     = data[:,4]

thrust     = signal.filtfilt(b3, a3, thrust)
bodyrate_x = signal.filtfilt(b3, a3, bodyrate_x)
bodyrate_y = signal.filtfilt(b3, a3, bodyrate_y)
bodyrate_z = signal.filtfilt(b3, a3, bodyrate_z)

pose_data = np.loadtxt(pose_file)
pose_time = pose_data[:,0] 
pose_time = (pose_time - time_begin)/1e9
pose_x    = pose_data[:,1]
pose_y    = pose_data[:,2]
pose_z    = pose_data[:,3]

q_x    = pose_data[:,4]
q_y    = pose_data[:,5]
q_z    = pose_data[:,6]
q_w    = pose_data[:,7]

interpolator_qx = interp1d(pose_time, q_x, kind='linear', fill_value='extrapolate')
interpolated_angular_qx = interpolator_qx(time)

interpolator_qy = interp1d(pose_time, q_y, kind='linear', fill_value='extrapolate')
interpolated_angular_qy = interpolator_qy(time)

interpolator_qz = interp1d(pose_time, q_z, kind='linear', fill_value='extrapolate')
interpolated_angular_qz = interpolator_qz(time)

interpolator_qw = interp1d(pose_time, q_w, kind='linear', fill_value='extrapolate')
interpolated_angular_qw = interpolator_qw(time)

# Compute rotated thrust vector for each quaternion in pose_data
rotated_thrust_vectors = []
for i in range(len(interpolated_angular_qx)):
    q = [interpolated_angular_qx[i], interpolated_angular_qy[i], interpolated_angular_qz[i], interpolated_angular_qw[i]]
    thrust_vector = np.array([0, 0, thrust[i]*42])
    rotation_matrix = quaternion_to_rotation_matrix(q)
    rotated_thrust_vector = rotation_matrix.dot(thrust_vector) - np.array([0, 0, weight * g])
    rotated_thrust_vectors.append(rotated_thrust_vector)

# Convert list to numpy array for easier handling
rotated_thrust_vectors = np.array(rotated_thrust_vectors)

# Plot all rotated thrust vectors
fig, ax = plt.subplots()
time_indices = range(len(rotated_thrust_vectors))

for i in range(3):  # Assuming the thrust vector is 3-dimensional
    ax.plot(time_indices, rotated_thrust_vectors[:, i], label=f'Component {i+1}')

ax.set_title('Rotated Thrust Vectors')
ax.set_xlabel('Time Index')
ax.set_ylabel('Value')
ax.legend()


# Apply the filter to the data

# pose_x = signal.filtfilt(b, a, pose_x)
# pose_y = signal.filtfilt(b, a, pose_y)
# for i in range(len(pose_x)):
#     if pose_x[i] > 2.3:
#         pose_x[i] = 0
#     if pose_x[i] < -2.3:
#         pose_x[i] = 0
time_duration = (time[1:-1] - time[0:-2])/1e6
rsm = []
rsm_t = []
rsm_t_f = []

seq = []
left = 0
right = 182



imu_data = np.loadtxt(imu_file)
imu_time = imu_data[:,0] 
imu_time = (imu_time - time_begin) / 1e9

# 1: static or moving box (plane) collision; 2: stick collision; 3: Landing collision
collision_type = [3, 3,   3,   3,   3,    3,    3,  3,   3,   3,   3,   3,    1,    1,    1,     2,     2,     2,     2,     2,     2,     2,     2,     2,    1,      1,      1,      1,      1,     1,      ]
collision_time = [1, 2.5, 5.0, 6.5, 10.0, 17.5, 93, 149, 161, 119, 134, 91.8, 23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 67.73, 71.73, 75.66, 80.67, 85.6, 103.17, 111.87, 126.54, 142.09, 155.7, 168.024]

# collision_type = [1,    1,    1,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,    1,       1,      1,      1,     1,     1,       3]
# collision_time = [23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 67.73, 71.73, 75.66, 80.67, 81.21, 82.19, 85.6, 102.091, 111.87, 126.54, 142.2, 155.7, 168.024, 181.151]

sample_index = 5
duration = 0.1
offset   = 0.0
start_time = collision_time[sample_index] - duration - offset
end_time   = collision_time[sample_index] - offset
sample_index
start_index = find_index(imu_time, start_time)
end_index   = find_index(imu_time, end_time)

angular_velocity_x = imu_data[:,1]
angular_velocity_y = imu_data[:,2]
angular_velocity_z = imu_data[:,3]

linear_acceleration_x = imu_data[:,4]
linear_acceleration_y = imu_data[:,5]
linear_acceleration_z = imu_data[:,6]

# Angular velocity
angular_velocity_x = signal.filtfilt(b2, a2, angular_velocity_x)
angular_velocity_y = signal.filtfilt(b2, a2, angular_velocity_y)
angular_velocity_z = signal.filtfilt(b2, a2, angular_velocity_z)

# Linear Acceleration
linear_acceleration_x = signal.filtfilt(b2, a2, linear_acceleration_x)
linear_acceleration_y = signal.filtfilt(b2, a2, linear_acceleration_y)
linear_acceleration_z = signal.filtfilt(b2, a2, linear_acceleration_z)

# Position xy
# plt.figure(figsize=(3.7, 2.7))
# set_dense_grid(plt)
# plt.plot(pose_x, pose_y, color=colors[0], linestyle='-', linewidth=1 )
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.title('Position')
# plt.savefig('figs/position_xy.svg', format='svg')

# Position XYZ
# fig = plt.figure(figsize=(3.7, 2.7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(pose_x, pose_y, pose_z, color=colors[0], linestyle='-', linewidth=1 )
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('Position')
# plt.savefig('figs/position_xyz.svg', format='svg')


plt.figure(figsize=(5.7, 2.7))
set_dense_grid(plt)
# plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
plt.plot(imu_time, angular_velocity_y, color=colors[1], linestyle='-', linewidth=1)
plt.plot(imu_time, angular_velocity_z, color=colors[2], linestyle='-', linewidth=1)
plt.plot(imu_time, angular_velocity_x, color=colors[0], linestyle='-', linewidth=1)
plot_vertical_lines(plt, collision_time, collision_type)
plt.xlim(left, right)
# plt.ylim(-2.5, 2.5)
plt.legend(['Y', 'Z', 'X'], frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Angular velocity')   
plt.savefig('figs/angular_velocity.svg', format='svg')


plt.figure(figsize=(5.7, 2.7))
set_dense_grid(plt)
# plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
plt.plot(imu_time, linear_acceleration_y, color=colors[1], linestyle='-', linewidth=1)
plt.plot(imu_time, linear_acceleration_z, color=colors[2], linestyle='-', linewidth=1)
plt.plot(imu_time, linear_acceleration_x, color=colors[0], linestyle='-', linewidth=1)
plt.plot(time,     thrust*21,             color=bottom_colors[0], linestyle='-', linewidth=1)
plot_vertical_lines(plt, collision_time, collision_type)
plt.xlim(left, right)
plt.legend(['Y', 'Z', 'X', 'T'], frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Linear acceleration (m/s2)')
plt.title('Linear acceleration')   
plt.savefig('figs/linear_acceleration.svg', format='svg')

# plt.figure(figsize=(3.7, 2.7))

# set_dense_grid(plt)
# # plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
# plt.plot(time, bodyrate_x, color=colors[0], linestyle='-', linewidth=1)
# plt.plot(time, bodyrate_y, color=colors[1], linestyle='-', linewidth=1)
# plt.plot(time, bodyrate_z, color=colors[2], linestyle='-', linewidth=1)
# plot_vertical_lines(plt, collision_time, collision_type)
# plt.xlim(left, right)
# plt.legend(['X', 'Y', 'Z'], frameon=False)
# plt.xlabel('Time (s)')
# plt.ylabel('Body rate (rad/s)')
# plt.title('Body rate command')   
# plt.savefig('figs/bodyrate_cmd.svg', format='svg')

# plt.figure(figsize=(5.7, 2.7))
# set_dense_grid(plt)

# Interpolate angular_velocity_y based on time
interpolator_y = interp1d(imu_time, angular_velocity_y, kind='linear', fill_value='extrapolate')
interpolated_angular_velocity_y = interpolator_y(time)
interpolator_x = interp1d(imu_time, angular_velocity_x, kind='linear', fill_value='extrapolate')
interpolated_angular_velocity_x = interpolator_x(time)
interpolator_z = interp1d(imu_time, angular_velocity_z, kind='linear', fill_value='extrapolate')
interpolated_angular_velocity_z = interpolator_z(time)
# Compute errors
errors_y = bodyrate_y - interpolated_angular_velocity_y
errors_x = bodyrate_x - interpolated_angular_velocity_x
errors_z = bodyrate_z - interpolated_angular_velocity_z

fig, (ax1, ax2, ax3) = plt.subplots(3)
# plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
ax1.plot(imu_time, angular_velocity_y, color=colors[1], linestyle='-', linewidth=1)
ax1.plot(time, bodyrate_y, color=bottom_colors[1], linestyle='-', linewidth=1)
ax1.plot(time, errors_y,   color='gray',           linestyle='--', linewidth=1)
ax1.legend(['Y', 'cY'], frameon=False)
ax1.set_title('Y')

ax2.plot(imu_time, angular_velocity_z, color=colors[2], linestyle='-', linewidth=1)
ax2.plot(time, bodyrate_z, color=bottom_colors[2], linestyle='-', linewidth=1)
ax2.plot(time, errors_z,   color='gray',           linestyle='--', linewidth=1)
ax2.legend(['Z', 'cZ'], frameon=False)
ax2.set_title('Z')

ax3.plot(imu_time, angular_velocity_x, color=colors[0], linestyle='-', linewidth=1)
ax3.plot(time, bodyrate_x, color=bottom_colors[0], linestyle='-', linewidth=1)
ax3.plot(time, errors_x,   color='gray',           linestyle='--', linewidth=1)
ax3.legend(['X', 'cX'], frameon=False)
ax3.set_title('X')

# plt.xlim(left, right)
# # plt.ylim(-2.5, 2.5)
# plt.legend(['Y', 'Z', 'X', 'cY', 'cZ', 'cX'], frameon=False)
# plt.xlabel('Time (s)')
# plt.ylabel('Angular velocity (rad/s) and Bodyrate Command')
# plt.title('Angular velocity')   
# plt.savefig('figs/angular_velocity_bodyrate_cmd.svg', format='svg')

# # plt.figure(figsize=(3.7, 3.0))
# fig, ax1 = plt.subplots(figsize=(3.7, 2.7))
# set_dense_grid(plt)

# ax1.plot(time, thrust, color=colors[0], linestyle='-', linewidth=1 )
# ax1.legend(['Thrust'], loc='best', frameon=False)

# ax2 = ax1.twinx()

# y_col = [1, 1, 1]
# ax2.plot(pose_time, pose_x, color=colors[1], linestyle='-', linewidth=1 )
# ax2.plot(pose_time, pose_y, color=colors[2], linestyle='-', linewidth=1 )
# # ax2.scatter(collision_time, y_col, color='red', marker='*', s=30, zorder = 3)
# plot_vertical_lines(plt, collision_time, collision_type)
# plt.xlim(left, right)
# # ax2.set_ylim(-0.5, 2.5)
# ax2.legend(['Pose X', 'Pose Y'], loc='best', frameon=False)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Thrust/Max')
# ax2.set_ylabel('Position X-Y (m)')
# plt.title('Thrust command and position X-Y')   
# plt.savefig('figs/Thrust_pos_xy.svg', format='svg')

# plt.figure(figsize=(3.7, 2.7))
# set_dense_grid(plt)
# # plt.grid(True, linewidth = 2, color='gray', alpha=0.8)

# plt.plot(imu_time[start_index:end_index], linear_acceleration_y[start_index:end_index], color=colors[1], linestyle='-', linewidth=1)
# plt.plot(imu_time[start_index:end_index], linear_acceleration_z[start_index:end_index], color=colors[2], linestyle='-', linewidth=1)
# plt.plot(imu_time[start_index:end_index], linear_acceleration_x[start_index:end_index], color=colors[0], linestyle='-', linewidth=1)
# plot_vertical_line(plt, collision_time[sample_index], collision_type[sample_index])
# plt.legend(['Y', 'Z', 'X'], frameon=False)
# plt.xlabel('Time (s)')
# plt.ylabel('Linear acceleration (m/s2)')
# plt.title('Linear acceleration sample')   

# plt.figure(figsize=(3.7, 2.7))
# set_dense_grid(plt)
# # plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
# plt.plot(imu_time[start_index:end_index], angular_velocity_y[start_index:end_index], color=colors[1], linestyle='-', linewidth=1)
# plt.plot(imu_time[start_index:end_index], angular_velocity_z[start_index:end_index], color=colors[2], linestyle='-', linewidth=1)
# plt.plot(imu_time[start_index:end_index], angular_velocity_x[start_index:end_index], color=colors[0], linestyle='-', linewidth=1)
# plot_vertical_line(plt, collision_time[sample_index], collision_type[sample_index])
# # plt.xlim(left, right)
# # plt.ylim(-2.5, 2.5)
# plt.legend(['Y', 'Z', 'X'], frameon=False)
# plt.xlabel('Time (s)')
# plt.ylabel('Angular velocity (rad/s)')
# plt.title('Angular velocity sample')   
# plt.savefig('figs/angular_velocity.svg', format='svg')

# fig, ax1 = plt.subplots(figsize=(3.7, 2.7))
# set_dense_grid(plt)

# start_index = find_index(time, start_time)
# end_index   = find_index(time, end_time)

# ax1.plot(time[start_index:end_index], thrust[start_index:end_index], color=colors[0], linestyle='-', linewidth=1 )
# ax1.legend(['Thrust'], loc='best', frameon=False)

# ax2 = ax1.twinx()

# start_index = find_index(pose_time, start_time)
# end_index   = find_index(pose_time, end_time)

# y_col = [1, 1, 1]
# ax2.plot(pose_time[start_index:end_index], pose_x[start_index:end_index], color=colors[1], linestyle='-', linewidth=1 )
# ax2.plot(pose_time[start_index:end_index], pose_y[start_index:end_index], color=colors[2], linestyle='-', linewidth=1 )
# plot_vertical_line(plt, collision_time[sample_index], collision_type[sample_index])
# # plt.xlim(left, right)
# # ax2.set_ylim(-0.5, 2.5)
# ax2.legend(['Pose X', 'Pose Y'], loc='best', frameon=False)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Thrust/Max')
# ax2.set_ylabel('Position X-Y (m)')
# plt.title('Thrust command and position X-Y sample')   
# plt.savefig('figs/Thrust_pos_xy.svg', format='svg')

plt.show()



