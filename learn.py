import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

theta = np.linspace(0, np.pi, 1000)
m = 0.5
cos_theta = np.cos(theta)
cos_theta_plus_m = np.cos(theta + m)

th = np.cos(np.pi - m)
mm = np.sin(np.pi - m) * m

phi = np.where(cos_theta > th, cos_theta_plus_m, cos_theta - mm)

plt.figure(figsize=(12, 6))
plt.plot(theta, cos_theta, label='cos(θ)')
plt.plot(theta, cos_theta_plus_m, label='cos(θ + m)', linestyle=':')
plt.plot(theta, phi, label='ArcFace 修正后的 φ(θ)', linestyle='--', linewidth=2)
plt.axvline(x=np.pi - m, color='red', linestyle='--', label='θ = π - m')

plt.scatter([np.pi - m], [np.cos(np.pi)], color='black', label='临界断点')
plt.text(np.pi - m + 0.05, -0.8, 'θ = π - m', color='red')

plt.title('ArcFace 中 cos(θ + m) 的修正机制')
plt.xlabel('θ (弧度)')
plt.ylabel('值')
plt.legend()
plt.grid()
plt.show()
