import torch
import numpy as np
import time

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda" if torch.cuda.is_available() else "cpu")

dt = 1e-5
T = 2
np_r_rx = 5
np_D = 79.4
np_r_mol = 0.5
np_num_of_tx = 2
M = 10000

r_rx = torch.tensor(np_r_rx, device=device, dtype=torch.float64)
D = torch.tensor(np_D, device=device, dtype=torch.float64)
r_mol = torch.tensor(np_r_mol, device=device, dtype=torch.float64)
num_of_tx = torch.tensor(np_num_of_tx, device=device, dtype=torch.float64)

steps = int(T / dt)
print(steps)

np_center_rx = np.array([0, 0, 0])
np_single_mol = np.array([8.517771814413306, 1.959802641534183, 7.304539092939853])

center_rx = torch.tensor(np_center_rx, device=device, dtype=torch.float64)
single_mol = torch.tensor(np_single_mol, device=device, dtype=torch.float64)

A_molecules = torch.tile(single_mol, (M, 1)).to(device)
B_molecules = torch.tile(single_mol, (M, 1)).to(device)

file = open('results.txt', 'w')

sigma = torch.sqrt(2 * D * dt).item()

for i in range(steps):
    delta_A = torch.normal(0, sigma, (M, 3), device=device, dtype=torch.float64)
    delta_B = torch.normal(0, sigma, (M, 3), device=device, dtype=torch.float64)

    A_molecules += delta_A
    B_molecules += delta_B

    distances_A = torch.norm(A_molecules - center_rx, dim=1)
    absorbed_A = distances_A <= r_rx

    distances_B = torch.norm(B_molecules - center_rx, dim=1)
    absorbed_B = distances_B <= r_rx

    filtered_A = A_molecules[absorbed_A]
    filtered_B = B_molecules[absorbed_B]

    A_molecules[absorbed_A] = torch.tensor([1e8, 1e8, 1e8], device=device, dtype=torch.float64)
    B_molecules[absorbed_B] = torch.tensor([1e8, 1e8, 1e8], device=device, dtype=torch.float64)

    for mol in filtered_A:
        file.write(f"{mol[0].item()} {mol[1].item()} {mol[2].item()} 0\n")

    for mol in filtered_B:
        file.write(f"{mol[0].item()} {mol[1].item()} {mol[2].item()} 1\n")

file.close()
end_time = time.time()

print(f"time passed: {end_time - start_time}")

