import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import mean_squared_error, r2_score  
import logging

logging.basicConfig(filename='training_log_pc2pc.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define custom dataset
class VortexRingDataset(Dataset):
    def __init__(self, x, y, z, xn, yn, zn):
        self.x = x
        self.y = y
        self.z = z
        self.xn = xn
        self.yn = yn
        self.zn = zn

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.z[idx],
            self.xn[idx],
            self.yn[idx],
            self.zn[idx]
        )

# Define the fully connected network
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            nn.Tanh()
        )
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                nn.Tanh()
            ) for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class PINN(nn.Module):
    def __init__(self, layer_sizes, rho0, g, device):
        super(PINN, self).__init__()
        self.net = FCN(layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3]).to(device)
        self.rho0 = rho0
        self.g = torch.tensor([0, 0, -g], dtype=torch.float32, device=device)
        self.device = device

    # MSE error for predicted and ground truth locations
    def data_loss(self, x_current, y_current, z_current, xn, yn, zn):
        mse = nn.MSELoss()
        loss = mse(x_current, xn) + mse(y_current, yn) + mse(z_current, zn)
        return loss

    def equation_loss(self, x, y, z, t):

        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)

        pred = self.forward(x, y, z, t)
        u, v, w, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

        # First-order derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]

        # Mass continuity equation
        f_mass = u_x + v_y + w_z

        # Euler momentum equations
        f_u = u_t + (u * u_x + v * u_y + w * u_z) + (1 / self.rho0) * p_x
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + (1 / self.rho0) * p_y
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + (1 / self.rho0) * p_z + self.g[2]

        mse = nn.MSELoss()
        zeros = torch.zeros_like(x, device=self.device)
        equation_loss = mse(f_u, zeros) + mse(f_v, zeros) + mse(f_w, zeros) + mse(f_mass, zeros)

        return equation_loss

    def forward(self, x, y, z, t):
        input_tensor = torch.cat([x, y, z, t], dim=1)
        output = self.net(input_tensor)
        return output

def predict_positions(model, x_init, y_init, z_init, t_init, num_steps, delta_t):
    x_current = x_init.clone().detach().to(model.device)
    y_current = y_init.clone().detach().to(model.device)
    z_current = z_init.clone().detach().to(model.device)
    t_current = t_init.clone().detach().to(model.device)

    delta_t_tensor = torch.tensor(delta_t, dtype=torch.float32, device=model.device)

    for _ in range(num_steps):
        t_current = t_current + delta_t_tensor

        x_current.requires_grad_(True)
        y_current.requires_grad_(True)
        z_current.requires_grad_(True)
        t_current.requires_grad_(True)

        pred = model(x_current, y_current, z_current, t_current)
        u_pred, v_pred, w_pred, _ = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        x_current = x_current + u_pred * delta_t_tensor
        y_current = y_current + v_pred * delta_t_tensor
        z_current = z_current + w_pred * delta_t_tensor

    return x_current, y_current, z_current

def train_pinn(model, train_loader, val_loader, epochs, learning_rate, num_steps=1, pinn_ratio=0.5, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_losses = []
    equation_losses = []
    total_losses = []
    val_total_losses = []

    delta_t = 1.0 / num_steps
    delta_t_tensor = torch.tensor(delta_t, dtype=torch.float32, device=model.device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    last_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_data_loss = 0.0
        total_eq_loss = 0.0

        for batch in train_loader:
            x_batch, y_batch, z_batch, xn_batch, yn_batch, zn_batch = [b.to(model.device) for b in batch]

            x_batch = x_batch.clone()
            y_batch = y_batch.clone()
            z_batch = z_batch.clone()

            t_batch = torch.zeros(len(x_batch), 1, dtype=torch.float32, device=model.device)

            optimizer.zero_grad()

            x_current = x_batch.clone()
            y_current = y_batch.clone()
            z_current = z_batch.clone()
            t_current = t_batch.clone()

            total_eq_loss_batch = 0.0

            for step in range(num_steps):
                t_current.requires_grad_(True)
                x_current.requires_grad_(True)
                y_current.requires_grad_(True)
                z_current.requires_grad_(True)

                pred = model(x_current, y_current, z_current, t_current)
                u_pred, v_pred, w_pred, _ = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

                x_current = x_current + u_pred * delta_t_tensor
                y_current = y_current + v_pred * delta_t_tensor
                z_current = z_current + w_pred * delta_t_tensor
                t_current = t_current + delta_t_tensor

                eq_loss = model.equation_loss(x_current, y_current, z_current, t_current)
                total_eq_loss_batch += eq_loss

            total_eq_loss_batch /= num_steps
            data_loss = model.data_loss(x_current, y_current, z_current, xn_batch, yn_batch, zn_batch)
            total_loss = pinn_ratio * total_eq_loss_batch + (1 - pinn_ratio) * data_loss

            total_data_loss += data_loss.item()
            total_eq_loss += total_eq_loss_batch.item()

            total_loss.backward()
            optimizer.step()

        data_losses.append(total_data_loss / len(train_loader))
        equation_losses.append(total_eq_loss / len(train_loader))
        total_losses.append((total_data_loss + total_eq_loss) / len(train_loader))

        # Validation
        model.eval()
        val_total_loss = 0.0

        for batch in val_loader:
            x_batch, y_batch, z_batch, xn_batch, yn_batch, zn_batch = [b.to(model.device) for b in batch]

            x_batch = x_batch.clone()
            y_batch = y_batch.clone()
            z_batch = z_batch.clone()

            t_batch = torch.zeros(len(x_batch), 1, dtype=torch.float32, device=model.device)

            x_current = x_batch.clone()
            y_current = y_batch.clone()
            z_current = z_batch.clone()
            t_current = t_batch.clone()

            total_eq_loss_batch = 0.0
            for step in range(num_steps):
                t_current.requires_grad_(True)
                x_current.requires_grad_(True)
                y_current.requires_grad_(True)
                z_current.requires_grad_(True)

                pred = model(x_current, y_current, z_current, t_current)
                u_pred, v_pred, w_pred, _ = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

                x_current = x_current + u_pred * delta_t_tensor
                y_current = y_current + v_pred * delta_t_tensor
                z_current = z_current + w_pred * delta_t_tensor
                t_current = t_current + delta_t_tensor

                eq_loss = model.equation_loss(x_current, y_current, z_current, t_current)
                total_eq_loss_batch += eq_loss

            total_eq_loss_batch /= num_steps
            data_loss = model.data_loss(x_current, y_current, z_current, xn_batch, yn_batch, zn_batch)
            total_loss = pinn_ratio * total_eq_loss_batch + (1 - pinn_ratio) * data_loss
            val_total_loss += total_loss.item()

        val_total_loss /= len(val_loader)
        val_total_losses.append(val_total_loss)

        # Early Stopping Check
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            last_epoch = epoch 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break

    logging.info(f"Final Epoch: {last_epoch}, Learning Rate: {learning_rate}, Num Steps: {num_steps}, "
                 f"PINN Ratio: {pinn_ratio}, Data Loss: {data_losses[last_epoch]:.6f}, "
                 f"Equation Loss: {equation_losses[last_epoch]:.6f}, "
                 f"Total Loss: {total_losses[last_epoch]:.6f}, "
                 f"Validation Loss: {val_total_losses[last_epoch]:.6f}")

    model.load_state_dict(best_model_state)

    return model, data_losses, equation_losses, total_losses, val_total_losses

def visualize_results(
    data_losses, equation_losses, total_losses, val_total_losses,
    xn_data_normalized, yn_data_normalized, zn_data_normalized,
    x_pred, y_pred, z_pred,
    lr, num_steps, pinn_ratio
):
    # Plot data loss and equation loss
    output_dir = 'results_pc2pc'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(data_losses, label='Data Loss')
    plt.plot(equation_losses, label='Equation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(total_losses, label='Training Total Loss')
    plt.plot(val_total_losses, label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Total Loss over Epochs (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(total_losses, label='Training Total Loss')
    plt.plot(val_total_losses, label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'losses_over_epochs_lr{lr}_steps{num_steps}_pinn{pinn_ratio}.png'))
    plt.close()

    # 2D scatter plots for predictions vs normalized target data
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(xn_data_normalized.cpu().detach().numpy(), x_pred.cpu().detach().numpy(), s=5)
    plt.xlabel('xn_data_normalized')
    plt.ylabel('x_pred')
    plt.title(f'Predicted vs Actual X (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')

    plt.subplot(1, 3, 2)
    plt.scatter(yn_data_normalized.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), s=5)
    plt.xlabel('yn_data_normalized')
    plt.ylabel('y_pred')
    plt.title(f'Predicted vs Actual Y (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')

    plt.subplot(1, 3, 3)
    plt.scatter(zn_data_normalized.cpu().detach().numpy(), z_pred.cpu().detach().numpy(), s=5)
    plt.xlabel('zn_data_normalized')
    plt.ylabel('z_pred')
    plt.title(f'Predicted vs Actual Z (LR={lr}, Steps={num_steps}, PINN Ratio={pinn_ratio})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_vs_actuals_lr{lr}_steps{num_steps}_pinn{pinn_ratio}.png'))
    plt.close()

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def grid_search(layer_sizes, rho0, g, device, train_loader, val_loader, epochs, learning_rates, num_steps_list, pinn_ratios, patience):
    best_model = None
    best_loss = float('inf')
    best_params = None
    results = []

    for lr in learning_rates:
        for num_steps in num_steps_list:
            for pinn_ratio in pinn_ratios:
                print(f"Training with learning rate = {lr}, num_steps = {num_steps}, pinn_ratio = {pinn_ratio}")

                model = PINN(layer_sizes, rho0, g, device)
                model.apply(reset_weights)

                model, data_losses, equation_losses, total_losses, val_total_losses = train_pinn(
                    model, train_loader, val_loader, epochs, lr, num_steps, pinn_ratio, patience
                )

                all_x_test, all_y_test, all_z_test = [], [], []
                all_xn_test, all_yn_test, all_zn_test = [], [], []
                all_x_pred, all_y_pred, all_z_pred = [], [], []

                for batch in val_loader:  
                    x_batch, y_batch, z_batch, xn_batch, yn_batch, zn_batch = [b.to(model.device) for b in batch]
                    t_batch = torch.zeros(len(x_batch), 1, dtype=torch.float32, device=model.device)

                    x_pred_batch, y_pred_batch, z_pred_batch = predict_positions(model, x_batch, y_batch, z_batch, t_batch, num_steps, delta_t=1.0 / num_steps)

                    all_x_test.append(x_batch)
                    all_y_test.append(y_batch)
                    all_z_test.append(z_batch)
                    all_xn_test.append(xn_batch)
                    all_yn_test.append(yn_batch)
                    all_zn_test.append(zn_batch)

                    all_x_pred.append(x_pred_batch)
                    all_y_pred.append(y_pred_batch)
                    all_z_pred.append(z_pred_batch)

                all_x_test = torch.cat(all_x_test, dim=0)
                all_y_test = torch.cat(all_y_test, dim=0)
                all_z_test = torch.cat(all_z_test, dim=0)
                all_xn_test = torch.cat(all_xn_test, dim=0)
                all_yn_test = torch.cat(all_yn_test, dim=0)
                all_zn_test = torch.cat(all_zn_test, dim=0)
                x_pred = torch.cat(all_x_pred, dim=0)
                y_pred = torch.cat(all_y_pred, dim=0)
                z_pred = torch.cat(all_z_pred, dim=0)

                xn_true = all_xn_test.cpu().numpy()
                yn_true = all_yn_test.cpu().numpy()
                zn_true = all_zn_test.cpu().numpy()

                x_pred_np = x_pred.cpu().detach().numpy()
                y_pred_np = y_pred.cpu().detach().numpy()
                z_pred_np = z_pred.cpu().detach().numpy()

                mse_x = mean_squared_error(xn_true, x_pred_np)
                mse_y = mean_squared_error(yn_true, y_pred_np)
                mse_z = mean_squared_error(zn_true, z_pred_np)
                mse_total = (mse_x + mse_y + mse_z) / 3

                r2_x = r2_score(xn_true, x_pred_np)
                r2_y = r2_score(yn_true, y_pred_np)
                r2_z = r2_score(zn_true, z_pred_np)
                r2_total = (r2_x + r2_y + r2_z) / 3

                visualize_results(
                    data_losses, equation_losses, total_losses, val_total_losses,
                    all_xn_test, all_yn_test, all_zn_test,
                    x_pred, y_pred, z_pred,
                    lr, num_steps, pinn_ratio
                )

                final_loss = val_total_losses[-1]  

                results.append({
                    'learning_rate': lr,
                    'num_steps': num_steps,
                    'pinn_ratio': pinn_ratio,
                    'final_loss': final_loss,
                    'mse_x': mse_x,
                    'mse_y': mse_y,
                    'mse_z': mse_z,
                    'mse_total': mse_total,
                    'r2_x': r2_x,
                    'r2_y': r2_y,
                    'r2_z': r2_z,
                    'r2_total': r2_total
                })

                logging.info(f"Learning Rate: {lr}, Num Steps: {num_steps}, PINN Ratio: {pinn_ratio}, Final Loss: {final_loss}, MSE Total: {mse_total}, R2 Total: {r2_total}")

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    best_params = (lr, num_steps, pinn_ratio)
                    best_mse_total = mse_total
                    best_r2_total = r2_total

    best_model = PINN(layer_sizes, rho0, g, device)
    best_model.load_state_dict(best_model_state_dict)

    logging.info(f"Best Parameters: Learning Rate = {best_params[0]}, Num Steps = {best_params[1]}, PINN Ratio = {best_params[2]} with Loss = {best_loss}, MSE = {best_mse_total}, R2 = {best_r2_total}")

    return best_model, best_params, results

if __name__ == "__main__":

    N_INPUT = 4
    N_OUTPUT = 4
    N_HIDDEN = 16
    N_LAYERS = 4
    layer_sizes = [N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS]

    # Define physical constants
    rho0 = 1.225
    g = 9.81

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    t0_data_path = "data/vortex_ring/t0/point_cloud.csv"
    t1_data_path = "data/vortex_ring/t1/point_cloud.csv"

    t0_df = pd.read_csv(t0_data_path)
    t1_df = pd.read_csv(t1_data_path)

    x_data = torch.tensor(t0_df['x'].values, dtype=torch.float32).unsqueeze(1)
    y_data = torch.tensor(t0_df['y'].values, dtype=torch.float32).unsqueeze(1)
    z_data = torch.tensor(t0_df['z'].values, dtype=torch.float32).unsqueeze(1)

    # Centering and normalizing
    mean_x, mean_y, mean_z = x_data.mean(), y_data.mean(), z_data.mean()
    x_data_centered = x_data - mean_x
    y_data_centered = y_data - mean_y
    z_data_centered = z_data - mean_z

    max_radius = torch.sqrt((x_data_centered**2 + y_data_centered**2 + z_data_centered**2).max())

    x_data_normalized = x_data_centered / max_radius
    y_data_normalized = y_data_centered / max_radius
    z_data_normalized = z_data_centered / max_radius

    xn_data = torch.tensor(t1_df['x'].values, dtype=torch.float32).unsqueeze(1)
    yn_data = torch.tensor(t1_df['y'].values, dtype=torch.float32).unsqueeze(1)
    zn_data = torch.tensor(t1_df['z'].values, dtype=torch.float32).unsqueeze(1)

    xn_data_normalized = (xn_data - mean_x) / max_radius
    yn_data_normalized = (yn_data - mean_y) / max_radius
    zn_data_normalized = (zn_data - mean_z) / max_radius

    # Split data into training, validation, and testing sets
    x_train_full, x_temp, y_train_full, y_temp, z_train_full, z_temp, xn_train_full, xn_temp, yn_train_full, yn_temp, zn_train_full, zn_temp = train_test_split(
        x_data_normalized, y_data_normalized, z_data_normalized,
        xn_data_normalized, yn_data_normalized, zn_data_normalized,
        test_size=0.3, random_state=42
    )

    x_val, x_test, y_val, y_test, z_val, z_test, xn_val, xn_test, yn_val, yn_test, zn_val, zn_test = train_test_split(
        x_temp, y_temp, z_temp,
        xn_temp, yn_temp, zn_temp,
        test_size=0.5, random_state=42
    )

    # Create datasets
    train_dataset = VortexRingDataset(x_train_full, y_train_full, z_train_full, xn_train_full, yn_train_full, zn_train_full)
    val_dataset = VortexRingDataset(x_val, y_val, z_val, xn_val, yn_val, zn_val)
    test_dataset = VortexRingDataset(x_test, y_test, z_test, xn_test, yn_test, zn_test)

    # Create DataLoaders
    batch_size = 500
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define grid search parameters
    epochs = 100
    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    num_steps_list = [1, 2, 5, 7, 10]
    pinn_ratios = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
    patience = 10  # Early stopping patience

    # Perform grid search
    best_model, best_params, results = grid_search(
        layer_sizes, rho0, g, device, train_loader, val_loader, epochs, learning_rates, num_steps_list, pinn_ratios, patience
    )

    # Test the best model on the test dataset
    best_lr, best_num_steps, best_pinn_ratio = best_params
    delta_t_test = 1.0 / best_num_steps

    all_x_test, all_y_test, all_z_test = [], [], []
    all_xn_test, all_yn_test, all_zn_test = [], [], []
    all_x_pred, all_y_pred, all_z_pred = [], [], []

    for batch in test_loader:
        x_batch, y_batch, z_batch, xn_batch, yn_batch, zn_batch = [b.to(device) for b in batch]
        t_batch = torch.zeros(len(x_batch), 1, dtype=torch.float32, device=device)

        x_pred_batch, y_pred_batch, z_pred_batch = predict_positions(best_model, x_batch, y_batch, z_batch, t_batch, best_num_steps, delta_t=delta_t_test)

        # Accumulate for visualization
        all_x_test.append(x_batch)
        all_y_test.append(y_batch)
        all_z_test.append(z_batch)
        all_xn_test.append(xn_batch)
        all_yn_test.append(yn_batch)
        all_zn_test.append(zn_batch)

        all_x_pred.append(x_pred_batch)
        all_y_pred.append(y_pred_batch)
        all_z_pred.append(z_pred_batch)

    # Concatenate all test data for visualization
    all_x_test = torch.cat(all_x_test, dim=0)
    all_y_test = torch.cat(all_y_test, dim=0)
    all_z_test = torch.cat(all_z_test, dim=0)
    all_xn_test = torch.cat(all_xn_test, dim=0)
    all_yn_test = torch.cat(all_yn_test, dim=0)
    all_zn_test = torch.cat(all_zn_test, dim=0)
    x_pred = torch.cat(all_x_pred, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)
    z_pred = torch.cat(all_z_pred, dim=0)

    # Visualize results on test data
    visualize_results([], [], [], [], all_xn_test, all_yn_test, all_zn_test, x_pred, y_pred, z_pred, best_lr, best_num_steps, best_pinn_ratio)
