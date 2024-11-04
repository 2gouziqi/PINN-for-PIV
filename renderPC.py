import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load point cloud data from CSV
point_cloud_path = "data/vortex_ring/t1/point_cloud.csv"
point_cloud_df = pd.read_csv(point_cloud_path)

# Extract x, y, z coordinates and convert to tensors
x_data = torch.tensor(point_cloud_df['x'].values, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(point_cloud_df['y'].values, dtype=torch.float32).unsqueeze(1)
z_data = torch.tensor(point_cloud_df['z'].values, dtype=torch.float32).unsqueeze(1)

# Combine into a single point cloud tensor
point_cloud = torch.cat((x_data, y_data, z_data), dim=1)


# Basic Point Cloud class
class BasicPointCloud:
    def __init__(self, points):
        self.points = points

# Gaussian Splatting Model class
class GaussianSplattingModel:
    def __init__(self):
        self.xyz = None

    def create_from_point_cloud(self, pcd):
        # Convert the point cloud to a parameter tensor
        self.xyz = torch.nn.Parameter(pcd.points.clone().detach().requires_grad_(True))
        print("Number of points in the model:", self.xyz.shape[0])

# Usage
pcd = BasicPointCloud(points=point_cloud)
gaussian_model = GaussianSplattingModel()
gaussian_model.create_from_point_cloud(pcd)


# Camera class for projection
class Camera:
    def __init__(self, transform_matrix, camera_angle_x, image_width, image_height):
        self.transform_matrix = np.array(transform_matrix)
        self.camera_angle_x = camera_angle_x
        self.image_width = image_width
        self.image_height = image_height

        # Compute intrinsic camera parameters
        self.focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        self.intrinsic_matrix = np.array([
            [self.focal_length, 0, image_width / 2],
            [0, self.focal_length, image_height / 2],
            [0, 0, 1]
        ])

    def project(self, points):
        # Convert points to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # Apply the camera transformation
        camera_coords = points_homogeneous @ self.transform_matrix.T

        # Perspective division
        camera_coords = camera_coords[:, :3] / camera_coords[:, 3][:, np.newaxis]

        # Project using intrinsic matrix
        pixel_coords_homogeneous = camera_coords @ self.intrinsic_matrix.T

        # Normalize homogeneous coordinates
        pixel_coords = pixel_coords_homogeneous[:, :2] / pixel_coords_homogeneous[:, 2][:, np.newaxis]

        return pixel_coords

class Renderer:
    def __init__(self, gaussian_model, camera):
        self.gaussian_model = gaussian_model
        self.camera = camera

    def render(self):
        # Get the 3D points from the Gaussian model
        points_3d = self.gaussian_model.xyz.detach().cpu().numpy()

        # Project points to 2D using the camera
        pixel_coords = self.camera.project(points_3d)

        # Round the coordinates and convert to integers
        x_coords = np.round(pixel_coords[:, 0]).astype(int)
        y_coords = np.round(pixel_coords[:, 1]).astype(int)

        # Filter out points outside the image boundaries
        valid_mask = (
            (x_coords >= 0) & (x_coords < self.camera.image_width) &
            (y_coords >= 0) & (y_coords < self.camera.image_height)
        )
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Create an empty image
        image = np.zeros((self.camera.image_height, self.camera.image_width, 3), dtype=np.uint8)

        # Set the pixel values to white where points are projected
        image[y_coords, x_coords] = [255, 255, 255]

        return image

    def save_image(self, image, output_path="output/rendered_image.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.imsave(output_path, image)
        print(f"Rendered image saved at: {output_path}")

# Camera parameters from your data
camera_params = {
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [-0.9244644045829773, -0.2542489469051361, 0.28411802649497986, 1.1453163623809814],
                [0.38126838207244873, -0.6164793968200684, 0.6889031529426575, 2.777057409286499],
                [1.4901161193847656e-08, 0.7451916933059692, 0.6668503284454346, 2.688159704208374],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    ]
}

# Extract camera parameters
camera_angle_x = camera_params['camera_angle_x']
transform_matrix = camera_params['frames'][0]['transform_matrix']

# Image dimensions
image_width = 768
image_height = 768

# Initialize the camera
camera = Camera(transform_matrix, camera_angle_x, image_width, image_height)

# Initialize the renderer
renderer = Renderer(gaussian_model, camera)

# Render the image
rendered_image = renderer.render()

# Save the image
renderer.save_image(rendered_image, "output/rendered_image.png")
