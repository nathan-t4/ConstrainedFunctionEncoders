from typing import Tuple, Dict
import torch
from FunctionEncoder.Dataset.BaseDataset import BaseDataset

class MassSpringDamperDataset(BaseDataset):

    def __init__(self,
                 mass_range=(0.5, 1.0),
                 spring_constant_range=(0.1, 1.0),
                 damping_coefficient_range=(0.1, 1.0),
                 force_range=(-5.0, 5.0),
                 initial_position_range=(-0.5, 0.5),  # Range for initial position y(0)
                 initial_velocity_range=(-0.5, 0.5),  # Range for initial velocity y_dot(0)
                 dt_range: tuple = (1e-3, 1e-2),  # Time steps
                 n_parameters_per_sample: int = 10,
                 n_examples_per_sample: int = 1_000, # 1_000
                 n_points_per_sample: int = 10_000, # 10_000
                 horizon_timesteps: int = 1,
                 control_type: str = "sinusoidal",
                 residuals: bool = False,
                 device: str = "auto"):
        super().__init__(input_size=(3,),  # time, position, force # TODO: position, velocity
                         output_size=(2,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="deterministic",
                         n_functions_per_sample=n_parameters_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         device=device)
        self.mass_range = torch.tensor(mass_range, device=self.device)
        self.spring_constant_range = torch.tensor(spring_constant_range, device=self.device)
        self.damping_coefficient_range = torch.tensor(damping_coefficient_range, device=self.device)
        self.force_range = torch.tensor(force_range, device=self.device)
        self.initial_position_range = torch.tensor(initial_position_range, device=self.device)
        self.initial_velocity_range = torch.tensor(initial_velocity_range, device=self.device)
        self.dt_range = dt_range  
        self.horizon_timesteps = horizon_timesteps
        self.control_type = control_type
        self.residuals = residuals

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            control_type = self.control_type
            n_parameters = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # Generate parameter sets for mass, spring constant, and damping coefficient
            masses = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
            spring_constants = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.spring_constant_range[1] - self.spring_constant_range[0]) + self.spring_constant_range[0]
            damping_coefficients = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.damping_coefficient_range[1] - self.damping_coefficient_range[0]) + self.damping_coefficient_range[0]

            def integrate_dynamics(n_parameters, n_examples, initial_pos, initial_vel, forces, dt):
                xs = torch.zeros((n_parameters, n_examples, 2), dtype=torch.float32, device=self.device)
                ys = torch.zeros((n_parameters, n_examples, *self.output_size), dtype=torch.float32, device=self.device)

                for i in range(n_parameters):
                    mass = masses[i]
                    k = spring_constants[i]
                    c = damping_coefficients[i]
                    F = forces[i]
                    
                    y = torch.zeros((F.shape[0]+1,F.shape[1]), dtype=torch.float32, device=self.device)
                    y_dot = torch.zeros((F.shape[0]+1,F.shape[1]), dtype=torch.float32, device=self.device)

                    # Set initial conditions
                    y[0] = initial_pos[i]
                    y_dot[0] = initial_vel[i]
                    
                    # Euler integration of ODEs
                    for t in range(1,n_examples):
                        y_ddot = (F[t-1] - c * y_dot[t-1] - k * y[t-1]) / mass
                        y_dot[t] = y_dot[t-1] + y_ddot * dt[i]
                        y[t] = y[t-1] + y_dot[t] * dt[i]
                    # Predict one time step ahead
                    xs[i] = torch.cat((y[:-1], y_dot[:-1]), dim=-1)
                    ys[i] = torch.cat((y[1:], y_dot[1:]), dim=-1)

                    # if self.residuals:
                    #     # Calculate dx
                    #     xs[i] = torch.cat((y[1:] - y[:-1], y_dot[1:] - y_dot[:-1]), dim=-1)
                
                return xs, ys
            
            def get_force(t):
                mag = torch.rand(1, dtype=torch.float32, device=self.device) * (self.force_range[1] - self.force_range[0]) + self.force_range[0]
                omega = torch.rand(1, dtype=torch.float32, device=self.device)
                return mag * torch.sin(omega * t)
                            
            # Randomize initial conditions (y(0) and y_dot(0)) within the specified ranges
            initial_positions = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.initial_position_range[1] - self.initial_position_range[0]) + self.initial_position_range[0]
            initial_velocities = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.initial_velocity_range[1] - self.initial_velocity_range[0]) + self.initial_velocity_range[0]
            
            # Generate time points starting at 0 (with random dt)
            dt = torch.rand(n_parameters, dtype=torch.float32, device=self.device) * (self.dt_range[1] - self.dt_range[0]) + self.dt_range[0]
            time_points = torch.cat([torch.arange(0, n_points * dt[k], dt[k], device=self.device)[:n_points].reshape(1,-1,1) for k in range(n_parameters)], dim=0)

            if control_type == "sinusoidal":
                # Generate random continuous forces
                forces = get_force(time_points)   
            elif control_type == "constant":
                # Generate random constant forces
                forces = torch.rand((n_parameters, 1, 1), dtype=torch.float32, device=self.device) * (self.force_range[1] - self.force_range[0]) + self.force_range[0]
                forces = forces.repeat(1, n_points, 1)
            
            # Compute the system's response based on the mass-spring-damper equation
            xs = torch.zeros((n_parameters, n_points, 1), dtype=torch.float32, device=self.device)
            ys = torch.zeros((n_parameters, n_points, *self.output_size), dtype=torch.float32, device=self.device)
            xs, ys = integrate_dynamics(n_parameters, n_points, initial_positions, initial_velocities, forces, dt)

            # Randomize the example initial conditions (y(0) and y_dot(0)) within the specified ranges
            example_initial_positions = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.initial_position_range[1] - self.initial_position_range[0]) + self.initial_position_range[0]
            example_initial_velocities = torch.rand((n_parameters, 1), dtype=torch.float32, device=self.device) * (self.initial_velocity_range[1] - self.initial_velocity_range[0]) + self.initial_velocity_range[0]
            
            # Generate example time starting at 0
            example_dt = torch.rand(n_parameters, dtype=torch.float32, device=self.device) * (self.dt_range[1] - self.dt_range[0]) + self.dt_range[0]
            example_time_points = torch.cat([torch.arange(0, n_examples * example_dt[k], example_dt[k], device=self.device)[:n_examples].reshape(1,-1,1) for k in range(n_parameters)], dim=0)


            if control_type == "sinusoidal":
                # Generate random continuous forces
                example_forces = get_force(example_time_points)          
            elif control_type == "constant":
                # Generate random constant forces
                example_forces_mag = torch.rand((n_parameters, 1, 1), dtype=torch.float32, device=self.device) * (self.force_range[1] - self.force_range[0]) + self.force_range[0]
                example_forces = example_forces_mag.repeat(1, n_examples, 1)
            
            # Compute the system's response based on the mass-spring-damper equation
            example_xs, example_ys = integrate_dynamics(n_parameters, n_examples, example_initial_positions, example_initial_velocities, example_forces, example_dt)

            dataset_cfg = {
                "masses": masses, 
                "spring_constants": spring_constants, 
                "damping_coefficients": damping_coefficients, 
                "forces_magnitude": example_forces[:,0,:].squeeze() if control_type == "sinusoidal" else example_forces_mag.squeeze(),
                "initial_position_range": self.initial_position_range, 
                "initial_velocity_range": self.initial_velocity_range,
                "initial_positions": initial_positions, 
                "initial_velocities": initial_velocities,
                "dt": dt,
            }

            # Inputs are [position(k), k, control]. Outputs are [position(k+1)]
            # xs = torch.cat([time_points, xs, forces[:,:-1]], dim=-1)
            # example_xs = torch.cat([example_time_points, example_xs, example_forces[:,:-1]], dim=-1)

            # Inputs are [dt, x]
            dts = torch.tile(dt, (1,n_points,1)).transpose(2,0)
            xs = torch.cat([dts, xs], dim=-1)
            example_dts = torch.tile(example_dt, (1,n_examples,1)).transpose(2,0)
            example_xs = torch.cat([example_dts, example_xs], dim=-1)
            
            return example_xs, example_ys, xs, ys, dataset_cfg