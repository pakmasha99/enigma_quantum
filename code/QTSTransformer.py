import torch
import pennylane as qml
from math import log2

# --------------------------------------------------------------------------------
# PennyLane Quantum Circuit
# --------------------------------------------------------------------------------
def sim14_circuit(params, wires, layers=1):
    """
    Implements the 'sim14' circuit from Sim et al. (2019) using PennyLane.
    This function is batch-aware and handles both 1D and 2D parameter tensors.
    """
    is_batched = params.ndim == 2
    param_idx = 0
    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        for i in range(wires - 1, -1, -1): # from last to the first qubit, by 1
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1

# --------------------------------------------------------------------------------
# Helper Functions for LCU and QSVT Simulation
# --------------------------------------------------------------------------------

def apply_unitaries_pl(base_states, unitary_params, qnode_state, coeffs):
    """
    Applies a linear combination of unitaries to a batch of states.
    
    MAJOR PERFORMANCE FIX: This function is now fully vectorized. Instead of
    looping over timesteps, it constructs one large batch of states and parameters
    and calls the QNode only once. This is vastly more efficient.
    """
    bsz, n_timesteps, n_rots = unitary_params.shape
    n_qbs = int(log2(base_states.shape[1]))

    # 1. Flatten the timestep parameters into a single large batch.
    # Shape: (bsz * n_timesteps, n_rots)
    flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)

    # 2. Repeat the base states to match the flattened parameters.
    # Each base state is repeated n_timesteps times.
    # Shape: (bsz * n_timesteps, 2**n_qbs)
    repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)

    # 3. Execute the QNode ONCE with the entire batch.
    # PennyLane will broadcast the operations in a highly parallelized way.
    evolved_states = qnode_state(
        initial_state=repeated_base_states,
        params=flat_params
    )

    # 4. Reshape the results back to include the timestep dimension.
    # Shape: (bsz, n_timesteps, 2**n_qbs)
    evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)

    # --- Ensure both tensors have the same complex dtype before einsum ---
    evolved_states_reshaped = evolved_states_reshaped.to(torch.complex64)
    coeffs = coeffs.to(torch.complex64)

    # 5. Apply the LCU coefficients using efficient tensor multiplication.
    # einsum reads as: for each (b)atch and (t)imestep, sum over the state (i)
    # by multiplying the evolved state vector with its corresponding coefficient.
    lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)
    
    return lcs


def evaluate_polynomial_state_pl(base_states, unitary_params, qnode_state, n_qbs, lcu_coeffs, poly_coeffs):
    """
    Simulates the QSVT polynomial state preparation.
    """
    acc = poly_coeffs[0] * base_states  # accumulator for storing the running sum of the state vector
    working_register = base_states
    for c in poly_coeffs[1:]:
        working_register = apply_unitaries_pl(working_register, unitary_params, qnode_state, lcu_coeffs)
        acc = acc + c * working_register
    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


class QuantumTSTransformer(torch.nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_timesteps: int,
                 degree: int,
                 n_ansatz_layers: int,
                 feature_dim: int,
                 output_dim: int,
                 dropout: float,
                 device,
                 rotation_scale: float = 2 * torch.pi):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device    # highest degree in the polynomial function of QSVT
        self.rotation_scale = rotation_scale

        self.n_rots = 4 * n_qubits * n_ansatz_layers    # number of angle parameters for the main circuit (parameters are unique for each timestep and each subject)
        self.qff_n_rots = 4 * n_qubits * 1  # number of angle parameters for the final classifier circuit (SAME across subjects; applied to time-aggregated states)

        # --- Classical Layers ---
        self.feature_projection = torch.nn.Linear(feature_dim, self.n_rots) # for each time step, convert brain ROIs into angle parameters for the main circuit
        self.dropout = torch.nn.Dropout(dropout)    # to prevent overfitting
        self.rot_sigm = torch.nn.Sigmoid() # to ensure angle parameters are within the [0, 1] range
        self.output_ff = torch.nn.Linear(3 * n_qubits , output_dim)  # final classical classfier, predicts the target variable from the quantum measurements

        # --- Trainable Quantum Parameters ---
        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = torch.nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = torch.nn.Parameter(torch.rand(self.n_timesteps, dtype=torch.complex64))
        self.qff_params = torch.nn.Parameter(torch.rand(self.qff_n_rots))

        # --- PennyLane Device and QNodes ---
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")  # QNode for Timestep State Evolution
        def _timestep_state_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            sim14_circuit(params, wires=self.n_qubits, layers=self.n_ansatz_layers)
            return qml.state()
        self.timestep_state_qnode = _timestep_state_qnode
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")  # QNode for Final State Evolution (with measurements)
        def _qff_qnode_expval(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            sim14_circuit(params, wires=self.n_qubits, layers=1)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]
        self.qff_qnode_expval = _qff_qnode_expval


    def forward(self, x):
        bsz = x.shape[0]

        # Transpose the input tensor to match the expected format.
        # The model expects (batch_size, n_timesteps, feature_dim).
        # Your data is (batch_size, feature_dim, n_timesteps).
        # x.permute(0, 2, 1) swaps the last two dimensions.
        # x = x.permute(0, 2, 1)

        x = self.feature_projection(self.dropout(x))
        timestep_params = self.rot_sigm(x) * self.rotation_scale # squash the angles to [0, 1] then scale to [0, 2pi]
        base_states = torch.zeros(bsz, 2 ** self.n_qubits, dtype=torch.complex64, device=self.device)  # initialize quantum states for each subject in the batch
        base_states[:, 0] = 1.0

        mixed_timestep = evaluate_polynomial_state_pl(   # takes the base_states and angle parameters, and outputs a non-linear, mixed_timestep vector for each subject (after LCU and QSVT)
            base_states,
            timestep_params,
            self.timestep_state_qnode,
            self.n_qubits,
            self.mix_coeffs.repeat(bsz, 1),
            self.poly_coeffs
        )

        # final_probs = torch.linalg.vector_norm(mixed_timestep, dim=-1)
        norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
        normalized_mixed_timestep = mixed_timestep / (norm + 1e-9)  # normalize the mixed_timestep

        exps = self.qff_qnode_expval(   # produce the measurements from normalized_mixed_timestep
            initial_state=normalized_mixed_timestep,
            params=self.qff_params
        )
        
        exps = torch.stack(exps, dim=1)
        exps = exps.float()
        op = self.output_ff(exps)   # predict the final OCD logit using the measurements
        return op.squeeze(1)