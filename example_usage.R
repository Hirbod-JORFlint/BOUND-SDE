# Example of using BOUND-SDE back and forth with R
# This script demonstrates generating a tree and trait in R,
# then using BOUND-SDE via reticulate to fit a model or simulate.

# Load required R packages
library(ape)  # for phylogenetic trees
library(reticulate)  # for Python interface

# Set Python path if needed (adjust to your Python environment with BOUND-SDE)
# use_python("path/to/python")  # Uncomment and set if necessary

# Source the bridge_r.py to access Python functions
source_python("bridge_r.py")

# Step 1: Generate tree and trait data (as per user)
set.seed(123)  # For reproducibility
tree <- pbtree(n = 20)
trait_data <- fastBM(tree)

# Step 2: Convert R tree to JAX arrays
tree_arrays <- r_tree_to_arrays(tree$edge, tree$edge.length)
parents <- tree_arrays$parents
branch_lengths <- tree_arrays$branch_lengths

# Step 3: Convert trait data to JAX array
traits_jax <- r_traits_to_jax(trait_data)

# Now, to demonstrate, let's assume we want to fit a model on a circle manifold
# First, we need to import manifolds and other modules
# Since source_python loaded bridge_r.py, we can access imported modules via py$

# Create a circle manifold (for bounded traits on a circle)
manifold <- py$create_circle_manifold()

# Spectral dimension (number of basis functions)
spectral_dim <- 10L

# For fitting, we need to define node loglik, but for simplicity, let's simulate new traits
# Define drift and diffusion functions (constant for simplicity)
drift_fn <- py$`_constant_drift`(0.1)  # Example drift
diffusion_fn <- py$`_constant_diffusion`(0.5)  # Example diffusion

# Get topological order (preorder)
topo_order <- r_get_preorder(parents)

# Root state
root_state <- c(0.0)  # Assuming 1D trait

# Simulate traits
key_seed <- 42L
dt <- 0.01

simulated_traits <- r_simulate_traits(
  key_seed,
  root_state,
  parents,
  branch_lengths,
  topo_order,
  drift_fn,
  diffusion_fn,
  manifold,
  dt
)

# Convert back to R
simulated_traits_r <- as.matrix(simulated_traits)

# Now, you can plot or analyze in R
print("Simulated traits shape:")
print(dim(simulated_traits_r))

# For fitting, it's more complex, but this shows back and forth
# To fit, you would need to define node_loglik based on data, then optimize params using r_compute_likelihood
