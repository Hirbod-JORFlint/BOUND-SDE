# BOUND-SDE
a unified framework for constrained trait evolution on phylogenetic trees
\section*{Overview}

\textbf{BOUND-SDE} is a high-performance computational framework for modeling trait evolution on phylogenetic trees under stochastic differential equations (SDEs) with hard constraints. The system is implemented in a fully vectorized, differentiable manner using JAX, enabling efficient likelihood computation, gradient-based parameter estimation, and forward simulation.

The framework supports:
\begin{itemize}
    \item Reflecting boundary conditions on compact intervals $[L,U]$.
    \item Diffusions on Riemannian manifolds, specifically:
    \begin{itemize}
        \item The circle $S^1$.
        \item The probability simplex $\Delta^d$.
    \end{itemize}
    \item Spectral transition approximations for efficient likelihood computation.
    \item A Boundary-Propagating Pruning (BPP) algorithm for recursive tree likelihood evaluation.
\end{itemize}

\section*{Mathematical Formulation}

\subsection*{Trait Evolution as an SDE}

Trait evolution along each branch is modeled as an It\^o SDE:
\begin{equation}
dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t,
\end{equation}
where:
\begin{itemize}
    \item $\mu(x)$ is the drift,
    \item $\sigma(x)$ is the diffusion coefficient,
    \item $W_t$ is standard Brownian motion.
\end{itemize}

The infinitesimal generator $\mathcal{L}$ of the process is
\begin{equation}
\mathcal{L}f(x)
=
\mu(x) \nabla f(x)
+
\frac{1}{2} \sigma(x)^2 \Delta f(x).
\end{equation}

\subsection*{Reflecting Boundaries on $[L,U]$}

For compact domains $[L,U]$, reflecting boundary conditions impose Neumann constraints:
\begin{equation}
\left. \frac{\partial f}{\partial n} \right|_{x=L}
=
\left. \frac{\partial f}{\partial n} \right|_{x=U}
=
0.
\end{equation}

This ensures that probability mass is conserved and trajectories are reflected rather than absorbed at the boundary.

\subsection*{Diffusions on Riemannian Manifolds}

On a Riemannian manifold $(\mathcal{M}, g)$, the generator generalizes to:
\begin{equation}
\mathcal{L} f
=
\frac{1}{2} \Delta_g f
+
\langle b, \nabla f \rangle_g,
\end{equation}
where:
\begin{itemize}
    \item $\Delta_g$ is the Laplace--Beltrami operator,
    \item $\langle \cdot, \cdot \rangle_g$ denotes the Riemannian metric,
    \item $b$ is the drift vector field.
\end{itemize}

\paragraph{Circle $S^1$.}
The Laplace--Beltrami operator reduces to:
\begin{equation}
\Delta_{S^1} f(\theta)
=
\frac{\partial^2 f}{\partial \theta^2}.
\end{equation}

\paragraph{Simplex $\Delta^d$.}
The geometry is induced by the Fisher--Rao metric, yielding nontrivial curvature and requiring intrinsic gradient and divergence operators.

\section*{Spectral Transition Approximation}

Transition densities are approximated using spectral decomposition:
\begin{equation}
p_t(x,y)
\approx
\sum_{k=0}^{K}
e^{-\lambda_k t}
\phi_k(x)\phi_k(y),
\end{equation}
where $\{(\lambda_k, \phi_k)\}$ are eigenpairs of $-\mathcal{L}$ satisfying the appropriate boundary or geometric constraints:
\begin{equation}
-\mathcal{L} \phi_k = \lambda_k \phi_k.
\end{equation}

This representation enables efficient and numerically stable likelihood computation on trees.

\section*{Boundary-Propagating Pruning (BPP)}

Given a phylogenetic tree $\mathcal{T}$ with $N$ nodes, the likelihood is computed via recursive message passing.

For each node $i$:
\begin{equation}
\mathcal{L}_i(x)
=
\prod_{j \in \text{children}(i)}
\int
p_{t_{ij}}(x,y)
\mathcal{L}_j(y)
\, dy.
\end{equation}

The total tree likelihood is:
\begin{equation}
\mathcal{L}_{\text{tree}}
=
\int
\pi(x_{\text{root}})
\mathcal{L}_{\text{root}}(x_{\text{root}})
\, dx_{\text{root}},
\end{equation}
where $\pi$ is the root distribution.

The BPP algorithm performs this recursion in $O(N)$ time using vectorized JAX primitives.

\section*{Architecture}

The system is modularized into the following components:

\begin{enumerate}[label=\arabic*.]
    \item \texttt{kernels.py} --- infinitesimal generators and spectral bases.
    \item \texttt{manifolds.py} --- Riemannian geometry definitions.
    \item \texttt{tree\_ops.py} --- JAX-compatible tree traversal.
    \item \texttt{spectral\_solver.py} --- eigenvalue and eigenfunction computation.
    \item \texttt{pruning.py} --- Boundary-Propagating Pruning implementation.
    \item \texttt{likelihood.py} --- log-likelihood wrappers.
    \item \texttt{optimizers.py} --- gradient-based parameter estimation.
    \item \texttt{simulations.py} --- forward simulation engine.
    \item \texttt{bridge\_r.py} --- R interface layer.
    \item \texttt{main.py} --- execution and validation pipeline.
\end{enumerate}

\section*{Computational Design Principles}

\begin{itemize}
    \item All functions are pure and JIT-compatible.
    \item Tree traversals use vectorized scans for $O(N)$ complexity.
    \item Automatic differentiation enables gradient-based optimization.
    \item Spectral truncation provides controlled approximation accuracy.
\end{itemize}

\section*{Validation}

Validation includes:
\begin{itemize}
    \item Convergence to classical Brownian and Ornstein--Uhlenbeck limits.
    \item Simulation-based calibration.
    \item Stability tests near reflecting boundaries and high curvature regions.
\end{itemize}

\section*{Conclusion}

BOUND-SDE provides a unified, differentiable, and geometrically principled framework for constrained trait evolution on phylogenetic trees. By combining spectral theory, Riemannian geometry, and scalable JAX-based computation, the system enables statistically rigorous and computationally efficient phylogenetic inference under boundary-aware stochastic processes.
