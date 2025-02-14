The general nonlinear optimization problem is expressed with Equation~\eqref{eq:opt_eq}, where \(f(x): \mathbb{R}^n \to \mathbb{R}\) is the objective function in the minimization direction; \(g(x): \mathbb{R}^n \to \mathbb{R}^p\) and \(h(x): \mathbb{R}^n \to \mathbb{R}^m\) are inequality and equality constraints, respectively.

\begin{equation}
    \min_{x} f(x)
    \label{eq:opt_eq}
\end{equation}
\begin{gather*}
    \text{s.t.} \\
    h(x) = 0 \\
    g(x) \leq 0
\end{gather*}
