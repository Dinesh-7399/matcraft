use crate::{matmul, mldivide, LinalgError};
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;
use tensor::Tensor;

/// Solves the Continuous-Time Lyapunov Equation $A X + X A^T + Q = 0$ (`lyap(A, Q)`)
/// via exact Kronecker vectorization $(I \otimes A + A \otimes I) \operatorname{vec}(X) = -\operatorname{vec}(Q)$.
pub fn lyap_continuous(
    a: &Tensor<f64>,
    q: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() || q.rows() != n || q.cols() != n {
        return Err(LinalgError::NotSquare {
            op: "lyap",
            shape: a.shape(),
        });
    }
    let n2 = n * n;
    let mut kron = Tensor::zeros(n2, n2, budget)?;

    // Build K = (I_n (x) A) + (A (x) I_n)
    for j in 0..n {
        for i in 0..n {
            let a_ij = a.get(i, j)?;
            for k in 0..n {
                // (I (x) A) has block diagonal A
                let row1 = k * n + i;
                let col1 = k * n + j;
                kron.set(row1, col1, kron.get(row1, col1)? + a_ij)?;

                // (A (x) I) has a_ij * I_n at block (i, j)
                let row2 = i * n + k;
                let col2 = j * n + k;
                kron.set(row2, col2, kron.get(row2, col2)? + a_ij)?;
            }
        }
    }

    let q_vec = q.reshape(n2, 1)?.map_unary(SiUnit::DIMENSIONLESS, |x| -x)?;
    let x_vec = mldivide(&kron, &q_vec, budget)?;
    Ok(x_vec.reshape(n, n)?)
}

/// Solves the Discrete-Time Stein/Lyapunov Equation $A X A^T - X + Q = 0$ (`dlyap(A, Q)`)
/// via quadratically convergent Smith Squaring Iteration ($X_{k+1} = X_k + A_k X_k A_k^T$, $A_{k+1} = A_k^2$).
pub fn dlyap_discrete(
    a: &Tensor<f64>,
    q: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() || q.rows() != n || q.cols() != n {
        return Err(LinalgError::NotSquare {
            op: "dlyap",
            shape: a.shape(),
        });
    }
    let mut x = q.clone();
    let mut ak = a.clone();
    for _iter in 0..40 {
        let ak_t = ak.transpose()?;
        let term = matmul(&matmul(&ak, &x, budget)?, &ak_t, budget)?;
        let diff = term.norm_frobenius()?;
        x = x.checked_add(&term)?;
        if diff < 1e-13 {
            break;
        }
        ak = matmul(&ak, &ak, budget)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_and_discrete_lyapunov_solvers() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // Stable A = [-2, 1; 0, -3], Q = I_2
        let a = Tensor::from_column_major(
            2,
            2,
            vec![-2.0, 0.0, 1.0, -3.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let q = Tensor::eye(2, &budget)?;
        let x = lyap_continuous(&a, &q, &budget)?;

        // Verify residual ||A*X + X*A^T + Q||_F < 1e-12
        let ax = matmul(&a, &x, &budget)?;
        let xat = matmul(&x, &a.transpose()?, &budget)?;
        let res = ax.checked_add(&xat)?.checked_add(&q)?;
        assert!(res.norm_frobenius()? < 1e-12);

        // Discrete Lyapunov with Schur-stable A_d = [0.5, 0.1; 0.0, 0.4]
        let ad = Tensor::from_column_major(
            2,
            2,
            vec![0.5, 0.0, 0.1, 0.4],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let xd = dlyap_discrete(&ad, &q, &budget)?;
        let res_d = matmul(&matmul(&ad, &xd, &budget)?, &ad.transpose()?, &budget)?
            .checked_sub(&xd)?
            .checked_add(&q)?;
        assert!(res_d.norm_frobenius()? < 1e-11);
        Ok(())
    }
}
