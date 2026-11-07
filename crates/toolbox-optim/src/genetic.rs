#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]

//! Real-coded genetic algorithm for minimization.

use crate::OptimError;

/// Options for the genetic algorithm.
#[derive(Clone, Debug)]
pub struct GaOptions {
    /// Population size.
    pub pop_size: usize,
    /// Maximum number of generations.
    pub max_gen: usize,
    /// Crossover rate (0.0–1.0).
    pub crossover_rate: f64,
    /// Mutation rate (0.0–1.0).
    pub mutation_rate: f64,
    /// Mutation scale: uniform ±mutation_scale perturbation.
    pub mutation_scale: f64,
}

/// Result of the genetic algorithm.
#[derive(Clone, Debug)]
pub struct GaResult {
    pub best_x: Vec<f64>,
    pub best_fitness: f64,
    pub generations: usize,
}

/// Deterministic LCG RNG state.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as f64 / u32::MAX as f64
    }

    /// Returns a float in [lo, hi).
    fn next_in_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

/// Minimize `f` over `bounds` using a real-coded genetic algorithm.
///
/// Uses a deterministic LCG for all randomness (no `rand` crate).
pub fn ga_minimize(
    f: impl Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    opts: GaOptions,
    seed: u64,
) -> Result<GaResult, OptimError> {
    let dim = bounds.len();
    if dim == 0 {
        return Err(OptimError::InvalidDimension);
    }
    if opts.pop_size < 2 {
        return Err(OptimError::InvalidDimension);
    }

    let mut rng = Lcg::new(seed);

    // Initialize population with random individuals within bounds
    let mut population: Vec<Vec<f64>> = (0..opts.pop_size)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.next_in_range(lo, hi))
                .collect()
        })
        .collect();

    // Evaluate fitness for entire population
    let mut fitness: Vec<f64> = population.iter().map(|ind| f(ind)).collect();

    // Find initial best
    let (best_idx, mut best_fitness) =
        fitness
            .iter()
            .enumerate()
            .fold((0usize, f64::INFINITY), |(bi, bf), (i, &fi)| {
                if fi < bf { (i, fi) } else { (bi, bf) }
            });

    let mut best_x = population.get(best_idx).cloned().unwrap_or_default();

    for gen in 0..opts.max_gen {
        // Tournament selection + SBX crossover + mutation
        let mut new_population: Vec<Vec<f64>> = Vec::with_capacity(opts.pop_size);
        let mut new_fitness: Vec<f64> = Vec::with_capacity(opts.pop_size);

        // Elitism: keep best individual
        new_population.push(best_x.clone());
        new_fitness.push(best_fitness);

        while new_population.len() < opts.pop_size {
            // Tournament selection for parent 1
            let p1_idx = tournament_select(&fitness, &mut rng);
            // Tournament selection for parent 2 (different from p1)
            let p2_idx = {
                let mut idx = tournament_select(&fitness, &mut rng);
                // Try once more if same
                if idx == p1_idx {
                    idx = tournament_select(&fitness, &mut rng);
                }
                idx
            };

            let parent1 = population.get(p1_idx).cloned().unwrap_or_default();
            let parent2 = population.get(p2_idx).cloned().unwrap_or_default();

            // Crossover
            let child = if rng.next_f64() < opts.crossover_rate {
                blx_alpha_crossover(&parent1, &parent2, 0.5, &mut rng)
            } else {
                parent1.clone()
            };

            // Mutation
            let child = mutate(child, bounds, opts.mutation_rate, opts.mutation_scale, &mut rng);

            // Clamp to bounds
            let child: Vec<f64> = child
                .iter()
                .zip(bounds.iter())
                .map(|(&x, &(lo, hi))| x.clamp(lo, hi))
                .collect();

            let child_fitness = f(&child);
            new_population.push(child);
            new_fitness.push(child_fitness);
        }

        population = new_population;
        fitness = new_fitness;

        // Update best
        for (i, &fi) in fitness.iter().enumerate() {
            if fi < best_fitness {
                best_fitness = fi;
                best_x = population.get(i).cloned().unwrap_or_default();
            }
        }

        // Early convergence check
        if best_fitness.abs() < 1e-14 {
            return Ok(GaResult {
                best_x,
                best_fitness,
                generations: gen + 1,
            });
        }
    }

    Ok(GaResult {
        best_x,
        best_fitness,
        generations: opts.max_gen,
    })
}

/// Binary tournament selection: pick 2 random indices, return the better one.
fn tournament_select(fitness: &[f64], rng: &mut Lcg) -> usize {
    let n = fitness.len();
    if n == 0 {
        return 0;
    }
    let i = (rng.next_f64() * n as f64) as usize % n;
    let j = (rng.next_f64() * n as f64) as usize % n;
    let fi = fitness.get(i).copied().unwrap_or(f64::INFINITY);
    let fj = fitness.get(j).copied().unwrap_or(f64::INFINITY);
    if fi <= fj { i } else { j }
}

/// BLX-α crossover: child = parent1 + α * (parent2 - parent1) + U[-α, α] * |parent2 - parent1|
fn blx_alpha_crossover(p1: &[f64], p2: &[f64], alpha: f64, rng: &mut Lcg) -> Vec<f64> {
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| {
            let lo = a.min(b) - alpha * (a - b).abs();
            let hi = a.max(b) + alpha * (a - b).abs();
            rng.next_in_range(lo, hi)
        })
        .collect()
}

/// Uniform mutation: each gene mutated with probability `rate` by ±scale.
fn mutate(
    mut ind: Vec<f64>,
    bounds: &[(f64, f64)],
    rate: f64,
    scale: f64,
    rng: &mut Lcg,
) -> Vec<f64> {
    for (i, x) in ind.iter_mut().enumerate() {
        if rng.next_f64() < rate {
            let delta = rng.next_in_range(-scale, scale);
            *x += delta;
            // Clamp to bounds
            if let Some(&(lo, hi)) = bounds.get(i) {
                *x = x.clamp(lo, hi);
            }
        }
    }
    ind
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_minimize_sphere() -> Result<(), Box<dyn std::error::Error>> {
        // Minimize f(x, y) = x² + y², minimum at (0, 0)
        let f = |v: &[f64]| -> f64 {
            v.iter().map(|&x| x * x).sum()
        };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GaOptions {
            pop_size: 50,
            max_gen: 200,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            mutation_scale: 0.5,
        };
        let result = ga_minimize(f, &bounds, opts, 42)?;
        assert!(result.best_fitness < 0.5, "GA should find near-zero, got {}", result.best_fitness);
        Ok(())
    }

    #[test]
    fn test_ga_minimize_1d() -> Result<(), Box<dyn std::error::Error>> {
        // f(x) = (x - 3)², minimum at x = 3
        let f = |v: &[f64]| -> f64 {
            let x = v.first().copied().unwrap_or(0.0);
            (x - 3.0) * (x - 3.0)
        };
        let bounds = vec![(0.0, 10.0)];
        let opts = GaOptions {
            pop_size: 30,
            max_gen: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.15,
            mutation_scale: 0.3,
        };
        let result = ga_minimize(f, &bounds, opts, 123)?;
        let best_x = result.best_x.first().copied().unwrap_or(0.0);
        assert!((best_x - 3.0).abs() < 1.0, "Best x should be near 3.0, got {best_x}");
        Ok(())
    }

    #[test]
    fn test_ga_empty_bounds_returns_error() {
        let f = |_: &[f64]| 0.0;
        let result = ga_minimize(f, &[], GaOptions {
            pop_size: 10,
            max_gen: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.05,
            mutation_scale: 0.1,
        }, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lcg_determinism() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_f64().to_bits(), rng2.next_f64().to_bits());
        }
    }
}
