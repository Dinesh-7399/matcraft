use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AllocError {
    #[error("Integer overflow computing element count for shape {shape:?}")]
    DimensionOverflow { shape: Vec<usize> },
    #[error("Byte count overflow for {elements} elements of {elem_bytes} bytes each")]
    ByteSizeOverflow { elements: usize, elem_bytes: usize },
    #[error("Memory budget exceeded: requested {requested} bytes, but only {available} bytes remain of {limit} byte workspace limit. Hint: use sparse matrices orTiled Uniform tensors.")]
    BudgetExceeded {
        requested: usize,
        available: usize,
        limit: usize,
    },
}

/// Thread-safe workspace memory budget tracker.
/// Rejects hostile or runaway allocations *before* invoking the system allocator.
#[derive(Debug)]
pub struct MemoryBudget {
    allocated_bytes: AtomicUsize,
    max_bytes: usize,
}

impl MemoryBudget {
    /// Default 2 GiB workspace budget if unspecified.
    pub const DEFAULT_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

    #[must_use]
    pub fn new(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            allocated_bytes: AtomicUsize::new(0),
            max_bytes,
        })
    }

    #[must_use]
    pub fn default_budget() -> Arc<Self> {
        Self::new(Self::DEFAULT_LIMIT_BYTES)
    }

    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    #[must_use]
    pub fn available_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.used_bytes())
    }

    /// Checked product of dimensions with zero-panic overflow protection.
    pub fn checked_element_count(shape: &[usize]) -> Result<usize, AllocError> {
        if shape.is_empty() {
            return Ok(0);
        }
        let mut total: usize = 1;
        for &dim in shape {
            total = total
                .checked_mul(dim)
                .ok_or_else(|| AllocError::DimensionOverflow {
                    shape: shape.to_vec(),
                })?;
        }
        Ok(total)
    }

    /// Attempts to reserve `elements * elem_bytes` from this budget using CAS loop.
    pub fn try_reserve_elements(
        self: &Arc<Self>,
        elements: usize,
        elem_bytes: usize,
    ) -> Result<AllocReservation, AllocError> {
        let requested = elements
            .checked_mul(elem_bytes)
            .ok_or(AllocError::ByteSizeOverflow {
                elements,
                elem_bytes,
            })?;
        self.try_reserve_bytes(requested)
    }

    pub fn try_reserve_bytes(
        self: &Arc<Self>,
        requested: usize,
    ) -> Result<AllocReservation, AllocError> {
        if requested == 0 {
            return Ok(AllocReservation {
                budget: Arc::clone(self),
                bytes: 0,
            });
        }
        let mut current = self.allocated_bytes.load(Ordering::Relaxed);
        loop {
            let available = self.max_bytes.saturating_sub(current);
            if requested > available {
                return Err(AllocError::BudgetExceeded {
                    requested,
                    available,
                    limit: self.max_bytes,
                });
            }
            let next = current.saturating_add(requested);
            match self.allocated_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(AllocReservation {
                        budget: Arc::clone(self),
                        bytes: requested,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }
}

/// RAII token that automatically returns reserved bytes to the `MemoryBudget` when dropped.
#[derive(Debug)]
pub struct AllocReservation {
    budget: Arc<MemoryBudget>,
    bytes: usize,
}

impl AllocReservation {
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for AllocReservation {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.budget
                .allocated_bytes
                .fetch_sub(self.bytes, Ordering::Release);
        }
    }
}
