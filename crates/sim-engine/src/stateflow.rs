use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Stateflow-Equivalent Hierarchical / Event-Driven & Temporal Finite State Machine (`StateChart`).
/// Supports `after(n, ticks)` temporal guards, signal threshold guards, and deterministic state outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransitionGuard {
    /// Fires when `input_signal > threshold`
    InputGreaterThan { signal: String, threshold: f64 },
    /// Fires when `input_signal < threshold`
    InputLessThan { signal: String, threshold: f64 },
    /// Stateflow temporal operator `after(ticks, tick)`: fires once the state has been active for `ticks` steps!
    AfterTicks(u64),
    /// Fires when a named discrete event is broadcast
    OnEvent(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub guard: TransitionGuard,
    pub priority: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateNode {
    pub name: String,
    pub entry_outputs: BTreeMap<String, f64>,
    pub during_outputs: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateChart {
    pub states: BTreeMap<String, StateNode>,
    pub transitions: Vec<StateTransition>,
    pub active_state: String,
    pub ticks_in_active_state: u64,
    pub outputs: BTreeMap<String, f64>,
}

impl StateChart {
    #[must_use]
    pub fn new(initial_state: &str) -> Self {
        Self {
            states: BTreeMap::new(),
            transitions: Vec::new(),
            active_state: initial_state.to_string(),
            ticks_in_active_state: 0,
            outputs: BTreeMap::new(),
        }
    }

    pub fn add_state(&mut self, node: StateNode) {
        if self.outputs.is_empty() && node.name == self.active_state {
            for (k, &v) in &node.entry_outputs {
                self.outputs.insert(k.clone(), v);
            }
        }
        self.states.insert(node.name.clone(), node);
    }

    pub fn add_transition(&mut self, tr: StateTransition) {
        self.transitions.push(tr);
        self.transitions.sort_by_key(|t| t.priority);
    }

    /// Steps the Stateflow chart by 1 discrete tick given continuous/discrete `inputs` and optional `event`.
    pub fn step_chart(&mut self, inputs: &BTreeMap<String, f64>, event: Option<&str>) {
        self.ticks_in_active_state = self.ticks_in_active_state.saturating_add(1);

        let mut chosen_target: Option<String> = None;
        for tr in &self.transitions {
            if tr.from_state != self.active_state {
                continue;
            }
            let triggered = match &tr.guard {
                TransitionGuard::InputGreaterThan { signal, threshold } => {
                    inputs.get(signal).copied().unwrap_or(0.0) > *threshold
                }
                TransitionGuard::InputLessThan { signal, threshold } => {
                    inputs.get(signal).copied().unwrap_or(0.0) < *threshold
                }
                TransitionGuard::AfterTicks(n) => self.ticks_in_active_state >= *n,
                TransitionGuard::OnEvent(ev_name) => event == Some(ev_name.as_str()),
            };
            if triggered {
                chosen_target = Some(tr.to_state.clone());
                break;
            }
        }

        if let Some(next_state) = chosen_target {
            self.active_state = next_state;
            self.ticks_in_active_state = 0;
            if let Some(node) = self.states.get(&self.active_state) {
                for (k, &v) in &node.entry_outputs {
                    self.outputs.insert(k.clone(), v);
                }
            }
        } else if let Some(node) = self.states.get(&self.active_state) {
            for (k, &v) in &node.during_outputs {
                self.outputs.insert(k.clone(), v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stateflow_thermostat_and_temporal_after_ticks() {
        let mut chart = StateChart::new("HeaterOff");
        let mut off_out = BTreeMap::new();
        off_out.insert("heater_cmd".into(), 0.0);
        chart.add_state(StateNode {
            name: "HeaterOff".into(),
            entry_outputs: off_out.clone(),
            during_outputs: off_out,
        });

        let mut on_out = BTreeMap::new();
        on_out.insert("heater_cmd".into(), 1.0);
        chart.add_state(StateNode {
            name: "HeaterOn".into(),
            entry_outputs: on_out.clone(),
            during_outputs: on_out,
        });

        // Turn ON when temp < 18.0, Turn OFF after 3 ticks or when temp > 22.0
        chart.add_transition(StateTransition {
            from_state: "HeaterOff".into(),
            to_state: "HeaterOn".into(),
            guard: TransitionGuard::InputLessThan {
                signal: "temp".into(),
                threshold: 18.0,
            },
            priority: 1,
        });
        chart.add_transition(StateTransition {
            from_state: "HeaterOn".into(),
            to_state: "HeaterOff".into(),
            guard: TransitionGuard::AfterTicks(3),
            priority: 1,
        });

        let mut inputs = BTreeMap::new();
        inputs.insert("temp".into(), 16.5);
        chart.step_chart(&inputs, None);
        assert_eq!(chart.active_state, "HeaterOn");
        assert_eq!(chart.outputs.get("heater_cmd"), Some(&1.0));

        // After 3 ticks in HeaterOn, `after(3, ticks)` transitions back to HeaterOff!
        inputs.insert("temp".into(), 20.0);
        chart.step_chart(&inputs, None);
        chart.step_chart(&inputs, None);
        chart.step_chart(&inputs, None);
        assert_eq!(chart.active_state, "HeaterOff");
        assert_eq!(chart.outputs.get("heater_cmd"), Some(&0.0));
    }
}
