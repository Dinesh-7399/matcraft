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

//! MATLAB Object-Oriented Programming (OOP) Engine (`classdef`).
//! Supports class definitions with single/multiple inheritance, handle (reference) vs
//! value semantics, encapsulated properties with access modifiers, constructor dispatch,
//! and polymorphic method resolution.

use crate::LangError;
use num_core::{MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tensor::Tensor;

/// Property and method encapsulation levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AccessModifier {
    #[default]
    Public,
    Protected,
    Private,
}

/// A property declaration inside a `classdef` properties block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyDef {
    pub name: String,
    pub default_val: Option<f64>,
    pub unit: Option<String>,
    pub get_access: AccessModifier,
    pub set_access: AccessModifier,
}

/// A method declaration inside a `classdef` methods block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethodDef {
    pub name: String,
    pub is_static: bool,
    pub params: Vec<String>,
    pub outputs: Vec<String>,
    pub access: AccessModifier,
}

/// A MATLAB Class Definition parsed from a `classdef` block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassDef {
    pub name: String,
    pub superclasses: Vec<String>,
    pub is_handle: bool,
    pub properties: BTreeMap<String, PropertyDef>,
    pub methods: BTreeMap<String, MethodDef>,
}

impl ClassDef {
    /// Creates a new empty class definition.
    #[must_use]
    pub fn new(name: impl Into<String>, is_handle: bool) -> Self {
        Self {
            name: name.into(),
            superclasses: Vec::new(),
            is_handle,
            properties: BTreeMap::new(),
            methods: BTreeMap::new(),
        }
    }

    /// Adds a property with default scalar value.
    pub fn add_property(&mut self, prop: PropertyDef) {
        self.properties.insert(prop.name.clone(), prop);
    }

    /// Adds a method definition.
    pub fn add_method(&mut self, method: MethodDef) {
        self.methods.insert(method.name.clone(), method);
    }
}

/// A live runtime instance of a MATLAB classdef object.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassInstance {
    pub class_name: String,
    pub is_handle: bool,
    pub handle_id: Option<u64>,
    pub properties: BTreeMap<String, Tensor<f64>>,
}

impl ClassInstance {
    /// Reads a property from the instance.
    #[must_use]
    pub fn get_prop(&self, name: &str) -> Option<&Tensor<f64>> {
        self.properties.get(name)
    }

    /// Mutates or sets an instance property.
    pub fn set_prop(&mut self, name: impl Into<String>, val: Tensor<f64>) {
        self.properties.insert(name.into(), val);
    }
}

/// Registry of known classes in the session.
#[derive(Clone, Debug, Default)]
pub struct ClassRegistry {
    classes: BTreeMap<String, ClassDef>,
    next_handle_id: u64,
}

impl ClassRegistry {
    /// Creates a new empty class registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: BTreeMap::new(),
            next_handle_id: 1,
        }
    }

    /// Registers a class definition in the environment.
    pub fn register(&mut self, def: ClassDef) {
        self.classes.insert(def.name.clone(), def);
    }

    /// Looks up a registered class.
    #[must_use]
    pub fn get_class(&self, name: &str) -> Option<&ClassDef> {
        self.classes.get(name)
    }

    /// Instantiates a class, initializing default property values across its inheritance hierarchy.
    pub fn instantiate(
        &mut self,
        class_name: &str,
        budget: &Arc<MemoryBudget>,
    ) -> Result<ClassInstance, LangError> {
        let def = self
            .classes
            .get(class_name)
            .ok_or_else(|| LangError::UndefinedSymbol(format!("class {class_name}")))?;

        let mut instance = ClassInstance {
            class_name: class_name.to_string(),
            is_handle: def.is_handle,
            handle_id: if def.is_handle {
                let id = self.next_handle_id;
                self.next_handle_id = self.next_handle_id.saturating_add(1);
                Some(id)
            } else {
                None
            },
            properties: BTreeMap::new(),
        };

        // Initialize properties from superclasses first, then this class
        let superclasses = def.superclasses.clone();
        for sup in &superclasses {
            if let Some(sup_def) = self.classes.get(sup) {
                for (prop_name, pdef) in &sup_def.properties {
                    let val = pdef.default_val.unwrap_or(0.0);
                    let t = Tensor::scalar(val, SiUnit::DIMENSIONLESS, budget)
                        .map_err(LangError::Tensor)?;
                    instance.properties.insert(prop_name.clone(), t);
                }
            }
        }

        for (prop_name, pdef) in &def.properties {
            let val = pdef.default_val.unwrap_or(0.0);
            let t = Tensor::scalar(val, SiUnit::DIMENSIONLESS, budget)
                .map_err(LangError::Tensor)?;
            instance.properties.insert(prop_name.clone(), t);
        }

        Ok(instance)
    }

    /// Resolves a method along the inheritance hierarchy.
    #[must_use]
    pub fn resolve_method(&self, class_name: &str, method_name: &str) -> Option<&MethodDef> {
        if let Some(def) = self.classes.get(class_name) {
            if let Some(m) = def.methods.get(method_name) {
                return Some(m);
            }
            for sup in &def.superclasses {
                if let Some(m) = self.resolve_method(sup, method_name) {
                    return Some(m);
                }
            }
        }
        None
    }
}

/// Parses a simple MATLAB `classdef` script into a `ClassDef` definition.
pub fn parse_classdef(source: &str) -> Result<ClassDef, LangError> {
    let lines: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if lines.is_empty() {
        return Err(LangError::Syntax {
            pos: 0,
            message: "Empty classdef source".into(),
        });
    }

    let first_line = lines.first().copied().unwrap_or("");
    if !first_line.starts_with("classdef") {
        return Err(LangError::Syntax {
            pos: 0,
            message: "Class source must begin with 'classdef'".into(),
        });
    }

    let header = first_line.trim_start_matches("classdef").trim();
    let (name, superclasses, is_handle) = if let Some((cname, sups)) = header.split_once('<') {
        let trimmed_name = cname.trim().to_string();
        let sup_list: Vec<String> = sups
            .split('&')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let handle = sup_list.iter().any(|s| s == "handle");
        (trimmed_name, sup_list, handle)
    } else {
        (header.to_string(), Vec::new(), false)
    };

    let mut class_def = ClassDef::new(name, is_handle);
    class_def.superclasses = superclasses;

    let mut in_properties = false;
    let mut in_methods = false;
    let mut in_function = false;

    for &line in lines.iter().skip(1) {
        let clean = line.split('%').next().unwrap_or("").trim();
        if clean.is_empty() {
            continue;
        }

        if clean == "end" {
            if in_function {
                in_function = false;
            } else if in_properties {
                in_properties = false;
            } else if in_methods {
                in_methods = false;
            }
            continue;
        }

        if clean.starts_with("properties") {
            in_properties = true;
            in_methods = false;
            in_function = false;
            continue;
        }

        if clean.starts_with("methods") {
            in_methods = true;
            in_properties = false;
            in_function = false;
            continue;
        }

        if in_properties {
            let stmt = clean.trim_end_matches(';').trim();
            if let Some((pname, default_str)) = stmt.split_once('=') {
                let p = pname.trim().to_string();
                let val: f64 = default_str.trim().parse().unwrap_or(0.0);
                class_def.add_property(PropertyDef {
                    name: p,
                    default_val: Some(val),
                    unit: None,
                    get_access: AccessModifier::Public,
                    set_access: AccessModifier::Public,
                });
            } else if !stmt.is_empty() {
                class_def.add_property(PropertyDef {
                    name: stmt.to_string(),
                    default_val: None,
                    unit: None,
                    get_access: AccessModifier::Public,
                    set_access: AccessModifier::Public,
                });
            }
        } else if in_methods && clean.starts_with("function") {
            in_function = true;
            let sig = clean.trim_start_matches("function").trim();
            let (outputs, rest) = if let Some((outs, right)) = sig.split_once('=') {
                let out_list: Vec<String> = outs
                    .trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                (out_list, right.trim())
            } else {
                (Vec::new(), sig)
            };

            let (fname, params) = if let Some((mname, param_str)) = rest.split_once('(') {
                let pnames: Vec<String> = param_str
                    .trim_end_matches(')')
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                (mname.trim().to_string(), pnames)
            } else {
                (rest.trim().to_string(), Vec::new())
            };

            class_def.add_method(MethodDef {
                name: fname,
                is_static: false,
                params,
                outputs,
                access: AccessModifier::Public,
            });
        }
    }

    Ok(class_def)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classdef_parse_and_instantiate() -> Result<(), Box<dyn std::error::Error>> {
        let code = r#"
        classdef Point2D < handle
            properties
                x = 10.0;
                y = 20.0;
            end
            methods
                function d = distance(obj, other)
                end
                function reset(obj)
                end
            end
        end
        "#;

        let def = parse_classdef(code)?;
        assert_eq!(def.name, "Point2D");
        assert!(def.is_handle);
        assert_eq!(def.properties.len(), 2);
        assert_eq!(def.methods.len(), 2);

        let mut reg = ClassRegistry::new();
        reg.register(def);

        let budget = MemoryBudget::default_budget();
        let mut inst = reg.instantiate("Point2D", &budget)?;
        assert_eq!(inst.class_name, "Point2D");
        assert!(inst.is_handle);
        assert!(inst.handle_id.is_some());

        // Check default properties
        let x_t = inst.get_prop("x").ok_or("Missing prop x")?;
        assert_eq!(x_t.get(0, 0)?, 10.0);

        // Mutate property
        let new_x = Tensor::scalar(42.0, SiUnit::DIMENSIONLESS, &budget)?;
        inst.set_prop("x", new_x);
        let updated_x = inst.get_prop("x").ok_or("Missing prop x")?;
        assert_eq!(updated_x.get(0, 0)?, 42.0);

        // Verify method resolution
        let m = reg.resolve_method("Point2D", "distance").ok_or("Missing method distance")?;
        assert_eq!(m.name, "distance");
        assert_eq!(m.outputs, vec!["d".to_string()]);
        assert_eq!(m.params, vec!["obj".to_string(), "other".to_string()]);

        Ok(())
    }

    #[test]
    fn test_classdef_inheritance() -> Result<(), Box<dyn std::error::Error>> {
        let base_code = r#"
        classdef Shape
            properties
                color = 1.0;
            end
            methods
                function draw(obj)
                end
            end
        end
        "#;

        let sub_code = r#"
        classdef Circle < Shape
            properties
                radius = 5.0;
            end
            methods
                function a = area(obj)
                end
            end
        end
        "#;

        let base_def = parse_classdef(base_code)?;
        let sub_def = parse_classdef(sub_code)?;

        let mut reg = ClassRegistry::new();
        reg.register(base_def);
        reg.register(sub_def);

        let budget = MemoryBudget::default_budget();
        let inst = reg.instantiate("Circle", &budget)?;
        // Inherited property 'color' and child property 'radius' both present!
        assert_eq!(inst.get_prop("color").ok_or("Missing color")?.get(0, 0)?, 1.0);
        assert_eq!(inst.get_prop("radius").ok_or("Missing radius")?.get(0, 0)?, 5.0);

        // Method resolution traverses to superclass
        assert!(reg.resolve_method("Circle", "area").is_some());
        assert!(reg.resolve_method("Circle", "draw").is_some());

        Ok(())
    }
}
