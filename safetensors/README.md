# Tract SafeTensors Loader (`tract-safetensors`)

This crate provides a loader for reading `.safetensors` files into the Tract inference engine. It allows you to load tensor data stored in the SafeTensors format and use it within Tract, typically as weights or constants in a model graph.

## Overview

The SafeTensors format is designed for safe and fast tensor storage. This loader integrates SafeTensors with Tract, enabling you to:

*   Load all tensors from a `.safetensors` file into a Tract `TypedModel`.
*   Represent these tensors primarily as `Const` (constant) nodes within the Tract graph.
*   Optionally designate specific tensors as model `Source` (input) nodes via metadata.
*   Use these loaded tensors in conjunction with graphs defined programmatically in Tract or loaded from other formats.

## Usage

### Dependency

To use `tract-safetensors` in your project, add it to your `Cargo.toml`:

```toml
[dependencies]
# If using from crates.io (once published)
tract-safetensors = "0.1.0" # Replace with actual version

# If using as a local path within the Tract workspace (e.g., from an example project)
# tract-safetensors = { path = "../safetensors" } 

tract-hir = "0.21.14-pre" # Or { path = "../hir" }
tract-core = "0.21.14-pre" # Or { path = "../core" }
```

*(Note: Adjust paths or specify versions depending on how you are consuming this crate, especially during development versus after potential publishing.)*

### Loading a .safetensors file

You can load a `.safetensors` file using the `SafetensorFramework` from this crate, which would implement Tract's `Framework` trait.

```rust
use tract_hir::prelude::*;
// Assuming SafetensorFramework will be exposed from the tract_safetensors crate
// use tract_safetensors::SafetensorFramework; 

fn main() -> TractResult<()> {
    let model_path = "path/to/your/model.safetensors";

    // The conventional way in Tract is to use a Framework struct.
    // Once implemented, loading would look like this:
    //
    // let safetensor_loader = SafetensorFramework::default(); // Or a ::new() constructor
    // let typed_model = safetensor_loader.model_for_path(model_path)?;
    // 
    // println!("Model loaded: {}", typed_model.name.as_deref().unwrap_or("Unnamed"));

    // If you have tract-cli installed and tract-safetensors is registered,
    // you might also be able to inspect it via:
    // tract dump --format safetensors path/to/your/model.safetensors

    println!("Conceptual: Model '{}' would be loaded using SafetensorFramework.", model_path);
    // For actual execution, see the 'Running a Model with Safetensors Weights' section.

    Ok(())
}
```

*Note: The final API for `SafetensorFramework` (e.g., constructor, registration) will follow Tract's established conventions once fully implemented.*

### Inspecting the Loaded Model

Once loaded, the `TypedModel` will contain nodes corresponding to the tensors in the `.safetensors` file.

```rust
// Assuming `typed_model` is loaded as above

// for node in typed_model.nodes() {
//     println!("Node Name: {}, Op: {:?}", node.name, node.op());
//     if let Some(const_op) = node.op().as_any().downcast_ref::<tract_hir::ops::konst::Const>() {
//         println!("  Tensor Value: {:?}", const_op.0);
//         println!("  Shape: {:?}, Dtype: {:?}", const_op.0.shape(), const_op.0.datum_type());
//     }
//     if let Some(source_op) = node.op().as_any().downcast_ref::<tract_hir::ops::source::Source>() {
//         println!("  Source Node (Input)");
//         println!("  Declared Fact: {:?}", source_op.fact);
//     }
// }

// // You can also access tensors by name if names were preserved as node names
// if let Some(node_id) = typed_model.node_by_name("my_tensor_name")? {
//     let node = &typed_model.nodes()[node_id];
//     // ... inspect node as above ...
// }
```

### Running a Model with Safetensors Weights

Since `.safetensors` files only store tensor data, you need to define the computation graph separately. Here's a conceptual example of how you might use Tract's API to build a simple graph and then use weights from a loaded SafeTensors model.

```rust
use tract_hir::prelude::*;
use tract_core::ndarray;

// Assume `safetensors_model: TypedModel` is loaded and contains your weights
// e.g., a tensor named "fc1.weight" and "fc1.bias"

fn build_and_run_model(safetensors_model: &TypedModel, input_data: Tensor) -> TractResult<TVec<Tensor>> {
    let mut graph = TypedModel::default();

    // 1. Get weights and biases from the safetensors_model
    // This requires knowing the names of your tensors in the .safetensors file.
    let weight_node_id = safetensors_model.node_by_name("fc1.weight")?;
    let weight_tensor = safetensors_model.nodes()[weight_node_id]
        .op().as_any().downcast_ref::<tract_hir::ops::konst::Const>().unwrap().0.clone();

    let bias_node_id = safetensors_model.node_by_name("fc1.bias")?;
    let bias_tensor = safetensors_model.nodes()[bias_node_id]
        .op().as_any().downcast_ref::<tract_hir::ops::konst::Const>().unwrap().0.clone();

    // 2. Define your graph structure using Tract's API
    let input_fact = InferenceFact::dt_shape(input_data.datum_type(), input_data.shape());
    let source = graph.add_source("input", input_fact)?;

    // Add the loaded weights and biases as constants in the new graph
    let weight_const = graph.add_const("fc1.weight", weight_tensor)?;
    let bias_const = graph.add_const("fc1.bias", bias_tensor)?;

    // Create a MatMul_x_Const op (representing input * weight)
    // The actual op might be different (e.g., MatMul) depending on Tract's conventions
    // and how you want to structure it. This is a simplified example.
    let matmul_op = tract_hir::ops::matmul::MatMul::default(); // Or appropriate constructor
    let matmul_node = graph.add_node("matmul", matmul_op, &[])?; // Output facts inferred or set
    graph.add_edge(source, InletId::new(matmul_node, 0))?;
    graph.add_edge(weight_const, InletId::new(matmul_node, 1))?;

    // Add bias using an Add op
    let add_op = tract_hir::ops::binary::BinaryOp::Add;
    let add_node = graph.add_node("add_bias", add_op, &[])?;
    graph.add_edge(OutletId::new(matmul_node, 0), InletId::new(add_node, 0))?;
    graph.add_edge(bias_const, InletId::new(add_node, 1))?;

    graph.set_output_outlets(&[OutletId::new(add_node, 0)])?;

    // 3. Optimize and run the model
    let runnable_model = graph.into_optimized()?.into_runnable()?;
    
    runnable_model.run(tvec!(input_data))
}

// Example usage:
// let example_input = Tensor::from_shape_vec(&[1, 10], vec![1.0f32; 10])?;
// let results = build_and_run_model(&loaded_safetensors_model, example_input)?;
// println!("Results: {:?}", results);
```

This example is conceptual. The exact way you retrieve tensors and build the graph will depend on your model's architecture and how tensors are named in the `.safetensors` file.

### Using Metadata for Inputs/Outputs

The SafeTensors format allows for a `__metadata__` section in its JSON header. The `tract-safetensors` loader can use this to determine which tensors should be treated as model inputs (created as `Source` nodes) rather than `Const` nodes.

*   **`__metadata__["tract_inputs"]`**: A comma-separated string of tensor names that should be treated as inputs. For example: `"input_ids,attention_mask"`.
*   **`__metadata__["tract_outputs"]`**: (Optional) A comma-separated string of tensor names that should be set as the model's output outlets. If not provided, all loaded tensors might be set as outputs by default in the initial implementation.

When `tract_inputs` is used, the corresponding tensors will be represented as `Source` nodes in the Tract graph. Their actual data from the `.safetensors` file might be ignored or could be stored in the `Source` op's `konst` field for inspection, depending on the final loader implementation.

## Testing

To test this loader (once implemented):

1.  **Create a simple `.safetensors` file:** You can use the `safetensors` Python library to create a file with a few tensors of known shapes, data types, and values. Include a `__metadata__` section if you want to test input/output designation.
    ```python
    from safetensors.torch import save_file
    import torch

    tensors = {
        "weight1": torch.ones((3, 2)),
        "bias1": torch.zeros((2)),
        "input_data": torch.rand((1, 3)) # Will be an input if specified in metadata
    }
    metadata = {
        "tract_inputs": "input_data",
        "tract_outputs": "bias1" # Or some other relevant output
    }
    save_file(tensors, "test_model.safetensors", metadata=metadata)
    ```
2.  **Write Rust unit tests:** In `tract-safetensors/src/lib.rs` or a dedicated test module, use the loading functions to load `test_model.safetensors`.
3.  **Verify:**
    *   Check that the correct number of nodes are in the loaded `TypedModel`.
    *   For each expected tensor, verify its name, shape, and `DatumType`.
    *   Compare the tensor data with the original values.
    *   If using `tract_inputs` metadata, verify that the specified tensors are `Source` nodes.
    *   Verify that model outputs are set correctly based on `tract_outputs` or the default behavior.

Example test structure (conceptual):

```rust
// in tract-safetensors/src/lib.rs or similar

// #[cfg(test)]
// mod tests {
//     use super::*; // Assuming your loader functions are in the parent module
//     use tract_hir::prelude::*;
//     use std::path::PathBuf;

//     fn create_test_safetensors_file() -> PathBuf {
//         // Use a library or command to run the Python script above to generate the file
//         // Or, check in a pre-generated test_model.safetensors file.
//         // For simplicity, assume it's pre-generated in "tests/test_data/test_model.safetensors"
//         let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//         path.push("tests/test_data/test_model.safetensors");
//         // Ensure this file is actually created by Python script during test setup or checked in.
//         if !path.exists() {
//             panic!("Test safetensors file not found at {:?}. Please generate it.", path);
//         }
//         path
//     }

//     #[test]
//     fn test_load_simple_safetensors() -> TractResult<()> {
//         let file_path = create_test_safetensors_file();
//         // let model = SafetensorFramework::new().model_for_path(file_path)?;
//         // Or: let model = crate::load_model(file_path)?;

//         // Dummy assertion until loader is implemented
//         // assert!(!model.nodes().is_empty(), "Model should have nodes");

//         // Example assertions:
//         // let weight1_node_id = model.node_by_name("weight1")?;
//         // let weight1_op = model.nodes()[weight1_node_id].op().as_any().downcast_ref::<tract_hir::ops::konst::Const>().unwrap();
//         // assert_eq!(weight1_op.0.shape(), &[3, 2]);
//         // assert_eq!(weight1_op.0.datum_type(), f32::datum_type());
//         // // Compare data if necessary

//         // let input_data_node_id = model.node_by_name("input_data")?;
//         // assert!(model.nodes()[input_data_node_id].op().as_any().is::<tract_hir::ops::source::Source>());

//         // assert_eq!(model.output_outlets()?.len(), 1);
//         // let output_node_name = model.node_name(model.output_outlets()?[0].node); 
//         // assert_eq!(output_node_name, "bias1");

//         Ok(())
//     }
// }
```

This detailed README should provide a good starting point for users and developers of the `tract-safetensors` crate.
