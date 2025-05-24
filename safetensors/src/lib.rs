use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf}; // Added PathBuf

use anyhow::{anyhow, Context, Result};
use memmap2::MmapOptions;

use safetensors::tensor::{SafeTensors, TensorView, Dtype as SafetensorsDtype};
use tract_core::datum_type::DatumType;
use tract_core::framework::Framework; // Import the Framework trait
use tract_core::prelude::{Arc, Datum, IntoArc, Tensor, TVec, InferenceFact}; // Added InferenceFact
use tract_hir::prelude::{InferenceModelExt, Node, TypedModel, InferenceModel}; // Added InferenceModel

/// Main struct for the Safetensors framework loader.
#[derive(Default, Clone, Debug)]
pub struct SafetensorFramework;

/// Maps safetensors::Dtype to tract_core::datum_type::DatumType
fn to_tract_dtype(dtype: SafetensorsDtype) -> Result<DatumType> {
    match dtype {
        SafetensorsDtype::F64 => Ok(DatumType::F64),
        SafetensorsDtype::F32 => Ok(DatumType::F32),
        SafetensorsDtype::F16 => Ok(DatumType::F16),
        SafetensorsDtype::BF16 => Ok(DatumType::BF16),
        SafetensorsDtype::I64 => Ok(DatumType::I64),
        SafetensorsDtype::I32 => Ok(DatumType::I32),
        SafetensorsDtype::I16 => Ok(DatumType::I16),
        SafetensorsDtype::I8 => Ok(DatumType::I8),
        SafetensorsDtype::U64 => Ok(DatumType::U64),
        SafetensorsDtype::U32 => Ok(DatumType::U32),
        SafetensorsDtype::U16 => Ok(DatumType::U16),
        SafetensorsDtype::U8 => Ok(DatumType::U8),
        SafetensorsDtype::BOOL => Ok(DatumType::Bool),
        _ => Err(anyhow!("Unsupported safetensors Dtype: {:?}", dtype)),
    }
}

/// Converts a safetensors::TensorView to a tract_hir::Tensor
fn view_to_tract_tensor(view: &TensorView) -> Result<Tensor> {
    let tract_dtype = to_tract_dtype(view.dtype())?;
    let shape: TVec<usize> = view.shape().to_vec().into();
    let data_bytes = view.data();

    let tensor = match tract_dtype {
        DatumType::F32 => {
            let mut vec_data = Vec::with_capacity(data_bytes.len() / std::mem::size_of::<f32>());
            for chunk in data_bytes.chunks_exact(std::mem::size_of::<f32>()) {
                vec_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Tensor::from_shape_vec(shape, vec_data)?
        }
        DatumType::F64 => {
            let mut vec_data = Vec::with_capacity(data_bytes.len() / std::mem::size_of::<f64>());
            for chunk in data_bytes.chunks_exact(std::mem::size_of::<f64>()) {
                vec_data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
            Tensor::from_shape_vec(shape, vec_data)?
        }
        DatumType::I64 => {
            let mut vec_data = Vec::with_capacity(data_bytes.len() / std::mem::size_of::<i64>());
            for chunk in data_bytes.chunks_exact(std::mem::size_of::<i64>()) {
                vec_data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
            }
            Tensor::from_shape_vec(shape, vec_data)?
        }
        DatumType::I32 => {
            let mut vec_data = Vec::with_capacity(data_bytes.len() / std::mem::size_of::<i32>());
            for chunk in data_bytes.chunks_exact(std::mem::size_of::<i32>()) {
                vec_data.push(i32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Tensor::from_shape_vec(shape, vec_data)?
        }
        // Extend with other types as necessary, ensuring correct byte handling
        DatumType::U8 => {
             Tensor::from_shape_vec(shape, data_bytes.to_vec())?
        }
        DatumType::I8 => {
            let s_i8: Vec<i8> = data_bytes.iter().map(|&x| x as i8).collect();
            Tensor::from_shape_vec(shape, s_i8)?
        }
        // ... other integer, boolean, float types
        _ => {
            let mut owned_data = Vec::with_capacity(data_bytes.len());
            owned_data.extend_from_slice(data_bytes);
            Tensor::from_raw_bytes_unchecked(shape, owned_data.into_arc_bytes(), tract_dtype)
        }
    };
    Ok(tensor)
}

/// Reads a .safetensors file and returns a map of tensor names to Tract tensors,
/// and any metadata.
pub fn load_safetensors_file(
    path: impl AsRef<Path>,
) -> Result<(HashMap<String, Tensor>, Option<HashMap<String, String>>)> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };

    let safetensors_data = SafeTensors::deserialize(&buffer)
        .map_err(|e| anyhow!("Failed to deserialize SafeTensors: {}", e))?;

    let mut tensors_map: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in safetensors_data.tensors() {
        let tract_tensor = view_to_tract_tensor(&view)
            .with_context(|| format!("Failed to convert tensor: {}", name))?;
        tensors_map.insert(name.clone(), tract_tensor);
    }
    
    let metadata = safetensors_data.metadata()
        .map(|hm| hm.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

    Ok((tensors_map, metadata))
}

// We use the path itself as the "ProtoModel" for simplicity, as Safetensors doesn't have
// a separate proto structure like ONNX or TFLite.
impl Framework<PathBuf, TypedModel> for SafetensorFramework {
    /// Load a model from a specified path.
    /// The `_model_template` is currently unused but part of the trait.
    fn model_for_path_with_model_template(
        &self,
        path: impl AsRef<Path>,
        _model_template: TypedModel,
    ) -> Result<TypedModel> {
        let (tensors_map, metadata) = load_safetensors_file(path.as_ref())?;
        let mut model = TypedModel::default(); // Or use _model_template if needed

        let input_names: HashMap<String, ()> = metadata
            .as_ref()
            .and_then(|md| md.get("tract_inputs"))
            .map(|s| s.split(',').map(|name| name.trim().to_string()).collect::<Vec<String>>())
            .unwrap_or_default()
            .into_iter()
            .map(|name| (name, ()))
            .collect();

        let mut output_outlets = vec![];

        for (name, tensor) in tensors_map {
            let fact = InferenceFact::from(&tensor);
            let node_name = name.clone(); // Node name can be same as tensor name

            let outlet_id = if input_names.contains_key(&name) {
                let source_node = tract_hir::ops::source::Source::new(node_name.clone(), fact);
                // If it's an input, the tensor data itself might be considered optional
                // or as a default value if not overridden by user input at runtime.
                // For now, we can store it with the source op if needed or make it purely symbolic.
                // Let's assume for now inputs are purely symbolic placeholders.
                // To make data available like a const that can be fed:
                // source_node.konst = Some(tensor.into_arc_tensor());
                model.add_node(node_name, source_node)?.into()
            } else {
                model.add_const(node_name, tensor)?
            };
            
            // By default, make all loaded tensors available as outputs
            output_outlets.push(outlet_id);
            model.set_outlet_label(outlet_id, name)?;
        }
        
        // Optionally, refine outputs based on metadata
        if let Some(output_names_str) = metadata.as_ref().and_then(|md| md.get("tract_outputs")) {
            let specified_output_names: Vec<String> = output_names_str.split(',').map(|s| s.trim().to_string()).collect();
            let mut new_output_outlets = vec![];
            for name_to_find in specified_output_names {
                if let Some(node) = model.node_by_name_maybe(&name_to_find)? {
                     // Assuming single outlet for const/source nodes
                    new_output_outlets.push(node.id.outlet(0));
                } else {
                    return Err(anyhow!("Specified output tensor '{}' not found in the model", name_to_find));
                }
            }
            output_outlets = new_output_outlets;
        }


        if !output_outlets.is_empty() {
            model.set_output_outlets(&output_outlets)?;
        }

        Ok(model)
    }
    
    // Implement other required methods of the Framework trait, potentially with default behavior or stubs for now.
    // For `model_for_proto_model_with_model_template`, if PathBuf is our "ProtoModel":
    fn model_for_proto_model_with_model_template(&self, proto_model: &PathBuf, model_template: TypedModel) -> Result<TypedModel> {
        self.model_for_path_with_model_template(proto_model, model_template)
    }

    // `proto_model_for_read` might not be directly applicable if we always expect a path.
    // Or it could read bytes and save to a temp file, then call model_for_path.
    // For now, let's stub it or make it error.
    fn proto_model_for_read(&self, _reader: &mut dyn std::io::Read) -> Result<PathBuf> {
        Err(anyhow!("Loading Safetensors from a generic reader is not yet implemented. Please provide a file path."))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    // For creating dummy safetensor files, we might need to use python if rust crate is too low level
    // Or have pre-generated files.
    use safetensors::serialize_to_file;
    // use safetensors::tensor::TensorView; // Required for serialize_to_file // This was commented out in the prompt

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(to_tract_dtype(SafetensorsDtype::F32).unwrap(), DatumType::F32);
        assert_eq!(to_tract_dtype(SafetensorsDtype::I64).unwrap(), DatumType::I64);
        // Add more basic dtype checks
    }

    // Helper function to create a simple safetensors file for testing
    // This requires the `View` trait to be implemented for the data we pass.
    // Let's define a minimal struct that implements View for testing.
    struct SimpleView<'a> {
        dtype: SafetensorsDtype,
        shape: Vec<usize>,
        data: &'a [u8],
    }

    impl<'a> safetensors::tensor::View for SimpleView<'a> {
        fn dtype(&self) -> SafetensorsDtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<[u8]> {
            std::borrow::Cow::Borrowed(self.data)
        }
    }


    #[test]
    fn test_load_and_parse_simple_safetensors_file() -> Result<()> {
        let mut dir = std::env::temp_dir();
        dir.push("test_simple.safetensors");
        let path = dir.as_path();

        let tensor_name = "test_tensor_1".to_string();
        let data_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_u8: Vec<u8> = data_f32.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let shape_f32: Vec<usize> = vec![2, 2];

        let view = SimpleView {
            dtype: SafetensorsDtype::F32,
            shape: shape_f32.clone(),
            data: &data_u8,
        };
        
        let mut tensors_to_save = HashMap::new();
        tensors_to_save.insert(tensor_name.clone(), view);

        serialize_to_file(&tensors_to_save, &None, path)
            .map_err(|e| anyhow!("Failed to serialize test safetensor file: {}", e))?;

        let framework = SafetensorFramework::default();
        let model = framework.model_for_path(path)?;

        assert_eq!(model.nodes().len(), 1); // Should have one Const node
        let node = &model.nodes()[0];
        assert_eq!(node.name, tensor_name);
        let op: &tract_hir::ops::konst::Const = node.op().downcast_ref().unwrap();
        assert_eq!(op.0.shape(), &*shape_f32);
        assert_eq!(op.0.as_slice::<f32>()?, &*data_f32);
        
        // Check default output
        assert_eq!(model.output_outlets()?.len(), 1);
        assert_eq!(model.output_outlets()?[0].node, node.id);

        std::fs::remove_file(path)?; // Clean up
        Ok(())
    }

    #[test]
    fn test_load_with_metadata_inputs_outputs() -> Result<()> {
        let mut dir = std::env::temp_dir();
        dir.push("test_metadata.safetensors");
        let path = dir.as_path();

        let tensor_in_name = "input_tensor".to_string();
        let data_in_f32: Vec<f32> = vec![1.0, 2.0];
        let data_in_u8: Vec<u8> = data_in_f32.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let shape_in_f32: Vec<usize> = vec![2];
        
        let view_in = SimpleView {
            dtype: SafetensorsDtype::F32,
            shape: shape_in_f32.clone(),
            data: &data_in_u8,
        };

        let tensor_const_name = "const_tensor".to_string();
        let data_const_f32: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let data_const_u8: Vec<u8> = data_const_f32.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let shape_const_f32: Vec<usize> = vec![2,2];

        let view_const = SimpleView {
            dtype: SafetensorsDtype::F32,
            shape: shape_const_f32.clone(),
            data: &data_const_u8,
        };
        
        let mut tensors_to_save = HashMap::new();
        tensors_to_save.insert(tensor_in_name.clone(), view_in);
        tensors_to_save.insert(tensor_const_name.clone(), view_const);
        
        let mut metadata = HashMap::new();
        metadata.insert("tract_inputs".to_string(), tensor_in_name.clone());
        metadata.insert("tract_outputs".to_string(), tensor_const_name.clone());

        serialize_to_file(&tensors_to_save, &Some(metadata), path)
            .map_err(|e| anyhow!("Failed to serialize test safetensor file: {}", e))?;

        let framework = SafetensorFramework::default();
        let model = framework.model_for_path(path)?;
        
        assert_eq!(model.nodes().len(), 2);
        let input_node_id = model.node_by_name(&tensor_in_name)?.id;
        let const_node_id = model.node_by_name(&tensor_const_name)?.id;

        assert!(model.node(input_node_id).op().is::<tract_hir::ops::source::Source>());
        assert!(model.node(const_node_id).op().is::<tract_hir::ops::konst::Const>());
        
        assert_eq!(model.input_outlets().len(), 1);
        assert_eq!(model.input_outlets()[0].node, input_node_id);
        
        assert_eq!(model.output_outlets()?.len(), 1);
        assert_eq!(model.output_outlets()?[0].node, const_node_id);

        std::fs::remove_file(path)?; // Clean up
        Ok(())
    }
}
