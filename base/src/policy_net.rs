use std::error::Error;
use std::io::Cursor;
use tract_onnx::prelude::*;

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct PolicyNet {
    model: RunnableModel,
}

impl PolicyNet {
    pub fn new(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut reader = Cursor::new(bytes);

        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)?
            // .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 6)))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model })
    }

    /// Forward pass. Returns (mean, std) vectors.
    pub fn forward(
        &mut self,
        input: Vec<f32>,
        input_shape: Vec<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let shape_tuple: Vec<usize> = input_shape; // tract compatible shape
        let input_array = tract_ndarray::ArrayD::from_shape_vec(shape_tuple, input)?;
        let input_tensor: Tensor = input_array.into();

        let result = self.model.run(tvec!(input_tensor.into()))?;

        if result.len() != 2 {
            return Err("Expected two outputs: mean and std".into());
        }

        let mean = result[0]
            .to_array_view::<f32>()?
            .as_slice()
            .unwrap()
            .to_vec();

        let std = result[1]
            .to_array_view::<f32>()?
            .as_slice()
            .unwrap()
            .to_vec();

        Ok((mean, std))
    }

    pub fn get_action(
        &mut self,
        input: Vec<f32>,
        input_shape: Vec<usize>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (mean, _) = self.forward(input, input_shape)?;

        // Apply tanh to mean
        let action: Vec<f32> = mean.into_iter().map(|x| x.tanh()).collect();
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_net_forward() {
        let model_bytes = std::fs::read("../python/models/policy_net.onnx").unwrap();

        let mut net = PolicyNet::new(&model_bytes).unwrap();
        let obs_dim = 6;
        let act_dim = 2;
        let input_shape = vec![1, obs_dim];
        let input = vec![0.0; obs_dim];

        let (mean, std) = net.forward(input, input_shape).unwrap();
        println!("Mean: {:?}", mean);
        assert_eq!(mean.len(), act_dim);
        assert_eq!(std.len(), act_dim);
    }

    #[test]
    fn test_policy_net_get_action() {
        let model_bytes = std::fs::read("../python/models/policy_net.onnx").unwrap();
        let mut net = PolicyNet::new(&model_bytes).unwrap();
        let obs_dim = 6;
        let act_dim = 2;
        let input_shape = vec![1, obs_dim];
        let input = vec![0.0; obs_dim];

        let action = net.get_action(input, input_shape).unwrap();
        println!("Action: {:?}", action);
        assert_eq!(action.len(), act_dim);
    }
}
