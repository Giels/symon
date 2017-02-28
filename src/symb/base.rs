use std::fmt::{Debug};

use ::symb::graph::{Graph};

pub type NodeID = u64;
pub type NodeData = ::arrayfire::Array;

pub trait Node: Debug {
    fn get_inputs(&self) -> Vec<NodeID>;
    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData;
	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID>;
}
