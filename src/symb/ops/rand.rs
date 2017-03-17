use std::fmt;

use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};
use ::arrayfire::{Dim4};

#[derive(Debug)]
pub struct Uniform {
    dims: Dim4,
}

impl Uniform {
    pub fn new(dims: Dim4) -> Box<Uniform> {
        Box::new(Uniform {
			dims: dims,
        })
    }
}

impl Node for Uniform {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::randu(self.dims)
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![ConstantLike(0., this)]
	}
}

#[derive(Debug)]
pub struct Normal {
    dims: Dim4,
}

impl Normal {
    pub fn new(dims: Dim4) -> Box<Normal> {
        Box::new(Normal {
			dims: dims,
        })
    }
}

impl Node for Normal {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::randn(self.dims)
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![ConstantLike(0., this)]
	}
}
