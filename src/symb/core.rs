use std::fmt;

use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};

use ::arrayfire::{print_gen, Dim4, ConvMode, ConvDomain};

pub struct Var {
    val: Option<NodeData>,
}

impl Var {
	pub fn new() -> Box<Var> {
        Box::new(Var {
            val: None,
        })
    }

	pub fn new_shared(val: NodeData) -> Box<Var> {
		Box::new(Var {
			val: Some(val),
		})
	}
}

impl fmt::Debug for Var {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self.val {
			Some(ref x) => {
					let mut data: Vec<f32> = vec![0.; x.elements() as usize];
					x.host(&mut data);
					write!(f, "Variable:\n{:?}", data)
				},
			None => write!(f, "Variable not set!"),
		}
	}
}
 
impl Node for Var {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        self.val.clone().unwrap()
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		g.and_then(|g| Some(vec![g])).unwrap()
	}
}

pub struct Identity {
}

impl Identity {
	pub fn new(side: u64, dtype: ::arrayfire::DType) -> Box<Var> {
		Box::new(Var {
			val: Some(::arrayfire::identity_t(Dim4::new(&[side, side, 1, 1]), dtype)),
		})
	}
}
