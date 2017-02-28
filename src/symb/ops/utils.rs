use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};

#[derive(Debug)]
pub struct ConstantLike {
    inp: NodeID,
	cst: f32,
}

impl ConstantLike {
    pub fn new(cst: f32, x: NodeID) -> Box<ConstantLike> {
        Box::new(ConstantLike {
			inp: x,
			cst: cst,
        })
    }
}

impl Node for ConstantLike {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::constant_t(::arrayfire::Scalar::F32(self.cst), inputs[0].dims(), inputs[0].get_type())
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., g.unwrap()))]
	}
}
