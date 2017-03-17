use std::fmt;

use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};

#[derive(Debug)]
pub struct Cos {
    inp: NodeID,
}

impl Cos {
    pub fn new(x: NodeID) -> Box<Cos> {
        Box::new(Cos {
			inp: x,
        })
    }
}

impl Node for Cos {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::cos(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let s = graph.add(Sin::new(self.inp));
		let ms = graph.add(Neg::new(s));
		let grad = graph.add(Mul::new(ms, g));
		vec![grad]
	}
}

#[derive(Debug)]
pub struct Sin {
    inp: NodeID,
}

impl Sin {
    pub fn new(x: NodeID) -> Box<Sin> {
        Box::new(Sin {
			inp: x,
        })
    }
}

impl Node for Sin {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::sin(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let c = graph.add(Cos::new(self.inp));
		let grad = graph.add(Mul::new(c, g));
		vec![grad]
	}
}

#[derive(Debug)]
pub struct Tan {
    inp: NodeID,
}

impl Tan {
    pub fn new(x: NodeID) -> Box<Tan> {
        Box::new(Tan {
			inp: x,
        })
    }
}

impl Node for Tan {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::tan(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let gg = graph.add(Add::new(g, g));
		let tantan = graph.add(Add::new(this, this));
		let one = graph.add(ConstantLike::new(1., tantan));
		let tan2p1 = graph.add(Add::new(tantan, one));
		let g2otan2p1 = graph.add(Div::new(gg, tan2p1));
		vec![g2otan2p1]
	}
}

#[derive(Debug)]
pub struct Cosh {
    inp: NodeID,
}

impl Cosh {
    pub fn new(x: NodeID) -> Box<Cosh> {
        Box::new(Cosh {
			inp: x,
        })
    }
}

impl Node for Cosh {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::cosh(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let s = graph.add(Sinh::new(self.inp));
		vec![graph.add(Mul::new(s, g))]
	}
}

#[derive(Debug)]
pub struct Sinh {
    inp: NodeID,
}

impl Sinh {
    pub fn new(x: NodeID) -> Box<Sinh> {
        Box::new(Sinh {
			inp: x,
        })
    }
}

impl Node for Sinh {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::sinh(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let c = graph.add(Cosh::new(self.inp));
		vec![graph.add(Mul::new(c, g))]
	}
}

#[derive(Debug)]
pub struct Tanh {
    inp: NodeID,
}

impl Tanh {
    pub fn new(x: NodeID) -> Box<Tanh> {
        Box::new(Tanh {
			inp: x,
        })
    }
}

impl Node for Tanh {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::tanh(&inputs[0])
    }

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		let cm = graph.add(Cosh::new(this));
		let num = graph.add(Add::new(cm, cm));
		let me2 = graph.add(Add::new(this, this));
		let cm2 = graph.add(Cosh::new(me2));
		let one = graph.add(ConstantLike(1., cm2));
		let denom = graph.add(Add::new(cm2, one));
		let denom2 = graph.add(Mul::new(denom, denom));
		let grad = graph.add(Mul::new(g, num));
		vec![graph.add(Div::new(grad, denom2))]
	}
}
