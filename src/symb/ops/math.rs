use std::fmt;

use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};

use ::symb::ops::tensor::{Transpose};

use ::arrayfire::{ConvMode, ConvDomain};

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
					write!(f, "Var({:?})", data)
				},
			None => write!(f, "Var()!"),
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

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		g.and_then(|g| Some(vec![g])).unwrap()
	}
}

#[derive(Debug)]
pub struct Add {
    l: NodeID,
    r: NodeID,
}

impl Add {
    pub fn new(a: NodeID, b: NodeID) -> Box<Add> {
        Box::new(Add {
            l: a,
            r: b,
        })
    }
}

impl Node for Add {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.l, self.r]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::add(inputs[0], inputs[1], true)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		vec![g, g]
	}
}

#[derive(Debug)]
pub struct Sub {
    l: NodeID,
    r: NodeID,
}

impl Sub {
    pub fn new(a: NodeID, b: NodeID) -> Box<Sub> {
        Box::new(Sub {
            l: a,
            r: b,
        })
    }
}

impl Node for Sub {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.l, self.r]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::sub(inputs[0], inputs[1], true)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		vec![g, graph.add(Neg::new(g))]
	}
}

#[derive(Debug)]
pub struct Conv {
    inp: NodeID,
    filt: NodeID,
	expand: bool,
}

impl Conv {
    pub fn new(a: NodeID, b: NodeID, expand: bool) -> Box<Conv> {
        Box::new(Conv {
            inp: a,
            filt: b,
			expand: expand,
        })
    }
}

impl Node for Conv {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp, self.filt]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let conv = match inputs[0].numdims() {
			4 => ::arrayfire::convolve3,
			3 => ::arrayfire::convolve2,
			_ => ::arrayfire::convolve1,
		};

        conv(inputs[0], inputs[1], if self.expand { ConvMode::EXPAND } else { ConvMode::DEFAULT }, ConvDomain::AUTO)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();
		vec![graph.add(Conv::new(g, self.filt, self.expand)), graph.add(Conv::new(self.inp, g, self.expand))]
	}
}

#[derive(Debug)]
pub struct Div {
    l: NodeID,
    r: NodeID,
}

impl Div {
    pub fn new(a: NodeID, b: NodeID) -> Box<Div> {
        Box::new(Div {
            l: a,
            r: b,
        })
    }
}

impl Node for Div {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.l, self.r]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::div(inputs[0], inputs[1], true)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();

		let r2 = graph.add(Mul::new(self.r, self.r));
		let lg = graph.add(Mul::new(self.l, g));
		let lg_r2 = graph.add(Div::new(lg, r2));
		let lg_mr2 = graph.add(Neg::new(lg_r2));
		let node_l = graph.add(Div::new(g, self.r));
		let node_r = lg_mr2;

		vec![node_l, node_r]
	}
}

#[derive(Debug)]
pub struct Mul {
    l: NodeID,
    r: NodeID,
}

impl Mul {
    pub fn new(a: NodeID, b: NodeID) -> Box<Mul> {
        Box::new(Mul {
            l: a,
            r: b,
        })
    }
}

impl Node for Mul {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.l, self.r]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::mul(inputs[0], inputs[1], true)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node_l = self.r;
		let node_r = self.l;

		vec![node_l, node_r]
	}
}

#[derive(Debug)]
pub struct MatMul {
    l: NodeID,
    r: NodeID,
}

impl MatMul {
    pub fn new(a: NodeID, b: NodeID) -> Box<MatMul> {
        Box::new(MatMul {
            l: a,
            r: b,
        })
    }
}

impl Node for MatMul {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.l, self.r]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::matmul(&inputs[0], &inputs[1], ::arrayfire::MatProp::NONE, ::arrayfire::MatProp::NONE)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let g = g.unwrap();

		let l_t = graph.add(Transpose::new(self.l));
		let r_t = graph.add(Transpose::new(self.r));
		
		let node_l = graph.add(MatMul::new(g, r_t));
		let node_r = graph.add(MatMul::new(l_t, g));

		vec![node_l, node_r]
	}
}

#[derive(Debug)]
pub struct Neg {
    inp: NodeID,
}

impl Neg {
    pub fn new(x: NodeID) -> Box<Neg> {
        Box::new(Neg {
			inp: x,
        })
    }
}

impl Node for Neg {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		-(inputs[0].clone())
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node = Neg::new(g.unwrap());
		vec![graph.add(node)]
	}
}