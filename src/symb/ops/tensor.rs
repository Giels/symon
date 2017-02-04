use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};
use ::symb::ops::utils::{ConstantLike};

use ::arrayfire::{Dim4};

#[derive(Debug)]
pub struct Transpose {
    inp: NodeID,
}

impl Transpose {
    pub fn new(x: NodeID) -> Box<Transpose> {
        Box::new(Transpose {
			inp: x,
        })
    }
}

impl Node for Transpose {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::transpose(&inputs[0], false)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node = Transpose::new(g.unwrap());
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct SumAll {
    inp: NodeID,
}

impl SumAll {
    pub fn new(x: NodeID) -> Box<SumAll> {
        Box::new(SumAll {
			inp: x,
        })
    }
}

impl Node for SumAll {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let res = ::arrayfire::sum_all(&inputs[0]);
	    if inputs[0].get_type() == ::arrayfire::DType::C64 || inputs[0].get_type() == ::arrayfire::DType::C32 {
        	::arrayfire::Array::new(&[res.0, res.1], Dim4::new(&[2, 1, 1, 1]))
		} else {
        	::arrayfire::Array::new(&[res.0], Dim4::new(&[1, 1, 1, 1]))
		}
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))]
					},
		}
	}
}

#[derive(Debug)]
pub struct FlipAll {
    inp: NodeID,
}

impl FlipAll {
    pub fn new(x: NodeID) -> Box<FlipAll> {
        Box::new(FlipAll {
			inp: x,
        })
    }
}

impl Node for FlipAll {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let mut accum = None;
		for i in (0..inputs[0].numdims()) {
			accum = accum.and_then(|x| Some(::arrayfire::flip(&x, i))).or(Some(::arrayfire::flip(&inputs[0], i)));
		}

		accum.unwrap()
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node = FlipAll::new(g.unwrap());
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct StdevAll {
    inp: NodeID,
}

impl StdevAll {
    pub fn new(x: NodeID) -> Box<StdevAll> {
        Box::new(StdevAll {
			inp: x,
        })
    }
}

impl Node for StdevAll {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let res = ::arrayfire::stdev_all(&inputs[0]);
	    if inputs[0].get_type() == ::arrayfire::DType::C64 || inputs[0].get_type() == ::arrayfire::DType::C32 {
        	::arrayfire::Array::new(&[res.0, res.1], Dim4::new(&[2, 1, 1, 1]))
		} else {
        	::arrayfire::Array::new(&[res.0], Dim4::new(&[1, 1, 1, 1]))
		}
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))] //TODO
					},
		}
	}
}

#[derive(Debug)]
pub struct VarianceAll {
    inp: NodeID,
}

impl VarianceAll {
    pub fn new(x: NodeID) -> Box<VarianceAll> {
        Box::new(VarianceAll {
			inp: x,
        })
    }
}

impl Node for VarianceAll {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let res = ::arrayfire::var_all(&inputs[0], true);
		if inputs[0].get_type() == ::arrayfire::DType::C64 || inputs[0].get_type() == ::arrayfire::DType::C32 {
        	::arrayfire::Array::new(&[res.0, res.1], Dim4::new(&[2, 1, 1, 1]))
		} else {
        	::arrayfire::Array::new(&[res.0], Dim4::new(&[1, 1, 1, 1]))
		}
		// sample variance
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))] //TODO
					},
		}
	}
}

#[derive(Debug)]
pub struct Sum {
    inp: NodeID,
	dim: i32,
}

impl Sum {
    pub fn new(x: NodeID, dim: i32) -> Box<Sum> {
        Box::new(Sum {
			inp: x,
			dim: dim,
        })
    }
}

impl Node for Sum {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        	::arrayfire::sum(&inputs[0], self.dim)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))] //TODO: Tile
					},
		}
	}
}

#[derive(Debug)]
pub struct Flip {
    inp: NodeID,
	dim: u32,
}

impl Flip {
    pub fn new(x: NodeID, dim: u32) -> Box<Flip> {
        Box::new(Flip {
			inp: x,
			dim: dim,
        })
    }
}

impl Node for Flip {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::flip(&inputs[0], self.dim)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node = Flip::new(g.unwrap(), self.dim);
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct Stdev {
    inp: NodeID,
	dim: i64,
}

impl Stdev {
    pub fn new(x: NodeID, dim: i64) -> Box<Stdev> {
        Box::new(Stdev {
			inp: x,
			dim: dim,
        })
    }
}

impl Node for Stdev {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
        ::arrayfire::stdev(&inputs[0], self.dim)
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))] //TODO
					},
		}
	}
}

#[derive(Debug)]
pub struct Variance {
    inp: NodeID,
	dim: i64,
}

impl Variance {
    pub fn new(x: NodeID, dim: i64) -> Box<Variance> {
        Box::new(Variance {
			inp: x,
			dim: dim,
        })
    }
}

impl Node for Variance {
    fn get_inputs(&self) -> Vec<NodeID> {
        vec![self.inp]
    }

    fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::var(&inputs[0], true, self.dim) // sample variance
    }

	fn backward(&self, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(self.inp, 1.))] //TODO
					},
		}
	}
}
