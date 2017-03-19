//TODO: inverse, max/min/argmax/argmin, median
// reshape, flatten, reorder, select, tile
// identity
// nonzero (locate)
// maxof, minof
use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};
use ::symb::ops::utils::{ConstantLike};

use ::arrayfire::{Dim4, Seq};

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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(1., self.inp))]
					},
		}
	}
}

#[derive(Debug)]
pub struct MeanAll {
	inp: NodeID,
}

impl MeanAll {
	pub fn new(x: NodeID) -> Box<MeanAll> {
		Box::new(MeanAll {
			inp: x,
		})
	}
}

impl Node for MeanAll {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let res = ::arrayfire::mean_all(&inputs[0]);
		if inputs[0].get_type() == ::arrayfire::DType::C64 || inputs[0].get_type() == ::arrayfire::DType::C32 {
			::arrayfire::Array::new(&[res.0, res.1], Dim4::new(&[2, 1, 1, 1]))
		} else {
			::arrayfire::Array::new(&[res.0], Dim4::new(&[1, 1, 1, 1]))
		}
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => {
							let n = graph.add(Elements::new(self.inp));
							vec![graph.add(Div::new(grad, n))]
						},
			None => {
						let n = graph.add(Elements::new(self.inp));
						let ones = ConstantLike::new(1., self.inp);
						vec![graph.add(Div::new(ones, n))]
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let half = graph.add(ConstantLike::new(0.5, this));
		let efx = graph.add(MeanAll::new(self.inp));
		let n = graph.add(Elements::new(self.inp));
		let nn = graph.add(Add::new(n, n));
		let nnfx = graph.add(Mul::new(nn, self.inp));
		let fx2 = graph.add(Add::new(self.inp, self.inp));
		let mfx2 = graph.add(Neg::new(fx2));
		let tnmt_fx = graph.add(Add::new(nnfx, mfx2));
		let tnmt_fx_g = graph.add(Mul::new(tnmt_fx, grad));
		let half_tnmt_fx_g = graph.add(Mul::new(half, tnmt_fx_g));
		let out = graph.add(Div::new(half_tnmt_fx_g, this));
		vec![out]
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let efx = graph.add(MeanAll::new(self.inp));
		let n = graph.add(Elements::new(self.inp));
		let nn = graph.add(Add::new(n, n));
		let nnfx = graph.add(Mul::new(nn, self.inp));
		let fx2 = graph.add(Add::new(self.inp, self.inp));
		let mfx2 = graph.add(Neg::new(fx2));
		let tnmt_fx = graph.add(Add::new(nnfx, mfx2));
		let tnmt_fx_g = graph.add(Mul::new(tnmt_fx, grad));
		vec![tnmt_fx_g]
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		match g {
			Some(grad) => vec![grad],
			None => {
						vec![graph.add(ConstantLike::new(1., self.inp))]
					},
		}
	}
}

#[derive(Debug)]
pub struct Mean {
	inp: NodeID,
	dim: i32,
}

impl Mean {
	pub fn new(x: NodeID, dim: i32) -> Box<Mean> {
		Box::new(Mean {
			inp: x,
			dim: dim,
		})
	}
}

impl Node for Mean {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::mean(&inputs[0], self.dim)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let shape = graph.add(Shape::new(self.inp));
		let denom = graph.add(Index::new(shape, [Seq::new(self.ndim, self.ndim, 1)]));
		vec![graph.add(Div::new(grad, denom))]
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let half = graph.add(ConstantLike::new(0.5, this));
		let efx = graph.add(Mean::new(self.inp, self.dim));
		let shape = graph.add(Shape::new(self.inp));
		let n = graph.add(Index::new(shape, [Seq::new(self.dim, self.dim, 1)]));
		let nn = graph.add(Add::new(n, n));
		let nnfx = graph.add(Mul::new(nn, self.inp));
		let fx2 = graph.add(Add::new(self.inp, self.inp));
		let mfx2 = graph.add(Neg::new(fx2));
		let tnmt_fx = graph.add(Add::new(nnfx, mfx2));
		let tnmt_fx_g = graph.add(Mul::new(tnmt_fx, grad));
		let half_tnmt_fx_g = graph.add(Mul::new(half, tnmt_fx_g));
		let out = graph.add(Div::new(half_tnmt_fx_g, this));
		vec![out]
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

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let efx = graph.add(Mean::new(self.inp, self.dim));
		let shape = graph.add(Shape::new(self.inp));
		let n = graph.add(Index::new(shape, [Seq::new(self.dim, self.dim, 1)]));
		let nn = graph.add(Add::new(n, n));
		let nnfx = graph.add(Mul::new(nn, self.inp));
		let fx2 = graph.add(Add::new(self.inp, self.inp));
		let mfx2 = graph.add(Neg::new(fx2));
		let tnmt_fx = graph.add(Add::new(nnfx, mfx2));
		let tnmt_fx_g = graph.add(Mul::new(tnmt_fx, grad));
		vec![tnmt_fx_g]
	}
}

#[derive(Debug)]
pub struct Shape {
	inp: NodeID,
}

impl Shape {
	pub fn new(x: NodeID) -> Box<Shape> {
		Box::new(Shape {
			inp: x,
		})
	}
}

impl Node for Shape {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::Array::new<u64>(inputs[0].dims.get(), Dim4::new(&[4, 1, 1, 1]))
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0, this))]
	}
}

#[derive(Debug)]
pub struct Elements {
    inp: NodeID,
}

impl Elements {
	pub fn new(x: NodeID) -> Box<Elements> {
		Box::new(Elements {
			inp: x,
		})
	}
}

impl Node for Elements {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::constant_t(::arrayfire::Scalar::S64(inputs[0].elements()), Dim4::new(&[1, 1, 1, 1]), ::arrayfire::DType::S64)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let node = ConstantLike(0., this);
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct Index {
	inp: NodeID,
	indices: [Seq],
}

impl Index {
	pub fn new(x: NodeID, indices: [Seq]) -> Box<Index> {
		Box::new(Index {
			inp: x,
			indices: indices,
		})
	}
}

impl Node for Index {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::index(&inputs[0], &self.indices)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let zeros = graph.add(ConstantLike::new(0., grad));
		let ones = graph.add(ConstantLike::new(1., grad));
		let mask = graph.add(SetIndex::new(zeros, &self.indices, ones));
		vec![graph.add(Mul::new(mask, grad))]
	}
}

#[derive(Debug)]
pub struct SetIndex {
	inp_d: NodeID,
	inp_s: NodeID,
	indices: [Seq],
}

impl SetIndex {
	pub fn new(x: NodeID, indices: [Seq], y: NodeID) -> Box<SetIndex> {
		Box::new(SetIndex {
			inp_d: x,
			inp_s: y,
			indices: indices,
		})
	}
}

impl Node for SetIndex {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_d, self.inp_s]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::assign_seq(&inputs[0], &self.indices, &inputs[1])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let zeros = graph.add(ConstantLike::new(0., grad));
		let ones = graph.add(ConstantLike::new(1., grad));
		let mask = graph.add(SetIndex::new(ones, &self.indices, zeros));
		vec![graph.add(Mul::new(mask, grad))]
	}
}
