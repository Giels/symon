//TODO: Check that imin/imax/nonzero grads are correct
use ::symb::base::{Node, NodeID, NodeData};
use ::symb::graph::{Graph};
use ::symb::ops::utils::{ConstantLike};
use ::symb::ops::math::{Mul, Sub, Add, Div, Neg};

use ::arrayfire::{Dim4, Seq};

use std::fmt;
use std::fmt::{Debug};

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
						let ones = graph.add(ConstantLike::new(1., self.inp));
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
		::arrayfire::mean(&inputs[0], self.dim as i64)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let shape = graph.add(Shape::new(self.inp));
		let denom = graph.add(Index::new(shape, Box::new([Seq::new(self.dim, self.dim, 1)])));
		vec![graph.add(Div::new(grad, denom))]
	}
}

#[derive(Debug)]
pub struct Min {
	inp: NodeID,
	dim: i32,
}

impl Min {
	pub fn new(x: NodeID, dim: i32) -> Box<Min> {
		Box::new(Min {
			inp: x,
			dim: dim,
		})
	}
}

impl Node for Min {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::min(&inputs[0], self.dim)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
	}
}

#[derive(Debug)]
pub struct Eq {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Eq {
	pub fn new(x: NodeID, y: NodeID) -> Box<Eq> {
		Box::new(Eq {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Eq {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::eq(inputs[0], inputs[1], false)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., self.inp_l))]
	}
}
#[derive(Debug)]
pub struct Ge {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Ge {
	pub fn new(x: NodeID, y: NodeID) -> Box<Ge> {
		Box::new(Ge {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Ge {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::ge(inputs[0], inputs[1], false)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., self.inp_l))]
	}
}

#[derive(Debug)]
pub struct Le {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Le {
	pub fn new(x: NodeID, y: NodeID) -> Box<Le> {
		Box::new(Le {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Le {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::le(inputs[0], inputs[1], false)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., self.inp_l))]
	}
}

#[derive(Debug)]
pub struct Gt {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Gt {
	pub fn new(x: NodeID, y: NodeID) -> Box<Gt> {
		Box::new(Gt {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Gt {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::gt(inputs[0], inputs[1], false)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., self.inp_l))]
	}
}

#[derive(Debug)]
pub struct Lt {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Lt {
	pub fn new(x: NodeID, y: NodeID) -> Box<Lt> {
		Box::new(Lt {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Lt {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::lt(inputs[0], inputs[1], false)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., self.inp_l))]
	}
}

#[derive(Debug)]
pub struct Maxof {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Maxof {
	pub fn new(x: NodeID, y: NodeID) -> Box<Maxof> {
		Box::new(Maxof {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Maxof {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::maxof(inputs[0], inputs[1])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let one = graph.add(ConstantLike::new(1., self.inp_l));
		let zero = graph.add(ConstantLike::new(0., self.inp_r));
		let ge = graph.add(Ge::new(self.inp_l, self.inp_r));
		let lt = graph.add(Lt::new(self.inp_l, self.inp_r));
		vec![graph.add(Mul::new(ge, grad)), graph.add(Mul::new(lt, grad))]
	}
}

#[derive(Debug)]
pub struct Minof {
	inp_l: NodeID,
	inp_r: NodeID,
}

impl Minof {
	pub fn new(x: NodeID, y: NodeID) -> Box<Minof> {
		Box::new(Minof {
			inp_l: x,
			inp_r: y,
		})
	}
}

impl Node for Minof {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::minof(inputs[0], inputs[1])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let one = graph.add(ConstantLike::new(1., self.inp_l));
		let zero = graph.add(ConstantLike::new(0., self.inp_r));
		let gt = graph.add(Gt::new(self.inp_l, self.inp_r));
		let le = graph.add(Le::new(self.inp_l, self.inp_r));
		vec![graph.add(Mul::new(le, grad)), graph.add(Mul::new(gt, grad))]
	}
}

#[derive(Debug)]
pub struct Max {
	inp: NodeID,
	dim: i32,
}

impl Max {
	pub fn new(x: NodeID, dim: i32) -> Box<Max> {
		Box::new(Max {
			inp: x,
			dim: dim,
		})
	}
}

impl Node for Max {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::max(&inputs[0], self.dim)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
	}
}

#[derive(Debug)]
pub struct Argmax {
	inp: NodeID,
	dim: i32,
}

impl Argmax {
	pub fn new(x: NodeID, dim: i32) -> Box<Argmax> {
		Box::new(Argmax {
			inp: x,
			dim: dim,
		})
	}
}

impl Node for Argmax {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::imax(&inputs[0], self.dim).1
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
	}
}

#[derive(Debug)]
pub struct Argmin {
	inp: NodeID,
	dim: i32,
}

impl Argmin {
	pub fn new(x: NodeID, dim: i32) -> Box<Argmin> {
		Box::new(Argmin {
			inp: x,
			dim: dim,
		})
	}
}

impl Node for Argmin {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::imin(&inputs[0], self.dim).1
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
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
	dim: i32,
}

impl Stdev {
	pub fn new(x: NodeID, dim: i32) -> Box<Stdev> {
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
		::arrayfire::stdev(&inputs[0], self.dim as i64)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let half = graph.add(ConstantLike::new(0.5, this));
		let efx = graph.add(Mean::new(self.inp, self.dim));
		let shape = graph.add(Shape::new(self.inp));
		let n = graph.add(Index::new(shape, Box::new([Seq::new(self.dim, self.dim, 1)])));
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
	dim: i32,
}

impl Variance {
	pub fn new(x: NodeID, dim: i32) -> Box<Variance> {
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
		::arrayfire::var(&inputs[0], true, self.dim as i64) // sample variance
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap_or(graph.add(ConstantLike::new(1., self.inp)));
		let efx = graph.add(Mean::new(self.inp, self.dim));
		let shape = graph.add(Shape::new(self.inp));
		let n = graph.add(Index::new(shape, Box::new([Seq::new(self.dim, self.dim, 1)])));
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
		let d = inputs[0].dims();
		let slice = d.get();
		let dims = Dim4::new(&[4, 1, 1, 1]);
		::arrayfire::Array::new(slice, dims)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
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
		let node = ConstantLike::new(0., this);
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct DynIndex {
	inp: NodeID,
	dim: u32,
	indices: [NodeID; 3],
}

impl DynIndex {
	pub fn new(x: NodeID, dim: u32, indices: [NodeID; 3]) -> Box<DynIndex> {
		Box::new(DynIndex {
			inp: x,
			indices: indices,
			dim: dim,
		})
	}
}

impl Node for DynIndex {
	fn get_inputs(&self) -> Vec<NodeID> {
		let mut ret = vec![self.inp];
		ret.extend(&self.indices);
		ret
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		use ::arrayfire::sum;
		let mut seqs: Box<[Seq<i32>]> = Box::new([Seq::default(), Seq::default(), Seq::default(), Seq::default()]);
		let mut start = vec![0; 1];
		let mut stop = vec![0; 1];
		let mut step = vec![0; 1];
		inputs[1].host(&mut start);
		inputs[2].host(&mut stop);
		inputs[3].host(&mut step);
		(*seqs)[self.dim as usize] = Seq::new(start[0], stop[0], step[0]);
		::arrayfire::index(&inputs[0], &*seqs)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let zeros = graph.add(ConstantLike::new(0., grad));
		let ones = graph.add(ConstantLike::new(1., grad));
		let mask = graph.add(SetDynIndex::new(zeros, self.dim, self.indices, ones));
		let zero = graph.add(ConstantLike::new(0., self.indices[0]));
		vec![graph.add(Mul::new(mask, grad)), zero, zero, zero]
	}
}

#[derive(Debug)]
pub struct SetDynIndex {
	inp_d: NodeID,
	inp_s: NodeID,
	dim: u32,
	indices: [NodeID; 3],
}

impl SetDynIndex {
	pub fn new(x: NodeID, dim: u32, indices: [NodeID; 3], y: NodeID) -> Box<SetDynIndex> {
		Box::new(SetDynIndex {
			inp_d: x,
			inp_s: y,
			dim: dim,
			indices: indices,
		})
	}
}

impl Node for SetDynIndex {
	fn get_inputs(&self) -> Vec<NodeID> {
		let mut ret = vec![self.inp_d, self.inp_s];
		ret.extend(&self.indices);
		ret
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let mut seqs: Box<[Seq<i32>]> = Box::new([Seq::default(), Seq::default(), Seq::default(), Seq::default()]);
		let mut start = vec![0; 1];
		let mut stop = vec![0; 1];
		let mut step = vec![0; 1];
		inputs[2].host(&mut start);
		inputs[3].host(&mut stop);
		inputs[4].host(&mut step);
		(*seqs)[self.dim as usize] = Seq::new(start[0], stop[0], step[0]);
		::arrayfire::assign_seq(inputs[0], &*seqs, inputs[1])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let zeros = graph.add(ConstantLike::new(0., grad));
		let ones = graph.add(ConstantLike::new(1., grad));
		let mask = graph.add(SetDynIndex::new(ones, self.dim, self.indices, zeros));
		let unmask = graph.add(SetDynIndex::new(zeros, self.dim, self.indices, ones));
		let zero = graph.add(ConstantLike::new(0., self.indices[0]));
		vec![graph.add(Mul::new(mask, grad)), graph.add(Mul::new(unmask, grad)), zero, zero, zero]
	}
}

pub struct Index {
	inp: NodeID,
	indices: Box<[Seq<i32>]>,
}

impl Debug for Index {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Index(...)")
	}
}

impl Index {
	pub fn new(x: NodeID, indices: Box<[Seq<i32>]>) -> Box<Index> {
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
		let mask = graph.add(SetIndex::new(zeros, self.indices.clone(), ones));
		vec![graph.add(Mul::new(mask, grad))]
	}
}

pub struct SetIndex {
	inp_d: NodeID,
	inp_s: NodeID,
	indices: Box<[Seq<i32>]>,
}

impl Debug for SetIndex {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "SetIndex(...)")
	}
}

impl SetIndex {
	pub fn new(x: NodeID, indices: Box<[Seq<i32>]>, y: NodeID) -> Box<SetIndex> {
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
		let mask = graph.add(SetIndex::new(ones, self.indices.clone(), zeros));
		let unmask = graph.add(SetIndex::new(zeros, self.indices.clone(), ones));
		vec![graph.add(Mul::new(mask, grad)), graph.add(Mul::new(unmask, grad))]
	}
}

#[derive(Debug)]
pub struct If {
	cond: NodeID,
	yes: NodeID,
	no: NodeID,
}

impl If {
	pub fn new(c: NodeID, x: NodeID, y: NodeID) -> Box<If> {
		Box::new(If {
			cond: c,
			yes: x,
			no: y,
		})
	}
}

impl Node for If {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.cond, self.yes, self.no]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::select(&inputs[0], &inputs[1], &inputs[2])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		// select is equivalent to `c * yes + (1-c) * no` for binary `c`
		// thus, grad wrt `c` is `yes - no`, wrt `yes` is `c`, and wrt `no` is `1-c`.
		let grad = g.unwrap();

		let dc = graph.add(Sub::new(self.yes, self.no));
		let dy = self.cond;
		let one = graph.add(ConstantLike::new(1., self.cond));
		let dn = graph.add(Sub::new(one, self.cond));
		vec![graph.add(Mul::new(dc, grad)), graph.add(Mul::new(dy, grad)), graph.add(Mul::new(dn, grad))]
	}
}

#[derive(Debug)]
pub struct Join {
	inp_l: NodeID,
	inp_r: NodeID,
	dim: u32,
}

impl Join {
	pub fn new(x: NodeID, y: NodeID, dim: u32) -> Box<Join> {
		Box::new(Join {
			inp_l: x,
			inp_r: y,
			dim: dim,
		})
	}
}

impl Node for Join {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp_l, self.inp_r]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::join(self.dim as i32, inputs[0], inputs[1])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let grad = g.unwrap();
		let shape = graph.add(Shape::new(self.inp_l));
		let s = graph.add(Index::new(shape, Box::new([Seq::new(self.dim as i32, self.dim as i32, 1)])));
		let zero = graph.add(ConstantLike::new(0., s));
		let one = graph.add(ConstantLike::new(1., s));
		let split = graph.add(DynIndex::new(grad, self.dim, [zero, s, one]));
		vec![graph.add(Index::new(split, Box::new([Seq::new(0, 0, 1)]))), graph.add(Index::new(split, Box::new([Seq::new(1, 1, 1)])))]
	}
}

#[derive(Debug)]
pub struct Reshape {
	inp: NodeID,
	shape: NodeID,
}

impl Reshape {
	pub fn new(x: NodeID, shape: NodeID) -> Box<Reshape> {
		Box::new(Reshape {
			inp: x,
			shape: shape,
		})
	}
}

impl Node for Reshape {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp, self.shape]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let mut dims: [u64; 4] = [1; 4];
		inputs[1].host(&mut dims);
		::arrayfire::moddims(&inputs[0], Dim4::new(&dims))
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let shape = graph.add(Shape::new(self.inp));
		let node = Reshape::new(g.unwrap(), shape);
		let zeros = graph.add(ConstantLike::new(0., self.shape));
		vec![graph.add(node), zeros]
	}
}

#[derive(Debug)]
pub struct Flatten {
	inp: NodeID,
}

impl Flatten {
	pub fn new(x: NodeID) -> Box<Flatten> {
		Box::new(Flatten {
			inp: x,
		})
	}
}

impl Node for Flatten {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::flat(&inputs[0])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let shape = graph.add(Shape::new(self.inp));
		let node = Reshape::new(g.unwrap(), shape);
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct Reorder {
	inp: NodeID,
	order: Dim4,
}

impl Reorder {
	pub fn new(x: NodeID, order: Dim4) -> Box<Reorder> {
		Box::new(Reorder {
			inp: x,
			order: order,
		})
	}
}

impl Node for Reorder {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::reorder(&inputs[0], self.order)
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let order = self.order.get();
		let mut unorder = order.clone();
		for i in 0..4 {
			for (j, k) in order.iter().enumerate() {
				if *k == i {
					unorder[i as usize] = j as u64;
				}
			}
		}
		let node = Reorder::new(g.unwrap(), Dim4::new(&unorder));
		vec![graph.add(node)]
	}
}

#[derive(Debug)]
pub struct Tile {
	inp: NodeID,
	shape: NodeID,
}

impl Tile {
	pub fn new(x: NodeID, shape: NodeID) -> Box<Tile> {
		Box::new(Tile {
			inp: x,
			shape: shape,
		})
	}
}

impl Node for Tile {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp, self.shape]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		let mut dims: [u64; 4] = [1; 4];
		inputs[1].host(&mut dims);
		::arrayfire::tile(&inputs[0], Dim4::new(&dims))
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		let shape = graph.add(Shape::new(self.inp));
		// XXX: does this work?
		let node = Tile::new(g.unwrap(), shape);
		let zeros = graph.add(ConstantLike::new(0., self.shape));
		vec![graph.add(node), zeros]
	}
}

#[derive(Debug)]
pub struct Nonzero {
	inp: NodeID,
}

impl Nonzero {
	pub fn new(x: NodeID) -> Box<Nonzero> {
		Box::new(Nonzero {
			inp: x,
		})
	}
}

impl Node for Nonzero {
	fn get_inputs(&self) -> Vec<NodeID> {
		vec![self.inp]
	}

	fn eval(&self, inputs: Vec<&NodeData>) -> NodeData {
		::arrayfire::locate(&inputs[0])
	}

	fn backward(&self, this: NodeID, g: Option<NodeID>, graph: &mut Graph) -> Vec<NodeID> {
		vec![graph.add(ConstantLike::new(0., this))]
	}
}
