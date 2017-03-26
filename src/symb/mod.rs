mod base;
mod graph;
mod ops;
mod core;

pub use self::base::*;
pub use self::graph::*;
pub use self::ops::*;
pub use self::core::*;

macro_rules! test_imports {
	() => {
		use ::arrayfire::{Dim4, Seq, all_true_all, max_all, le, ge, eq};
	}
}

macro_rules! gen_test {
	( $symb:expr, $af:expr ) => {
		setup();

		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let x = graph.add(Var::new_shared(xval.clone()));
		let y = graph.add(Var::new_shared(yval.clone()));
		let xpy = graph.add($symb(x, y));
		let eval = graph.eval(xpy);

		// Simply check if the grad can be taken
		let o = graph.add(SumAll::new(xpy));
		let do_x = graph.grad(o, vec![x]);
		let do_x_eval = graph.eval(do_x[0]);
		let do_y = graph.grad(o, vec![y]);
		let do_y_eval = graph.eval(do_y[0]);

		assert_eq!(all_true_all(&eq(&eval, &$af(&xval, &yval, false), false)).0, 1.);
	}
}

macro_rules! gen_test_diff {
	( $symb: expr, $af:expr ) => {
		setup();
		let slack = 1e-1;
		let eps = 6e-4;

		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let x = graph.add(Var::new_shared(xval.clone()));
		let y = graph.add(Var::new_shared(yval.clone()));
		let xpy = graph.add($symb(x, y));
		let o = graph.add(SumAll::new(xpy));

		let do_x = graph.grad(o, vec![x]);
		let do_x_eval = graph.eval(do_x[0]);
		let do_y = graph.grad(o, vec![y]);
		let do_y_eval = graph.eval(do_y[0]);

		let do_x_ref = ($af(&(&xval + eps / 2.), &yval, false) - $af(&(&xval - eps / 2.), &yval, false)) / eps;
		let do_y_ref = ($af(&xval, &(&yval + eps / 2.), false) - $af(&xval, &(&yval - eps / 2.), false)) / eps;

		let mx = (max_all(&do_x_ref).0).abs();
		let my = (max_all(&do_y_ref).0).abs();
		assert_eq!(all_true_all(&::arrayfire::and(&ge(&(&do_x_ref + slack * mx), &do_x_eval, false), &le(&(&do_x_ref - slack * mx), &do_x_eval, false), false)).0, 1.);
		assert_eq!(all_true_all(&::arrayfire::and(&ge(&(&do_y_ref + slack * my), &do_y_eval, false), &le(&(&do_y_ref - slack * my), &do_y_eval, false), false)).0, 1.);
	}
}

macro_rules! gen_dim_test {
	( $symb:expr, $af:expr, $dim:expr ) => {
		setup();

		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let x = graph.add(Var::new_shared(xval.clone()));
		let fx = graph.add($symb(x, $dim));
		let eval = graph.eval(fx);

		// Simply check if the grad can be taken
		let o = graph.add(SumAll::new(fx));
		let do_x = graph.grad(o, vec![x]);
		let do_x_eval = graph.eval(do_x[0]);

		assert_eq!(all_true_all(&eq(&eval, &$af(&xval, $dim), false)).0, 1.);
	}
}

macro_rules! gen_dim_test_diff {
	( $symb: expr, $af:expr, $dim:expr ) => {
		setup();
		let slack = 1e-1;
		let eps = 6e-4;

		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let x = graph.add(Var::new_shared(xval.clone()));
		let fx = graph.add($symb(x, $dim));
		let o = graph.add(SumAll::new(fx));

		let do_x = graph.grad(o, vec![x]);
		let do_x_eval = graph.eval(do_x[0]);

		let do_x_ref = ($af(&(&xval + eps / 2.), $dim) - $af(&(&xval - eps / 2.), $dim)) / eps;

		let mx = (max_all(&do_x_ref).0).abs();
		let greater = &ge(&(&do_x_ref + slack * mx), &do_x_eval, false);
		let lesser = &le(&(&do_x_ref - slack * mx), &do_x_eval, false);
		assert_eq!(all_true_all(&::arrayfire::and(greater, lesser, false)).0, 1.);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn setup() {
		::arrayfire::set_backend(::arrayfire::Backend::CPU);
	}

	#[test]
	fn test_vars() {
		setup();
		use ::arrayfire::{all_true_all, eq, Dim4};

		let mut graph = Graph::new();

		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[1, 5, 1, 1]));
		let x = graph.add(Var::new());
		let y = graph.add(Var::new_shared(yval.clone()));
		graph.replace(x, Var::new_shared(xval.clone()));

		let eval_x = graph.eval(x);
		let eval_y = graph.eval(y);

		assert_eq!(all_true_all(&eq(&eval_x, &xval, false)).0, 1.);
		assert_eq!(all_true_all(&eq(&eval_y, &yval, false)).0, 1.);
	}

	#[test]
	fn test_sumall() {
		setup();
		use ::arrayfire::{Dim4, sum_all};

		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
	    let x = graph.add(Var::new_shared(xval.clone()));
		let sx = graph.add(SumAll::new(x));
		let eval = graph.eval(sx);

		assert_eq!(sum_all(&eval).0, sum_all(&xval).0);
	}

	#[test]
	fn test_add() {
		test_imports!();
		gen_test!(Add::new, ::arrayfire::add);
	}

	#[test]
	fn test_add_diff() {
		test_imports!();
		gen_test_diff!(Add::new, ::arrayfire::add);
	}

	#[test]
	fn test_mul() {
		test_imports!();
		gen_test!(Mul::new, ::arrayfire::mul);
	}

	#[test]
	fn test_mul_diff() {
		test_imports!();
		gen_test_diff!(Mul::new, ::arrayfire::mul);
	}

	#[test]
	fn test_div() {
		test_imports!();
		gen_test!(Div::new, ::arrayfire::div);
	}

	#[test]
	fn test_div_diff() {
		test_imports!();
		gen_test_diff!(Div::new, ::arrayfire::div);
	}

	#[test]
	fn test_sub() {
		test_imports!();
		gen_test!(Sub::new, ::arrayfire::sub);
	}

	#[test]
	fn test_sub_diff() {
		test_imports!();
		gen_test_diff!(Sub::new, ::arrayfire::sub);
	}

	#[test]
	fn test_conv() {
		setup();
		use ::arrayfire::{ConvDomain, ConvMode};
		use ::arrayfire::{Dim4, all_true_all, eq};
	
		let mut graph = Graph::new();
		let dims = [64, 32, 16, 8];
		for i in (0..3) {
			let mut these_dims = [1, 1, 1, 1];
			for j in (0..i+1) {
				these_dims[j] = dims[j];
			}
			let af_conv = match i {
							3 => ::arrayfire::convolve3,
							2 => ::arrayfire::convolve2,
							_ => ::arrayfire::convolve1,
						};
			let xval = ::arrayfire::randn::<f32>(Dim4::new(&these_dims));
			let yval = ::arrayfire::randn::<f32>(Dim4::new(&these_dims));
			let x = graph.add(Var::new_shared(xval.clone()));
			let y = graph.add(Var::new_shared(yval.clone()));
			let xpy = graph.add(Conv::new(x, y, false));
			let eval = graph.eval(xpy);
			let ref_val = af_conv(&xval, &yval, ConvMode::DEFAULT, ConvDomain::AUTO);

			assert_eq!(all_true_all(&eq(&eval, &ref_val, false)).0, 1.);
		}
	}

	#[test]
	fn test_conv_diff() {
		//TODO: ConvMode::EXPAND
		//TODO: more dimensions for kernel
		setup();
		use ::arrayfire::{ConvDomain, ConvMode};
		use ::arrayfire::{Dim4, all_true_all, le, ge, max_all};
		let slack = 1e-2;
		let eps = 6e-4;
		
		let mut graph = Graph::new();
		let dims = [64, 32, 16, 8];
		for i in (0..3) {
			let mut inp_dims = [1, 1, 1, 1];
			for j in (0..i+1) {
				inp_dims[j] = dims[j];
			}
			let filt_dims = inp_dims.clone();
			let af_conv = match i {
							3 => ::arrayfire::convolve3,
							2 => ::arrayfire::convolve2,
							_ => ::arrayfire::convolve1,
						};
			let xval = ::arrayfire::randn::<f32>(Dim4::new(&inp_dims));
			let yval = ::arrayfire::randn::<f32>(Dim4::new(&filt_dims));
			let x = graph.add(Var::new_shared(xval.clone()));
			let y = graph.add(Var::new_shared(yval.clone()));
			let xpy = graph.add(Conv::new(x, y, false));
			let o = graph.add(SumAll::new(xpy));
			
			let do_x = graph.grad(o, vec![x]);
			let do_x_eval = graph.eval(do_x[0]);
			let do_y = graph.grad(o, vec![y]);
			let do_y_eval = graph.eval(do_y[0]);

			let do_x_ref = (af_conv(&(&xval + eps), &yval, ConvMode::DEFAULT, ConvDomain::AUTO)
							- af_conv(&xval, &yval, ConvMode::DEFAULT, ConvDomain::AUTO)) / eps;
			let do_y_ref = (af_conv(&xval, &(&yval + eps), ConvMode::DEFAULT, ConvDomain::AUTO)
							- af_conv(&xval, &yval, ConvMode::DEFAULT, ConvDomain::AUTO)) / eps;

			let mx = (max_all(&do_x_ref).0).abs();
			let my = (max_all(&do_y_ref).0).abs();

			assert_eq!(all_true_all(&::arrayfire::and(&ge(&(&do_x_ref + slack * mx), &do_x_eval, false), &le(&(&do_x_ref - slack * mx), &do_x_eval, false), false)).0, 1.);
			assert_eq!(all_true_all(&::arrayfire::and(&ge(&(&do_y_ref + slack * my), &do_y_eval, false), &le(&(&do_y_ref - slack * my), &do_y_eval, false), false)).0, 1.);
		}
	}

	#[test]
	fn test_affine() {
		setup();
		use ::arrayfire::{Dim4, matmul, add, sum_all, MatProp};

		let mut graph = Graph::new();

		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let wval = ::arrayfire::randn::<f32>(Dim4::new(&[3, 5, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[1, 5, 1, 1]));

		let x = graph.add(Var::new());
		let y = graph.add(Var::new_shared(yval.clone()));
		let w = graph.add(Var::new_shared(wval.clone()));

		let wx = graph.add(MatMul::new(x, w));
		let wxpy = graph.add(Add::new(wx, y));

		let z = graph.add(SumAll::new(wxpy));

		graph.replace(x, Var::new_shared(xval.clone()));

		let eval = graph.eval(z);

		let af_z = sum_all(&add(&matmul(&xval, &wval, MatProp::NONE, MatProp::NONE), &yval, true)).0;

		assert_eq!(sum_all(&eval).0, af_z);
	}

	#[test]
	fn test_affine_diff() {
		setup();
		use ::arrayfire::{Dim4, matmul, MatProp, eq, all_true_all, transpose, constant};

		let mut graph = Graph::new();

		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let wval = ::arrayfire::randn::<f32>(Dim4::new(&[3, 5, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[1, 5, 1, 1]));

		let x = graph.add(Var::new());
		let y = graph.add(Var::new_shared(yval.clone()));
		let w = graph.add(Var::new_shared(wval.clone()));

		let wx = graph.add(MatMul::new(x, w));
		let wxpy = graph.add(Add::new(wx, y));

		let z = graph.add(SumAll::new(wxpy));

		graph.replace(x, Var::new_shared(xval.clone()));

		let dz_w = graph.grad(z, vec![w]);
		let dz_w_eval = graph.eval(dz_w[0]);
		let cst = constant::<f32>(1., Dim4::new(&[16, 5, 1, 1]));
		let dz_w_ref = matmul(&transpose(&xval, false), &cst, MatProp::NONE, MatProp::NONE);
		assert_eq!(all_true_all(&eq(&dz_w_ref, &dz_w_eval, false)).0, 1.);

		let dz_y = graph.grad(z, vec![y]);
		let dz_y_eval = graph.eval(dz_y[0]);

		// NOTE: the `y` is broadcasted but that fact is not being tracked yet, so
		//right now the correct behavior is to return a differential that's fully-sized.
		//eventually, we should ensure the size is the same as the real input's.
		let dz_y_ref = constant::<f32>(1., Dim4::new(&[16, 5, 1, 1]));
		assert_eq!(all_true_all(&eq(&dz_y_ref, &dz_y_eval, false)).0, 1.);

		let dz_x = graph.grad(z, vec![x]);
		let dz_x_eval = graph.eval(dz_x[0]);
		let cst = constant::<f32>(1., Dim4::new(&[16, 5, 1, 1]));
		let dz_x_ref = matmul(&cst, &transpose(&wval, false), MatProp::NONE, MatProp::NONE);
		assert_eq!(all_true_all(&eq(&dz_x_ref, &dz_x_eval, false)).0, 1.);
	}

	#[test]
	fn test_maxof() {
		test_imports!();
		gen_test!(Maxof::new, ::arrayfire::maxof);
	}

	#[test]
	fn test_maxof_diff() {
		test_imports!();
		gen_test_diff!(Maxof::new, ::arrayfire::maxof);
	}

	#[test]
	fn test_minof() {
		test_imports!();
		gen_test!(Minof::new, ::arrayfire::minof);
	}

	#[test]
	fn test_minof_diff() {
		test_imports!();
		gen_test_diff!(Minof::new, ::arrayfire::minof);
	}

	#[test]
	fn test_eq() {
		test_imports!();
		gen_test!(Eq::new, ::arrayfire::eq);
	}

	#[test]
	fn test_eq_diff() {
		test_imports!();
		gen_test_diff!(Eq::new, ::arrayfire::eq);
	}

	#[test]
	fn test_lt() {
		test_imports!();
		gen_test!(Lt::new, ::arrayfire::lt);
	}

	#[test]
	fn test_lt_diff() {
		test_imports!();
		gen_test_diff!(Lt::new, ::arrayfire::lt);
	}

	#[test]
	fn test_le() {
		test_imports!();
		gen_test!(Le::new, ::arrayfire::le);
	}

	#[test]
	fn test_le_diff() {
		test_imports!();
		gen_test_diff!(Le::new, ::arrayfire::le);
	}

	#[test]
	fn test_gt() {
		test_imports!();
		gen_test!(Gt::new, ::arrayfire::gt);
	}

	#[test]
	fn test_gt_diff() {
		test_imports!();
		gen_test_diff!(Gt::new, ::arrayfire::gt);
	}

	#[test]
	fn test_ge() {
		test_imports!();
		gen_test!(Ge::new, ::arrayfire::ge);
	}

	#[test]
	fn test_ge_diff() {
		test_imports!();
		gen_test_diff!(Ge::new, ::arrayfire::ge);
	}

	#[test]
	fn test_sum() {
		test_imports!();
		gen_dim_test!(Sum::new, ::arrayfire::sum, 0);
		gen_dim_test!(Sum::new, ::arrayfire::sum, 1);
	}

	// #[test]
	// fn test_sum_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Sum::new, ::arrayfire::sum, 0);
	// 	gen_dim_test_diff!(Sum::new, ::arrayfire::sum, 1);
	// }

	#[test]
	fn test_mean() {
		test_imports!();
		gen_dim_test!(Mean::new, ::arrayfire::mean, 0);
		gen_dim_test!(Mean::new, ::arrayfire::mean, 1);
	}

	// #[test]
	// fn test_mean_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Mean::new, ::arrayfire::mean, 0);
	// 	gen_dim_test_diff!(Mean::new, ::arrayfire::mean, 1);
	// }

	#[test]
	fn test_min() {
		test_imports!();
		gen_dim_test!(Min::new, ::arrayfire::min, 0);
		gen_dim_test!(Min::new, ::arrayfire::min, 1);
	}

	// #[test]
	// fn test_min_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Min::new, ::arrayfire::min, 0);
	// 	gen_dim_test_diff!(Min::new, ::arrayfire::min, 1);
	// }

	#[test]
	fn test_max() {
		test_imports!();
		gen_dim_test!(Max::new, ::arrayfire::max, 0);
		gen_dim_test!(Max::new, ::arrayfire::max, 1);
	}

	// #[test]
	// fn test_max_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Max::new, ::arrayfire::max, 0);
	// 	gen_dim_test_diff!(Max::new, ::arrayfire::max, 1);
	// }

	#[test]
	fn test_argmin() {
		test_imports!();
		gen_dim_test!(Argmin::new, |x, y| ::arrayfire::imin(x, y).1, 0);
		gen_dim_test!(Argmin::new, |x, y| ::arrayfire::imin(x, y).1, 1);
	}

	#[test]
	fn test_argmin_diff() {
		test_imports!();
		gen_dim_test_diff!(Argmax::new, |x, y| ::arrayfire::imin(x, y).1, 0);
		gen_dim_test_diff!(Argmax::new, |x, y| ::arrayfire::imin(x, y).1, 1);
	}

	#[test]
	fn test_argmax() {
		test_imports!();
		gen_dim_test!(Argmax::new, |x, y| ::arrayfire::imax(x, y).1, 0);
		gen_dim_test!(Argmax::new, |x, y| ::arrayfire::imax(x, y).1, 1);
	}

	#[test]
	fn test_argmax_diff() {
		test_imports!();
		gen_dim_test_diff!(Argmax::new, |x, y| ::arrayfire::imax(x, y).1, 0);
		gen_dim_test_diff!(Argmax::new, |x, y| ::arrayfire::imax(x, y).1, 1);
	}

	#[test]
	fn test_stdev() {
		test_imports!();
		gen_dim_test!(Stdev::new, ::arrayfire::stdev, 0);
		gen_dim_test!(Stdev::new, ::arrayfire::stdev, 1);
	}

	// #[test]
	// fn test_stdev_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Stdev::new, ::arrayfire::stdev, 0);
	// 	gen_dim_test_diff!(Stdev::new, ::arrayfire::stdev, 1);
	// }

	#[test]
	fn test_variance() {
		test_imports!();
		gen_dim_test!(Variance::new, |x, y| ::arrayfire::var(x, true, y), 0);
		gen_dim_test!(Variance::new, |x, y| ::arrayfire::var(x, true, y), 1);
	}

	// #[test]
	// fn test_variance_diff() {
	// 	test_imports!();
	// 	gen_dim_test_diff!(Variance::new, |x, y| ::arrayfire::var(x, true, y), 0);
	// 	gen_dim_test_diff!(Variance::new, |x, y| ::arrayfire::var(x, true, y), 1);
	// }

	#[test]
	fn test_flip() {
		test_imports!();
		gen_dim_test!(Flip::new, ::arrayfire::flip, 0);
		gen_dim_test!(Flip::new, ::arrayfire::flip, 1);
	}

	#[test]
	fn test_flip_diff() {
		test_imports!();
		gen_dim_test_diff!(Flip::new, ::arrayfire::flip, 0);
		gen_dim_test_diff!(Flip::new, ::arrayfire::flip, 1);
	}

	#[test]
	fn test_shape() {
		test_imports!();
		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
	    let x = graph.add(Var::new_shared(xval.clone()));
		let shape = graph.add(Shape::new(x));
		let eval = graph.eval(shape);
	
		assert_eq!(all_true_all(&eq(&eval, &::arrayfire::Array::new(xval.dims().get(), Dim4::new(&[2, 1, 1, 1])), false)).0, 1.);
	}

	#[test]
	fn test_index() {
		test_imports!();
		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
	    let x = graph.add(Var::new_shared(xval.clone()));
		let idx = Box::new([Seq::new(0, 0, 1), Seq::default(), Seq::new(0, 0, 1), Seq::new(0, 0, 1)]);
		let indexed = graph.add(Index::new(x, idx.clone()));
		let eval = graph.eval(indexed);
	
		assert_eq!(all_true_all(&eq(&eval, &::arrayfire::index(&xval, &*idx), false)).0, 1.);
	}

	#[test]
	fn test_set_index() {
		test_imports!();
		let mut graph = Graph::new();
		let xval = ::arrayfire::randn::<f32>(Dim4::new(&[16, 3, 1, 1]));
		let yval = ::arrayfire::randn::<f32>(Dim4::new(&[1, 3, 1, 1]));
	    let x = graph.add(Var::new_shared(xval.clone()));
		let y = graph.add(Var::new_shared(yval.clone()));
		let idx = Box::new([Seq::new(0, 0, 1), Seq::default()]);
		let indexed = graph.add(SetIndex::new(x, idx.clone(), y));
		let eval = graph.eval(indexed);
	
		assert_eq!(all_true_all(&eq(&eval, &::arrayfire::assign_seq(&xval, &*idx, &yval), false)).0, 1.);
	}
}
