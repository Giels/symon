use std::collections::{HashMap, HashSet};

use ::symb::base::{Node, NodeID, NodeData};
use ::symb::ops::{Add};

use std::cell::{RefCell};
use std::rc::{Rc};

#[derive(Debug)]
pub struct Graph {
    edges: HashMap<NodeID, Vec<NodeID>>,
	roots: HashMap<NodeID, HashSet<NodeID>>,
    nodes: Vec<Rc<RefCell<Box<Node>>>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            edges: HashMap::new(),
            nodes: vec![],
			roots: HashMap::new(),
        }
    }

	pub fn replace(&mut self, node: NodeID, mut new_node: Box<Node>) {
		self.prepare_node(&mut new_node, node);
		*(self.nodes[node as usize].borrow_mut()) = new_node;
	}

    fn connect(&mut self, node: NodeID, mut inp: Vec<NodeID>) {
        let nodes = self.edges.entry(node).or_insert(Vec::new());
		nodes.append(&mut inp);
    }

	fn prepare_node(&mut self, node: &mut Box<Node>, id: NodeID) {
		let mut root_set = HashSet::new();
		let inputs = node.get_inputs();

		for i in inputs.iter() {
			for r in self.roots[i].iter() {
				root_set.insert(r.clone());
			}
			root_set.insert(*i);
		}

		self.roots.insert(id, root_set);

		if inputs.len() == 0 {
			let root = self.roots.get_mut(&id).unwrap();
			root.insert(id);
		}
	}
   
    pub fn add(&mut self, mut node: Box<Node>) -> NodeID {
        let id = (self.nodes.len()) as NodeID;

		self.prepare_node(&mut node, id);

		let inputs = node.get_inputs();
        self.nodes.push(Rc::new(RefCell::new(node)));

        self.connect(id, inputs);

        id
    }
   
    pub fn eval(&self, node: NodeID) -> NodeData {
        let mut vals = vec![None; self.nodes.len()];
        let mut path = vec![node];
       
        loop {
            let target = path.pop().unwrap();
           
            if self.edges[&target].len() == 0 {
                vals[target as usize] = Some(self.nodes[target as usize].borrow().eval(vec![]));
            } else {
                let mut pushed = false;
               
                for e in self.edges[&target].iter() {
                    if vals[*e as usize].is_none() {
                        if !pushed {
                            path.push(target);
                            pushed = true;
                        }
                        path.push(*e);
                    }
                }
               
                if !pushed {
                    vals[target as usize] = Some({
                    	let inputs = self.edges[&target].iter().map(|i| vals[*i as usize].as_ref().unwrap()).collect::<Vec<&NodeData>>();
						self.nodes[target as usize].borrow().eval(inputs)
					});
                }
            }
           
            if path.len() == 0 {
                break;
			}
        }
       
        vals.swap_remove(node as usize).unwrap()
    }

    pub fn grad(&mut self, node: NodeID, wrt: Vec<NodeID>) -> Vec<NodeID> {
		let mut grads = vec![];

		for w in wrt.iter() {
			let mut w_grads = vec![];

			let mut path = vec![node];
			let mut work_g = vec![None];
			let mut g = vec![];

			loop {
				if path.len() == 0 {
					w_grads.append(&mut g);
					break;
				}

				let target = path.pop().unwrap();
				let this_g = work_g.pop().unwrap();

				let mut inps: Vec<NodeID> = self.nodes[target as usize].borrow().get_inputs();
				let node = self.nodes[target as usize].clone();
				let nb = node.borrow();

				let mut all_new_g: Vec<_> = nb.backward(target, this_g, self).into_iter().map(|x| Some(x)).collect();

				if inps.len() != 0 {
					path.append(&mut inps);
					work_g.append(&mut all_new_g);
				} else {
					if *w == target {
						g.push(this_g.unwrap());
					}
				}
			}

			grads.push(w_grads.iter()
								.fold(None,
										|prev, x| prev
											.and_then(|p|
														Some(self.add(Add::new(p, *x))))
											.or(Some(*x))
									).unwrap());
		}

		grads
	}
}
 
