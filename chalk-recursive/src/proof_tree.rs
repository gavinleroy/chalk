//! Proof tree for solving a goal

use crate::UCanonicalGoal;
use chalk_ir::{interner::Interner, *};
use chalk_solve::infer::{InferenceTable, ParameterEnaVariableExt};
use chalk_solve::{Guidance, Solution};

use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::usize;
use tracing::{debug, instrument};

use index_vec::{index_vec, IndexSlice, IndexVec};

use crate::fixed_point::{search_graph::SearchGraph, stack::Stack};

// ------------------------------------------------------------------
// FIXME: I don't know what I'm doing yet, these aren't suitable
//        for actual tracing ...
//
//        One of the things we need is data along a tree edge to
//        know "why" something happened, and the data we are storing
//        currently is not helpful and very wasteful.
// ------------------------------------------------------------------

// XXX ---------
//
// The ideal structure for a "ProofTree" is one that holds a
// Goal in the nodes and concrete answers in the leaves.
// E.g.
//
// ```ignore
// [member(X,[a,b,c])]
//       /   \
//  X=a /     \X=X’
//     /       \
//    []      [member(X’,[b,c])]
//              /   \
//       X’= b /     \X’=X’’
//            /       \
//           []      [member(X’’,[c])]
//                     /    \
//              X’’=c /      \X’’=X’’’
//                   /        \
//                  []      [member(X’’’,[])]
//                               /  \
//                              /    \
//                             x      x
// ```
//
// We can see that the edges represent substitutions, as the goals
// become more concrete. Rules who match the head generate multiple
// obligations for a given goal node to be solved.

// --------------------
// Functional structure

pub trait HasFail {
    fn fail() -> Self;
}

// TODO reduce the amount of code by creating a Layer trait
//      and using a function with_layer(IntoLayer<I>) to
//      add information to an existing tree.

#[derive(Clone, Debug, PartialEq)]
pub enum ProofTree<I: Interner> {
    // XXX: Use this to introduce "new information" into the tree
    //      we use this for learning information about the environemtn
    //      but also to specify new subgoals.
    Introducing(EdgeInfo<I>, Box<ProofTree<I>>),
    Nested(NestedNode<I>),
    Leaf(LeafNode<I>),
    Halted,
}

macro_rules! can_unify {
    ($($t:ident,)*) => {
        #[derive(Clone, Debug, PartialEq)]
        pub enum UnifyKind<I: Interner> {
            $( $t($t<I>, $t<I>), ) *
        }

        $(
        impl<I: Interner> From<(Environment<I>, Variance, $t<I>, $t<I>)> for EdgeInfo<I> {
            fn from((env, var, a, b): (Environment<I>, Variance, $t<I>, $t<I>)) -> Self {
                EdgeInfo::Unification {
                    environment: env,
                    variance: var,
                    kind: UnifyKind::$t(a, b)
                }
            }
        }

        impl<I: Interner> From<(Environment<I>, Variance, &$t<I>, &$t<I>)> for EdgeInfo<I> {
            fn from((env, var, a, b): (Environment<I>, Variance, &$t<I>, &$t<I>)) -> Self {
                EdgeInfo::Unification {
                    environment: env,
                    variance: var,
                    kind: UnifyKind::$t(a.clone(), b.clone())
                }
            }
        }
        )*

    }
}

can_unify!(GenericArg, Ty, DomainGoal,);

#[derive(Clone)]
pub struct Opaque<T>(T);

impl<T> Debug for Opaque<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "Opaque(..)")
    }
}

impl<T> PartialEq for Opaque<T> {
    fn eq(&self, rhs: &Self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EdgeInfo<I: Interner> {
    UsingClauses {
        goal: UCanonical<InEnvironment<DomainGoal<I>>>,
    },
    UsingSimplification {
        goal: UCanonicalGoal<I>,
    },
    PCImplication(PCINode<I>),
    Obligation {
        goal: InEnvironment<Goal<I>>,
    },
    Unification {
        environment: Environment<I>,
        variance: Variance,
        kind: UnifyKind<I>,
    },
    SubGoal {
        goal: UCanonicalGoal<I>,
    },
    UsingSubstitution {
        free_vars: Vec<GenericArg<I>>,
        universes: Opaque<UniverseMap>,
        subst: Canonical<ConstrainedSubst<I>>,
    },
}

#[derive(Clone)]
pub struct PCINode<I: Interner> {
    pub clause: Binders<ProgramClauseImplication<I>>,
    pub infer: InferenceTable<I>,
}

impl<I: Interner> std::cmp::PartialEq for PCINode<I> {
    fn eq(&self, other: &Self) -> bool {
        self.clause.eq(&other.clause)
    }
}

impl<I: Interner> Debug for PCINode<I> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{:?}", self.clause)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SubGoalKind {
    Disjunction,
    Conjunction,
    FixpointIteration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedNode<I: Interner> {
    pub subgoals: Vec<ProofTree<I>>,
    pub kind: SubGoalKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClauseNode<I: Interner> {
    pub clause: ProgramClauseImplication<I>,
    pub subgoal: Box<ProofTree<I>>,
}

// XXX: Additional information here we could use
//      for unification failure state, or other operations
//      that failed with a specific state.
#[derive(Debug, Clone)]
pub enum LeafNode<I: Interner> {
    FromCache {
        goal: UCanonicalGoal<I>,
        result: Box<LeafNode<I>>,
    },
    Resolved(Solution<I>),
    UnificationSuccess,
    NoSolution,
    Floundered,
    Unknown,
}

impl<I: Interner> std::cmp::PartialEq for LeafNode<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LeafNode::FromCache { result, .. }, _) => (**result).eq(other),
            (_, LeafNode::FromCache { result, .. }) => self.eq(&result),
            (LeafNode::Resolved(sl), LeafNode::Resolved(sr)) => sl.eq(sr),
            (LeafNode::NoSolution, LeafNode::NoSolution)
            | (LeafNode::Unknown, LeafNode::Unknown) => true,
            (_, _) => false,
        }
    }
}

// TODO: we should probably want to talk about unification
//       in a broader sense. Not just for goals.
#[derive(Debug, Clone, PartialEq)]
pub struct UnifyNode
// <I: Interner>
{
    // environment?
    // pub left: K,
    // pub right: K,
    // pub variance: Variance,
    // pub result: V,
}

// ---------------------------------

pub trait FromRoot<I: Interner> {
    fn assemble(self, root: UCanonicalGoal<I>) -> ProofTree<I>;
}

impl<F, I: Interner> FromRoot<I> for F
where
    F: FnOnce(UCanonicalGoal<I>) -> ProofTree<I>,
{
    fn assemble(self, root: UCanonicalGoal<I>) -> ProofTree<I> {
        (self)(root)
    }
}

// ----------------
// Stateful builder

// TODO: we need to represent the recursive expansion
//       of goals, here we just keep the flattened results.
//
//       This also happens with unification, which can fail,
//       and also spawn nested obligations.
#[derive(Clone)]
pub enum StagedAssembler<I: Interner> {
    Tree(ProofTree<I>),
    // Assembler(ProofTreeAssembler<T, I>),
}

// #[derive(Clone)]
// pub struct ProofTreeAssembler<T, I: Interner> {
//     pending: Vec<(T, StagedAssembler<I>)>,
//     kind: SubGoalKind,
// }

// ---------------
// Assembler impls

impl<I: Interner> FromRoot<I> for ProofTree<I> {
    fn assemble(self, root: UCanonicalGoal<I>) -> ProofTree<I> {
        ProofTree::Introducing(EdgeInfo::SubGoal { goal: root }, Box::new(self))
    }
}

impl<I: Interner> FromRoot<I> for StagedAssembler<I> {
    fn assemble(self, root: UCanonicalGoal<I>) -> ProofTree<I> {
        match self {
            StagedAssembler::Tree(t) => {
                ProofTree::Introducing(EdgeInfo::SubGoal { goal: root }, Box::new(t))
            }
        }
    }
}

impl<I: Interner> From<Fallible<Solution<I>>> for LeafNode<I> {
    fn from(fallible: Fallible<Solution<I>>) -> Self {
        match fallible {
            Ok(sol) => sol.into(),
            Err(..) => LeafNode::NoSolution,
        }
    }
}

impl<I: Interner> From<Floundered> for LeafNode<I> {
    fn from(err: Floundered) -> Self {
        LeafNode::Floundered
    }
}

impl<I: Interner> From<Floundered> for ProofTree<I> {
    fn from(err: Floundered) -> Self {
        ProofTree::Leaf(err.into())
    }
}

impl<I: Interner> From<NoSolution> for LeafNode<I> {
    fn from(err: NoSolution) -> Self {
        LeafNode::NoSolution
    }
}

impl<I: Interner> From<NoSolution> for ProofTree<I> {
    fn from(err: NoSolution) -> Self {
        ProofTree::Leaf(err.into())
    }
}

impl<I: Interner> From<Solution<I>> for LeafNode<I> {
    fn from(solution: Solution<I>) -> Self {
        LeafNode::Resolved(solution)
    }
}

impl<I: Interner> From<Fallible<Solution<I>>> for ProofTree<I> {
    fn from(fallible: Fallible<Solution<I>>) -> Self {
        ProofTree::Leaf(fallible.into())
    }
}

impl<I: Interner> From<Solution<I>> for ProofTree<I> {
    fn from(solution: Solution<I>) -> Self {
        ProofTree::Leaf(solution.into())
    }
}

// impl<I: Interner> From<ProofTreeAssembler<(), I>> for ProofTree<I> {
//     fn from(builder: ProofTreeAssembler<(), I>) -> Self {
//         ProofTree::Nested(NestedNode {
//             subgoals: builder
//                 .pending
//                 .into_iter()
//                 .map(|((), b)| match b {
//                     StagedAssembler::Tree(s) => s,
//                 })
//                 .collect(),
//             kind: builder.kind,
//         })
//         .compress()
//     }
// }

// impl<T, I: Interner> ProofTreeAssembler<T, I> {
//     pub fn new(kind: SubGoalKind) -> Self {
//         Self {
//             pending: vec![],
//             kind,
//         }
//     }

//     pub fn stage(&mut self, t: T, builder: StagedAssembler<I>) {
//         self.pending.push((t, builder));
//     }
// }

// impl<I: Interner> HasFail for ProofTreeAssembler<(), I> {
//     fn fail() -> Self {
//         let mut asm = ProofTreeAssembler::new(SubGoalKind::Conjunction);
//         asm.stage((), StagedAssembler::Tree(NoSolution.into()));
//         asm
//     }
// }

// impl<T, I: Interner> FromRoot<I> for ProofTreeAssembler<T, I> {
//     // TODO: generalize these!!!
//     fn assemble(self, root: UCanonicalGoal<I>) -> ProofTree<I> {
//         let mut attempts = self
//             .pending
//             .into_iter()
//             .map(|(_, bldr)| bldr.assemble(root.clone()))
//             .collect::<Vec<_>>();

//         if attempts.is_empty() {
//             attempts.push(panic!("dont' assume anything about empty subgoals"));
//         }

//         ProofTree::Introducing(
//             EdgeInfo::SubGoal { goal: root },
//             Box::new(ProofTree::Nested(NestedNode {
//                 subgoals: attempts,
//                 kind: self.kind,
//             })),
//         )
//         .compress()
//     }
// }

// -------------
// Builder impls

impl<I: Interner> ProofTree<I> {
    // We can use this to represent something that was stopped from the outside,
    // this doesn't actually provide anything for the proof, but gives us a way
    // to generate a dummy tree.
    pub fn proof_halted() -> Self {
        ProofTree::Halted
    }

    pub fn unify_success() -> Self {
        ProofTree::Leaf(LeafNode::UnificationSuccess)
    }

    pub fn unknown() -> Self {
        ProofTree::Leaf(LeafNode::Resolved(Solution::Ambig(Guidance::Unknown)))
    }

    pub fn fail_for(goal: UCanonicalGoal<I>) -> ProofTree<I> {
        ProofTree::Introducing(EdgeInfo::SubGoal { goal }, Box::new(NoSolution.into()))
    }

    pub fn compress(self) -> ProofTree<I> {
        match self {
            Self::Introducing(ei, nested) => Self::Introducing(ei, Box::new(nested.compress())),
            Self::Nested(NestedNode { subgoals, kind }) if !subgoals.is_empty() => {
                let mut subgoals = subgoals
                    .into_iter()
                    .map(|s| s.compress())
                    .collect::<Vec<_>>();

                if subgoals.len() == 1 || subgoals[1..].iter().all(|e| *e == subgoals[0]) {
                    return subgoals.pop().unwrap();
                }

                Self::Nested(NestedNode { subgoals, kind })
            }
            n => n,
        }
    }
}
