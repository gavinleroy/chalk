//! Proof tree for solving a goal

use chalk_ir::{interner::Interner, Fallible, NoSolution, ProgramClause, Variance};
use chalk_solve::Solution;

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

index_vec::define_index_type! {
    pub struct NodeIdx = usize;
    MAX_INDEX = usize::max_value();
    DISABLE_MAX_INDEX_CHECK = cfg!(not(debug_assertions));
}

index_vec::define_index_type! {
    pub struct EdgeIdx = usize;
    MAX_INDEX = usize::max_value();
    DISABLE_MAX_INDEX_CHECK = cfg!(not(debug_assertions));
}

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

#[derive(Debug, Clone)]
pub enum ProofTree<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    WithGoal(GoalNode<K, V>),
    Leaf(LeafNode<K, V>),
    Halted,
}

#[derive(Debug, Copy, Clone)]
pub enum SubGoalKind {
    Disjunction,
    Conjunction,
}

#[derive(Debug, Clone)]
pub struct GoalNode<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    // TODO: this isn't sufficient for the entirety of Chalk,
    //       but we'll expand later to handle other things.
    pub goal: K,
    pub kind: SubGoalKind,
    pub subgoals: Vec<ProofTree<K, V>>,
    //
}

// XXX: Additional information here we could use
//      for unification failure state, or other operations
//      that failed with a specific state.
#[derive(Debug, Clone)]
pub enum LeafNode<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    Unification(UnifyInfo<K, V>),
    Resolved(V),
}

// ---------------------------------

pub trait FromRoot<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    fn assemble(self, k: K) -> ProofTree<K, V>;
}

// ----------------
// Stateful builder

#[derive(Debug, Clone)]
pub struct UnifyInfo<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    // environment?
    pub left: K,
    pub right: K,
    pub variance: Variance,
    pub result: V,
}

// TODO: we need to represent the recursive expansion
//       of goals, here we just keep the flattened results.
//
//       This also happens with unification, which can fail,
//       and also spawn nested obligations.
pub struct ProofTreeBuilder<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    subgoals: Vec<ProofTree<K, V>>,
    unification_fail: Option<UnifyInfo<K, V>>,
}

pub struct ProofTreeAssembler<T, K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    pending: Vec<(T, ProofTreeBuilder<K, V>)>,
}

impl<T, K, V> ProofTreeAssembler<T, K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    pub fn new() -> Self {
        Self { pending: vec![] }
    }

    pub fn stage(&mut self, t: T, builder: ProofTreeBuilder<K, V>) {
        self.pending.push((t, builder));
    }
}

impl<T, K, V> FromRoot<K, V> for ProofTreeAssembler<T, K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    // TODO: generalize these!!!
    fn assemble(self, root: K) -> ProofTree<K, V> {
        let attempts = self
            .pending
            .into_iter()
            .map(|(_, bldr)| bldr.assemble(root.clone()))
            .collect::<Vec<_>>();
        ProofTree::WithGoal(GoalNode {
            goal: root,
            kind: SubGoalKind::Disjunction,
            subgoals: attempts,
        })
    }
}

impl<K, V> FromRoot<K, V> for ProofTreeBuilder<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    fn assemble(mut self, root: K) -> ProofTree<K, V> {
        if let Some(unify_info) = self.unification_fail {
            self.subgoals
                .push(ProofTree::Leaf(LeafNode::Unification(unify_info)));
        }

        // TODO: handle the unification failures
        ProofTree::WithGoal(GoalNode {
            goal: root,
            subgoals: self.subgoals,
            kind: SubGoalKind::Conjunction,
        })
    }
}

impl<K, V> ProofTreeBuilder<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    pub fn new() -> Self {
        Self {
            subgoals: Vec::default(),
            unification_fail: None,
        }
    }

    pub fn add_subtree(&mut self, subgoal: ProofTree<K, V>) {
        self.subgoals.push(subgoal);
    }

    pub fn register_unification_failure(&mut self, fail: UnifyInfo<K, V>) {
        assert!(self.unification_fail.is_none());
        self.unification_fail = Some(fail);
    }

    pub fn register_clause_implication<I: Interner>(&mut self, clause: ProgramClause<I>) {}
}

// ---------------------------------------
// OLD CRAP TO PORT OVER (or throw away)

// impl<K, V> Debug for ProofTree<K, V>
// where
//     K: Hash + Eq + Debug + Clone,
//     V: Debug + Clone,
// {
//     fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
//         write!(fmt, "#<proof-tree>")
//     }
// }

impl<K, V> Debug for ProofTreeBuilder<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "#<proof-tree-builder>")
    }
}

impl<K, V> ProofTree<K, V>
where
    K: Hash + Eq + Debug + Clone,
    V: Debug + Clone,
{
    // We can use this to represent something that was stopped from the outside,
    // this doesn't actually provide anything for the proof, but gives us a way
    // to generate a dummy tree.
    pub fn proof_halted() -> Self {
        ProofTree::Halted
    }
}

impl<K, I: Interner> ProofTree<K, Fallible<Solution<I>>>
where
    K: Hash + Eq + Debug + Clone,
{
    pub fn no_solution() -> Self {
        ProofTree::Leaf(LeafNode::Resolved(Err(NoSolution)))
    }
}
