use crate::fixed_point::Minimums;
use crate::solve::{SolveDatabase, TracedFallible};

use argus::proof_tree::*;
use chalk_ir::cast::Cast;
use chalk_ir::fold::TypeFoldable;
use chalk_ir::interner::{HasInterner, Interner};
use chalk_ir::visit::TypeVisitable;
use chalk_ir::zip::Zip;
use chalk_ir::{
    Binders, BoundVar, Canonical, ConstrainedSubst, Constraint, Constraints, DomainGoal,
    Environment, EqGoal, Fallible, GenericArg, GenericArgData, Goal, GoalData, InEnvironment,
    NoSolution, ProgramClauseImplication, QuantifierKind, Substitution, SubtypeGoal, TyKind,
    TyVariableKind, UCanonical, UnificationDatabase, UniverseMap, Variance,
};
use chalk_solve::debug_span;
use chalk_solve::infer::{InferenceTable, ParameterEnaVariableExt};
use chalk_solve::solve::truncate;
use chalk_solve::{Guidance, Solution};
use fluid_let::{fluid_let, fluid_set};
use index_vec::IndexVec;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt::Debug;
use tracing::{debug, instrument};

// TODO: this was a hack solution to track where the
//       updated information should go. Pls change.
fluid_let!(static CURRENT_OBLIGATION: OIdx);

index_vec::define_index_type! {
    pub struct OIdx = usize;
}

// We use this to track where the goals and obligations came from.
struct ObligationSet<I: Interner> {
    nil_idx: OIdx,
    goals: IndexVec<OIdx, EdgeInfo<I>>,
    edges: FxHashMap<OIdx, Vec<OIdx>>,
    resolution_map: FxHashMap<OIdx, ProofTree<I>>,
    obligation_map: FxHashMap<OIdx, Obligation<I>>,
    iteration_info: FxHashMap<OIdx, EdgeInfo<I>>,
    obligations: Vec<(OIdx, Obligation<I>)>,
}

impl<I: Interner> ObligationSet<I> {
    fn new() -> Self {
        Self {
            nil_idx: OIdx::new(usize::MAX),
            goals: IndexVec::new(),
            edges: FxHashMap::default(),
            resolution_map: FxHashMap::default(),
            obligation_map: FxHashMap::default(),
            iteration_info: FxHashMap::default(),
            obligations: Vec::default(),
        }
    }

    fn register_new_goal(&mut self, info: EdgeInfo<I>) -> OIdx {
        CURRENT_OBLIGATION.get(|parent| {
            let my_idx = self.goals.push(info);
            let mut parent_idx = *parent.unwrap_or(&my_idx);
            if parent_idx == self.nil_idx {
                parent_idx = my_idx;
            }
            self.edges.entry(parent_idx).or_default().push(my_idx);
            my_idx
        })
    }

    #[allow(dead_code)]
    fn register_iteration_substitution(&mut self, info: EdgeInfo<I>) {
        CURRENT_OBLIGATION.get(|idx| {
            let my_idx = *idx.unwrap();
            if let Some(prev) = self.iteration_info.insert(my_idx, info) {
                panic!("throwing out iteration info {:#?}", prev);
            }
        })
    }

    #[allow(dead_code)]
    fn obligation_idx(&self, obl: &Obligation<I>) -> OIdx {
        self.obligation_map
            .iter()
            .find_map(|(k, v)| (*v == *obl).then_some(*k))
            .unwrap()
    }

    fn update_proof(&mut self, trace: ProofTree<I>) {
        use std::collections::hash_map::Entry;
        CURRENT_OBLIGATION.get(|idx| {
            let idx = *idx.unwrap();

            // TODO: we can track the incremental update of
            //       the traces here if something else existed.

            match self.resolution_map.entry(idx) {
                Entry::Occupied(o) => {
                    let (k, mut first) = o.remove_entry();

                    if let ProofTree::Nested(NestedNode {
                        ref mut subgoals,
                        kind: SubGoalKind::FixpointIteration,
                    }) = &mut first
                    {
                        subgoals.push(trace);
                    } else {
                        let mut second = trace;
                        if let Some(info) = self.iteration_info.remove(&idx) {
                            second = ProofTree::Introducing(info, Box::new(second));
                        }
                        first = ProofTree::Nested(NestedNode {
                            subgoals: vec![first, second],
                            kind: SubGoalKind::FixpointIteration,
                        });
                    }

                    self.resolution_map.insert(k, first);
                }

                Entry::Vacant(o) => {
                    o.insert(trace);
                }
            }
        });
    }

    fn push(&mut self, obligation: Obligation<I>) {
        CURRENT_OBLIGATION.get(|idx| {
            let my_idx = *idx.unwrap();
            self.obligation_map.insert(my_idx, obligation.clone());
            self.obligations.push((my_idx, obligation));
        });
    }

    fn construct_proof(mut self, answer: Option<Fallible<Solution<I>>>) -> ProofTree<I> {
        fn visit_node<I: Interner>(
            i: OIdx,
            os: &mut ObligationSet<I>,
            leaf: &mut Option<LeafNode<I>>,
        ) -> ProofTree<I> {
            let mut subs = vec![];

            if let Some(children) = os.edges.remove(&i) {
                let subgoals = children.into_iter().filter_map(|j| {
                    if i != j {
                        Some(visit_node(j, os, leaf))
                    } else {
                        None
                    }
                });

                subs.extend(subgoals);
            }

            if let Some(nested_form) = os.resolution_map.remove(&i) {
                subs.push(nested_form);
            }

            if subs.is_empty() {
                if let Some(last) = leaf.take() {
                    debug_span!("construct_proof::visit_node", last = ?last);
                    subs.push(ProofTree::Leaf(last));
                } else {
                    subs.push(ProofTree::unknown());
                    // panic!("index doesn't have nested anything! {:?}", i);
                }
            }

            assert!(!subs.is_empty());

            let nested_form = ProofTree::Nested(NestedNode {
                subgoals: subs,
                kind: SubGoalKind::Conjunction,
            });

            let info = os.goals[i].clone();
            ProofTree::Introducing(info, Box::new(nested_form))
        }

        let mut leaf = answer.map(|v| v.into());

        let roots = self
            .edges
            .iter()
            .filter_map(|(k, vs)| vs.contains(k).then_some(*k))
            .collect::<Vec<_>>();

        let subgoals = roots
            .into_iter()
            .map(|i| visit_node(i, &mut self, &mut leaf))
            .collect::<Vec<_>>();

        assert!(self.edges.is_empty(), "not all edges used {:?}", self.edges);
        assert!(
            self.resolution_map.is_empty(),
            "not all solutions used {:?}",
            self.resolution_map
        );

        match (leaf, subgoals.is_empty()) {
            (Some(sol), true) => {
                return ProofTree::Leaf(sol);
            }
            (None, true) => panic!("no default soution with empty set"),
            (_, _) => ProofTree::Nested(NestedNode {
                subgoals,
                kind: SubGoalKind::Conjunction,
            })
            .compress(),
        }
    }
}

impl<I: Interner> std::fmt::Debug for ObligationSet<I> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{:?}", self.obligations)
    }
}

impl<I: Interner> std::ops::Deref for ObligationSet<I> {
    type Target = Vec<(OIdx, Obligation<I>)>;

    fn deref(&self) -> &Self::Target {
        &self.obligations
    }
}

impl<I: Interner> std::ops::DerefMut for ObligationSet<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.obligations
    }
}

enum Outcome {
    Complete,
    Incomplete,
}

impl Outcome {
    fn is_complete(&self) -> bool {
        matches!(self, Outcome::Complete)
    }
}

/// A goal that must be resolved
#[derive(Clone, Debug, PartialEq, Eq)]
enum Obligation<I: Interner> {
    /// For "positive" goals, we flatten all the way out to leafs within the
    /// current `Fulfill`
    Prove(InEnvironment<Goal<I>>),

    /// For "negative" goals, we don't flatten in *this* `Fulfill`, which would
    /// require having a logical "or" operator. Instead, we recursively solve in
    /// a fresh `Fulfill`.
    Refute(InEnvironment<Goal<I>>),
}

/// When proving a leaf goal, we record the free variables that appear within it
/// so that we can update inference state accordingly.
#[derive(Clone, Debug)]
struct PositiveSolution<I: Interner> {
    free_vars: Vec<GenericArg<I>>,
    universes: UniverseMap,
    solution: Solution<I>,
}

/// When refuting a goal, there's no impact on inference state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NegativeSolution {
    Refuted,
    Ambiguous,
}

fn canonicalize<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: I,
    value: T,
) -> (Canonical<T>, Vec<GenericArg<I>>)
where
    T: TypeFoldable<I>,
    T: HasInterner<Interner = I>,
{
    let res = infer.canonicalize(interner, value);
    let free_vars = res
        .free_vars
        .into_iter()
        .map(|free_var| free_var.to_generic_arg(interner))
        .collect();
    (res.quantified, free_vars)
}

fn u_canonicalize<I: Interner, T>(
    _infer: &mut InferenceTable<I>,
    interner: I,
    value0: &Canonical<T>,
) -> (UCanonical<T>, UniverseMap)
where
    T: Clone + HasInterner<Interner = I> + TypeFoldable<I> + TypeVisitable<I>,
    T: HasInterner<Interner = I>,
{
    let res = InferenceTable::u_canonicalize(interner, value0);
    (res.quantified, res.universes)
}

fn unify<I: Interner, T>(
    infer: &mut InferenceTable<I>,
    interner: I,
    db: &dyn UnificationDatabase<I>,
    environment: &Environment<I>,
    variance: Variance,
    a: &T,
    b: &T,
) -> Fallible<Vec<InEnvironment<Goal<I>>>>
where
    T: ?Sized + Zip<I>,
{
    let res = infer.relate(interner, db, environment, variance, a, b)?;
    Ok(res.goals)
}

/// A `Fulfill` is where we actually break down complex goals, instantiate
/// variables, and perform inference. It's highly stateful. It's generally used
/// in Chalk to try to solve a goal, and then package up what was learned in a
/// stateless, canonical way.
///
/// In rustc, you can think of there being an outermost `Fulfill` that's used when
/// type checking each function body, etc. There, the state reflects the state
/// of type inference in general. But when solving trait constraints, *fresh*
/// `Fulfill` instances will be created to solve canonicalized, free-standing
/// goals, and transport what was learned back to the outer context.
pub(super) struct Fulfill<'s, I: Interner, Solver: SolveDatabase<I>> {
    solver: &'s mut Solver,
    subst: Substitution<I>,
    infer: InferenceTable<I>,

    /// The remaining goals to prove or refute
    obligations: ObligationSet<I>,

    /// Lifetime constraints that must be fulfilled for a solution to be fully
    /// validated.
    constraints: FxHashSet<InEnvironment<Constraint<I>>>,

    /// Record that a goal has been processed that can neither be proved nor
    /// refuted. In such a case the solution will be either `CannotProve`, or `Err`
    /// in the case where some other goal leads to an error.
    cannot_prove: bool,
}

// FIXME: Previously we returned a `Fallible` type which simply uses the NoSolution
//        variant as the error, here we just return the attempted proof tree, and
//        later can use NoSolution as the Fallible<Solution<I>> type.
type TracedCreate<T, I> = Result<T, ProofTree<I>>;

impl<'s, I: Interner, Solver: SolveDatabase<I>> Fulfill<'s, I, Solver> {
    #[instrument(level = "debug", skip(solver, infer))]
    pub(super) fn new_with_clause(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<DomainGoal<I>>,
        clause: &Binders<ProgramClauseImplication<I>>,
    ) -> TracedCreate<Self, I> {
        let mut fulfill = Fulfill {
            solver,
            infer,
            subst,
            obligations: ObligationSet::new(),
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        fluid_set!(CURRENT_OBLIGATION, fulfill.obligations.nil_idx);

        let ProgramClauseImplication {
            consequence,
            conditions,
            constraints,
            priority: _,
        } = fulfill
            .infer
            .instantiate_binders_existentially(fulfill.solver.interner(), clause.clone());

        debug!(?consequence, ?conditions, ?constraints);
        fulfill
            .constraints
            .extend(constraints.as_slice(fulfill.interner()).to_owned());

        debug!("the subst is {:?}", fulfill.subst);

        if let Err(_e) = fulfill.unify(
            &canonical_goal.environment,
            Variance::Invariant,
            &canonical_goal.goal,
            &consequence,
        ) {
            return Err(fulfill.obligations.construct_proof(Some(Err(NoSolution))));
        }

        // if so, toss in all of its premises
        for condition in conditions.as_slice(fulfill.solver.interner()) {
            if let Err(_e) = fulfill.push_goal(&canonical_goal.environment, condition.clone()) {
                return Err(fulfill.obligations.construct_proof(Some(Err(NoSolution))));
            }
        }

        Ok(fulfill)
    }

    pub(super) fn new_with_simplification(
        solver: &'s mut Solver,
        infer: InferenceTable<I>,
        subst: Substitution<I>,
        canonical_goal: InEnvironment<Goal<I>>,
    ) -> TracedCreate<Self, I> {
        let mut fulfill = Fulfill {
            solver,
            infer,
            subst,
            obligations: ObligationSet::new(),
            constraints: FxHashSet::default(),
            cannot_prove: false,
        };

        fluid_set!(CURRENT_OBLIGATION, fulfill.obligations.nil_idx);

        if let Err(_e) = fulfill.push_goal(&canonical_goal.environment, canonical_goal.goal.clone())
        {
            return Err(fulfill.obligations.construct_proof(Some(Err(NoSolution))));
        }

        Ok(fulfill)
    }

    fn push_obligation(&mut self, obligation: Obligation<I>) {
        // truncate to avoid overflows
        match &obligation {
            Obligation::Prove(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    return;
                }
            }
            Obligation::Refute(goal) => {
                if truncate::needs_truncation(
                    self.solver.interner(),
                    &mut self.infer,
                    self.solver.max_size(),
                    goal,
                ) {
                    // the goal is too big. Record that we should return Ambiguous
                    self.cannot_prove = true;
                    return;
                }
            }
        };
        self.obligations.push(obligation);
    }

    /// Unifies `a` and `b` in the given environment.
    ///
    /// Wraps `InferenceTable::unify`; any resulting normalizations are added
    /// into our list of pending obligations with the given environment.
    pub(super) fn unify<T>(
        &mut self,
        environment: &Environment<I>,
        variance: Variance,
        a: &T,
        b: &T,
    ) -> Fallible<()>
    where
        T: Clone + Zip<I> + Debug,
        (Environment<I>, Variance, T, T): Into<EdgeInfo<I>>,
    {
        let goal_idx = self
            .obligations
            .register_new_goal((environment.clone(), variance, a.clone(), b.clone()).into());
        fluid_set!(CURRENT_OBLIGATION, goal_idx);

        let goals = match unify(
            &mut self.infer,
            self.solver.interner(),
            self.solver.db().unification_database(),
            environment,
            variance,
            a,
            b,
        ) {
            Ok(goals) => goals,
            Err(e) => {
                self.obligations.update_proof(e.clone().into());
                return Err(e);
            }
        };

        debug!("unify({:?}, {:?}) succeeded", a, b);
        debug!("unify: goals={:?}", goals);

        // If there are no more subgoals then the unification
        // succeeded without requiring any additional constraints.
        if goals.is_empty() {
            self.obligations.update_proof(ProofTree::unify_success());
        }

        for goal in goals {
            let goal = goal.cast(self.solver.interner());

            let goal_idx = self
                .obligations
                .register_new_goal(EdgeInfo::Obligation { goal: goal.clone() });
            fluid_set!(CURRENT_OBLIGATION, goal_idx);

            self.push_obligation(Obligation::Prove(goal));
        }
        Ok(())
    }

    /// Create obligations for the given goal in the given environment. This may
    /// ultimately create any number of obligations.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn push_goal(
        &mut self,
        environment: &Environment<I>,
        goal: Goal<I>,
    ) -> Fallible<()> {
        let goal_idx = self.obligations.register_new_goal(EdgeInfo::Obligation {
            goal: InEnvironment::new(environment, goal.clone()),
        });
        fluid_set!(CURRENT_OBLIGATION, goal_idx);

        let interner = self.solver.interner();
        match goal.data(interner) {
            GoalData::Quantified(QuantifierKind::ForAll, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_universally(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Quantified(QuantifierKind::Exists, subgoal) => {
                let subgoal = self
                    .infer
                    .instantiate_binders_existentially(self.solver.interner(), subgoal.clone());
                self.push_goal(environment, subgoal)?;
            }
            GoalData::Implies(wc, subgoal) => {
                let new_environment =
                    &environment.add_clauses(interner, wc.iter(interner).cloned());
                self.push_goal(new_environment, subgoal.clone())?;
            }
            GoalData::All(goals) => {
                for subgoal in goals.as_slice(interner) {
                    self.push_goal(environment, subgoal.clone())?;
                }
            }
            GoalData::Not(subgoal) => {
                let in_env = InEnvironment::new(environment, subgoal.clone());
                self.push_obligation(Obligation::Refute(in_env));
            }
            GoalData::DomainGoal(_) => {
                let in_env = InEnvironment::new(environment, goal);
                self.push_obligation(Obligation::Prove(in_env));
            }
            GoalData::EqGoal(EqGoal { a, b }) => {
                self.unify(environment, Variance::Invariant, &a, &b)?;
            }
            GoalData::SubtypeGoal(SubtypeGoal { a, b }) => {
                let a_norm = self.infer.normalize_ty_shallow(interner, a);
                let a = a_norm.as_ref().unwrap_or(a);
                let b_norm = self.infer.normalize_ty_shallow(interner, b);
                let b = b_norm.as_ref().unwrap_or(b);

                if matches!(
                    a.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) && matches!(
                    b.kind(interner),
                    TyKind::InferenceVar(_, TyVariableKind::General)
                ) {
                    self.cannot_prove = true;
                    self.obligations.update_proof(ProofTree::unknown());
                } else {
                    self.unify(environment, Variance::Covariant, &a, &b)?;
                }
            }
            GoalData::CannotProve => {
                debug!("Pushed a CannotProve goal, setting cannot_prove = true");
                self.cannot_prove = true;
                self.obligations.update_proof(ProofTree::unknown());
            }
        }
        Ok(())
    }

    #[instrument(level = "debug", skip(self, minimums, should_continue))]
    fn prove(
        &mut self,
        wc: InEnvironment<Goal<I>>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<PositiveSolution<I>> {
        let interner = self.solver.interner();
        let (quantified, free_vars) = canonicalize(&mut self.infer, interner, wc);
        let (quantified, universes) = u_canonicalize(&mut self.infer, interner, &quantified);
        let TracedFallible { solution, trace } =
            self.solver
                .solve_goal(quantified, minimums, should_continue);
        self.obligations.update_proof(trace);
        solution.map(|solution| PositiveSolution {
            free_vars,
            universes,
            solution,
        })
    }

    fn refute(
        &mut self,
        goal: InEnvironment<Goal<I>>,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<NegativeSolution> {
        let canonicalized = match self
            .infer
            .invert_then_canonicalize(self.solver.interner(), goal)
        {
            Some(v) => v,
            None => {
                // Treat non-ground negatives as ambiguous. Note that, as inference
                // proceeds, we may wind up with more information here.
                return Ok(NegativeSolution::Ambiguous);
            }
        };

        // Negate the result
        let (quantified, _) =
            u_canonicalize(&mut self.infer, self.solver.interner(), &canonicalized);
        let mut minimums = Minimums::new(); // FIXME -- minimums here seems wrong
        let TracedFallible { solution, trace } =
            self.solver
                .solve_goal(quantified, &mut minimums, should_continue);

        self.obligations.update_proof(trace);
        if let Ok(solution) = solution {
            if solution.is_unique() {
                Err(NoSolution)
            } else {
                Ok(NegativeSolution::Ambiguous)
            }
        } else {
            Ok(NegativeSolution::Refuted)
        }
    }

    /// Trying to prove some goal led to a the substitution `subst`; we
    /// wish to apply that substitution to our own inference variables
    /// (and incorporate any region constraints). This substitution
    /// requires some mapping to get it into our namespace -- first,
    /// the universes it refers to have been canonicalized, and
    /// `universes` stores the mapping back into our
    /// universes. Second, the free variables that appear within can
    /// be mapped into our variables with `free_vars`.
    fn apply_solution(
        &mut self,
        free_vars: Vec<GenericArg<I>>,
        universes: UniverseMap,
        subst: Canonical<ConstrainedSubst<I>>,
    ) {
        use chalk_solve::infer::ucanonicalize::UniverseMapExt;
        let subst = universes.map_from_canonical(self.interner(), &subst);
        let ConstrainedSubst { subst, constraints } = self
            .infer
            .instantiate_canonical(self.solver.interner(), subst);

        debug!(
            "fulfill::apply_solution: adding constraints {:?}",
            constraints
        );
        self.constraints
            .extend(constraints.as_slice(self.interner()).to_owned());

        // We use the empty environment for unification here because we're
        // really just doing a substitution on unconstrained variables, which is
        // guaranteed to succeed without generating any new constraints.
        let empty_env = &Environment::new(self.solver.interner());

        for (i, free_var) in free_vars.into_iter().enumerate() {
            let subst_value = subst.at(self.interner(), i);
            self.unify(empty_env, Variance::Invariant, &free_var, subst_value)
                .unwrap_or_else(|err| {
                    panic!(
                        "apply_solution failed with free_var={:?}, subst_value={:?}: {:?}",
                        free_var, subst_value, err
                    );
                });
        }
    }

    fn fulfill(
        &mut self,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> Fallible<Outcome> {
        debug_span!("fulfill", obligations=?self.obligations);

        // Try to solve all the obligations. We do this via a fixed-point
        // iteration. We try to solve each obligation in turn. Anything which is
        // successful, we drop; anything ambiguous, we retain in the
        // `obligations` array. This process is repeated so long as we are
        // learning new things about our inference state.
        let mut obligations = Vec::with_capacity(self.obligations.len());
        let mut progress = true;

        while progress {
            progress = false;
            debug!("start of round, {} obligations", self.obligations.len());

            // Take the list of `obligations` to solve this round and replace it
            // with an empty vector. Iterate through each obligation to solve
            // and solve it if we can. If not (because of ambiguity), then push
            // it back onto `self.to_prove` for next round. Note that
            // `solve_one` may also push onto the `self.to_prove` list
            // directly.
            assert!(obligations.is_empty());
            while let Some((goal_idx, obligation)) = self.obligations.pop() {
                fluid_set!(CURRENT_OBLIGATION, goal_idx);
                let ambiguous = match &obligation {
                    Obligation::Prove(wc) => {
                        let PositiveSolution {
                            free_vars,
                            universes,
                            solution,
                        } = self.prove(wc.clone(), minimums, should_continue.clone())?;

                        if let Some(constrained_subst) = solution.definite_subst(self.interner()) {
                            // If the substitution is trivial, we won't actually make any progress by applying it!
                            // So we need to check this to prevent endless loops.
                            let nontrivial_subst = !is_trivial_canonical_subst(
                                self.interner(),
                                &constrained_subst.value.subst,
                            );

                            let has_constraints = !constrained_subst
                                .value
                                .constraints
                                .is_empty(self.interner());

                            if nontrivial_subst || has_constraints {
                                self.apply_solution(free_vars, universes, constrained_subst);
                                progress = true;
                            }
                        }

                        solution.is_ambig()
                    }
                    Obligation::Refute(goal) => {
                        let answer = self.refute(goal.clone(), should_continue.clone())?;
                        answer == NegativeSolution::Ambiguous
                    }
                };

                if ambiguous {
                    debug!("ambiguous result: {:?}", obligation);
                    obligations.push((goal_idx, obligation));
                }
            }

            self.obligations.append(&mut obligations);
            debug!("end of round, {} obligations left", self.obligations.len());
        }

        // At the end of this process, `self.obligations` should have
        // all of the ambiguous obligations, and `obligations` should
        // be empty.
        assert!(obligations.is_empty());

        if self.obligations.is_empty() {
            Ok(Outcome::Complete)
        } else {
            Ok(Outcome::Incomplete)
        }
    }

    /// Try to fulfill all pending obligations and build the resulting
    /// solution. The returned solution will transform `subst` substitution with
    /// the outcome of type inference by updating the replacements it provides.
    pub(super) fn solve(
        mut self,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        macro_rules! consume_trace {
            ($this:tt, $val1:expr, $val2:expr) => {{
                TracedFallible {
                    solution: $val1,
                    trace: $this.obligations.construct_proof(Some($val2)),
                }
            }};
            ($this:tt, Unique => $sol:expr) => {{
                consume_trace!(
                    $this,
                    Ok(Solution::Unique($sol.clone())),
                    Ok(Solution::Unique($sol))
                )
            }};
            ($this:tt, Suggested => $sol:expr) => {{
                consume_trace!(
                    $this,
                    Ok(Solution::Ambig(Guidance::Suggested($sol.clone()))),
                    Ok(Solution::Ambig(Guidance::Suggested($sol)))
                )
            }};
            ($this:tt, Definite => $sol:expr) => {{
                consume_trace!(
                    $this,
                    Ok(Solution::Ambig(Guidance::Definite($sol.clone()))),
                    Ok(Solution::Ambig(Guidance::Definite($sol)))
                )
            }};
            ($this:tt, Unknown) => {{
                consume_trace!(
                    $this,
                    Ok(Solution::Ambig(Guidance::Unknown)),
                    Ok(Solution::Ambig(Guidance::Unknown))
                )
            }};
            ($this:tt, Error => $err:expr) => {{
                consume_trace!($this, Err($err.clone()), Err($err))
            }};
        }

        let outcome = match self.fulfill(minimums, should_continue.clone()) {
            Ok(o) => o,
            Err(e) => return consume_trace!(self, Error => e),
        };

        if self.cannot_prove {
            debug!("Goal cannot be proven (cannot_prove = true), returning ambiguous");
            return consume_trace!(self, Unknown);
        }

        if outcome.is_complete() {
            // No obligations remain, so we have definitively solved our goals,
            // and the current inference state is the unique way to solve them.

            let constraints = Constraints::from_iter(self.interner(), self.constraints.clone());
            let constrained = canonicalize(
                &mut self.infer,
                self.solver.interner(),
                ConstrainedSubst {
                    subst: self.subst,
                    constraints,
                },
            );
            return consume_trace!(self, Unique => constrained.0);
        }

        // Otherwise, we have (positive or negative) obligations remaining, but
        // haven't proved that it's *impossible* to satisfy out obligations. we
        // need to determine how to package up what we learned about type
        // inference as an ambiguous solution.

        let canonical_subst =
            canonicalize(&mut self.infer, self.solver.interner(), self.subst.clone());

        if canonical_subst
            .0
            .value
            .is_identity_subst(self.solver.interner())
        {
            // In this case, we didn't learn *anything* definitively. So now, we
            // go one last time through the positive obligations, this time
            // applying even *tentative* inference suggestions, so that we can
            // yield these upwards as our own suggestions. There are no
            // particular guarantees about *which* obligaiton we derive
            // suggestions from.

            while let Some((goal_idx, obligation)) = self.obligations.pop() {
                fluid_set!(CURRENT_OBLIGATION, goal_idx);
                if let Obligation::Prove(goal) = obligation {
                    let PositiveSolution {
                        free_vars,
                        universes,
                        solution,
                    } = self.prove(goal, minimums, should_continue.clone()).unwrap();
                    if let Some(constrained_subst) =
                        solution.constrained_subst(self.solver.interner())
                    {
                        self.apply_solution(free_vars, universes, constrained_subst);
                        return consume_trace!(
                            self,
                            Suggested => canonical_subst.0
                        );
                    }
                }
            }

            consume_trace!(self, Unknown)
        } else {
            // While we failed to prove the goal, we still learned that
            // something had to hold. Here's an example where this happens:
            //
            // ```rust
            // trait Display {}
            // trait Debug {}
            // struct Foo<T> {}
            // struct Bar {}
            // struct Baz {}
            //
            // impl Display for Bar {}
            // impl Display for Baz {}
            //
            // impl<T> Debug for Foo<T> where T: Display {}
            // ```
            //
            // If we pose the goal `exists<T> { T: Debug }`, we can't say
            // for sure what `T` must be (it could be either `Foo<Bar>` or
            // `Foo<Baz>`, but we *can* say for sure that it must be of the
            // form `Foo<?0>`.
            consume_trace!(
                self,
                Definite => canonical_subst.0
            )
        }
    }

    fn interner(&self) -> I {
        self.solver.interner()
    }
}

fn is_trivial_canonical_subst<I: Interner>(interner: I, subst: &Substitution<I>) -> bool {
    // A subst is trivial if..
    subst.iter(interner).enumerate().all(|(index, parameter)| {
        let is_trivial = |b: Option<BoundVar>| match b {
            None => false,
            Some(bound_var) => {
                if let Some(index1) = bound_var.index_if_innermost() {
                    index == index1
                } else {
                    false
                }
            }
        };

        match parameter.data(interner) {
            // All types and consts are mapped to distinct variables. Since this
            // has been canonicalized, those will also be the first N
            // variables.
            GenericArgData::Ty(t) => is_trivial(t.bound_var(interner)),
            GenericArgData::Const(t) => is_trivial(t.bound_var(interner)),
            GenericArgData::Lifetime(t) => is_trivial(t.bound_var(interner)),
        }
    })
}
