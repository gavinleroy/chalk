use crate::fixed_point::{Cache, Minimums, RecursiveContext, SolverStuff};
use crate::solve::{IsTracing, SolveDatabase, SolveIteration};
use crate::UCanonicalGoal;

use chalk_ir::Constraints;
use chalk_ir::{interner::Interner, Fallible, NoSolution};
use chalk_ir::{Canonical, ConstrainedSubst, Goal, InEnvironment, UCanonical};
use chalk_solve::{coinductive_goal::IsCoinductive, RustIrDatabase, Solution};
use std::fmt;

use argus::proof_tree::{flat::ProofNodeIdx, ProofNode, TracedFallible, TreeDescription};

/// A Solver is the basic context in which you can propose goals for a given
/// program. **All questions posed to the solver are in canonical, closed form,
/// so that each question is answered with effectively a "clean slate"**. This
/// allows for better caching, and simplifies management of the inference
/// context.
pub(crate) struct Solver<'me, I: Interner> {
    program: &'me dyn RustIrDatabase<I>,
    context: &'me mut RecursiveContext<UCanonicalGoal<I>, TracedFallible<I>, I>,
}

pub struct RecursiveSolver<I: Interner> {
    ctx: Box<RecursiveContext<UCanonicalGoal<I>, TracedFallible<I>, I>>,
}

impl<I: Interner> RecursiveSolver<I> {
    pub fn new(
        overflow_depth: usize,
        max_size: usize,
        cache: Option<Cache<UCanonicalGoal<I>, TracedFallible<I>>>,
    ) -> Self {
        Self {
            ctx: Box::new(RecursiveContext::new(overflow_depth, max_size, cache)),
        }
    }

    pub fn consume_tree(self, desc: TreeDescription) -> argus::proof_tree::flat::ProofTreeNav<I> {
        let ctx = *self.ctx;
        let builder = ctx.inspect;
        builder.root_at(desc)
    }
}

impl<I: Interner> fmt::Debug for RecursiveSolver<I> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "RecursiveSolver")
    }
}

impl<'me, I: Interner> Solver<'me, I> {
    pub(crate) fn new(
        context: &'me mut RecursiveContext<UCanonicalGoal<I>, TracedFallible<I>, I>,
        program: &'me dyn RustIrDatabase<I>,
    ) -> Self {
        Self { program, context }
    }
}

impl<I: Interner> SolverStuff<UCanonicalGoal<I>, TracedFallible<I>, I> for &dyn RustIrDatabase<I> {
    fn is_coinductive_goal(self, goal: &UCanonicalGoal<I>) -> bool {
        goal.is_coinductive(self)
    }

    fn initial_value(self, goal: &UCanonicalGoal<I>, coinductive_goal: bool) -> TracedFallible<I> {
        if coinductive_goal {
            let solution = Solution::Unique(Canonical {
                value: ConstrainedSubst {
                    subst: goal.trivial_substitution(self.interner()),
                    constraints: Constraints::empty(self.interner()),
                },
                binders: goal.canonical.binders.clone(),
            });
            TracedFallible::from_built(Ok(solution.clone()), TreeDescription::auto_true())
        } else {
            TracedFallible::from_built(Err(NoSolution), TreeDescription::auto_false())
        }
    }

    fn solve_iteration(
        self,
        context: &mut RecursiveContext<UCanonicalGoal<I>, TracedFallible<I>, I>,
        goal: &UCanonicalGoal<I>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        Solver::new(context, self).solve_iteration(goal, minimums, should_continue)
    }

    fn reached_fixed_point(
        self,
        old_answer: &TracedFallible<I>,
        current_answer: &TracedFallible<I>,
    ) -> bool {
        // Some of our subgoals depended on us. We need to re-run
        // with the current answer.
        old_answer.solution == current_answer.solution || {
            // Subtle: if our current answer is ambiguous, we can just stop, and
            // in fact we *must* -- otherwise, we sometimes fail to reach a
            // fixed point. See `multiple_ambiguous_cycles` for more.
            match &current_answer.solution {
                Ok(s) => s.is_ambig(),
                Err(_) => false,
            }
        }
    }

    fn combine_values(self, old_value: &TracedFallible<I>, new_value: &mut TracedFallible<I>) {
        todo!()
    }

    fn error_value(self) -> TracedFallible<I> {
        TracedFallible::from_built(Err(NoSolution), TreeDescription::auto_false())
    }
}

impl<'me, I: Interner> SolveDatabase<I> for Solver<'me, I> {
    fn solve_goal(
        &mut self,
        goal: UCanonicalGoal<I>,
        minimums: &mut Minimums,
        should_continue: impl std::ops::Fn() -> bool + Clone,
    ) -> TracedFallible<I> {
        self.context
            .solve_goal(&goal, minimums, self.program, should_continue)
    }

    fn interner(&self) -> I {
        self.program.interner()
    }

    fn db(&self) -> &dyn RustIrDatabase<I> {
        self.program
    }

    fn max_size(&self) -> usize {
        self.context.max_size()
    }
}

impl<I: Interner> IsTracing<I> for Solver<'_, I> {
    fn get_inspector(&mut self) -> &mut argus::proof_tree::flat::ProofTreeBuilder<I> {
        &mut self.context.inspect
    }
}

impl<I: Interner> RecursiveSolver<I> {
    pub fn solve_traced(
        &mut self,
        program: &dyn RustIrDatabase<I>,
        goal: &UCanonical<InEnvironment<Goal<I>>>,
        should_continue: &dyn std::ops::Fn() -> bool,
    ) -> TracedFallible<I> {
        self.ctx.solve_root_goal(goal, program, should_continue)
    }
}

impl<I: Interner> chalk_solve::Solver<I> for RecursiveSolver<I> {
    fn solve(
        &mut self,
        program: &dyn RustIrDatabase<I>,
        goal: &UCanonical<InEnvironment<Goal<I>>>,
    ) -> Option<chalk_solve::Solution<I>> {
        self.solve_traced(program, goal, &|| true).solution.ok()
    }

    fn solve_limited(
        &mut self,
        program: &dyn RustIrDatabase<I>,
        goal: &UCanonical<InEnvironment<Goal<I>>>,
        should_continue: &dyn std::ops::Fn() -> bool,
    ) -> Option<chalk_solve::Solution<I>> {
        self.solve_traced(program, goal, should_continue)
            .solution
            .ok()
    }

    fn solve_multiple(
        &mut self,
        _program: &dyn RustIrDatabase<I>,
        _goal: &UCanonical<InEnvironment<Goal<I>>>,
        _f: &mut dyn FnMut(
            chalk_solve::SubstitutionResult<Canonical<ConstrainedSubst<I>>>,
            bool,
        ) -> bool,
    ) -> bool {
        unimplemented!("Recursive solver doesn't support multiple answers")
    }
}
