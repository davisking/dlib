# Dlib contracts and coding style

This guide is for contributors designing, documenting, and reviewing dlib
components. Contracts are the most important element of good software
design. A function is correct if and only if it honors its contract. A caller
must be able to understand how to use a tool correctly from its interface and
contract, without reading its implementation. Writing that contract is part of
designing the tool, not a documentation task performed after the design is finished.

Contracts let us reason about the correctness of a function or class in isolation
from the rest of the system. Given its preconditions, its implementation must
establish its postconditions and honor its specified failure behavior, relying
only on the contracts of the tools it calls. Passing tests, or working for the
current caller, is not a substitute for meeting that obligation.

The existing [contribution guidelines](docs/howto_contribute.xml),
[contract notation](docs/intro.xml), and neighboring `*_abstract.h` files remain
important references. This guide supplements them; it is not a claim that all
older code already follows every rule.

## Make repository content self-contained

Documentation, comments, and examples must make sense to their intended readers
without access to private conversations, review discussions, or an author's
memory. State the context needed to understand a rule or example. Introduce
specialized terms, identify the relevant inputs and public operations, and explain
any assumption on which the example depends. E.g. if an example illustrates improper
use of a private member, explicitly say that the member is private.

Link to an existing specification when it defines a concept, but include the
facts needed to understand the immediate point. Record decisions and their
rationale rather than referring to "what we discussed" or an unexplained earlier
experiment. Distinguish established behavior from proposed changes. Review the
text as someone who knows C++ and the relevant subject, but has not followed the
development of this particular component.

## Describe the component without boilerplate

Define a component in its own terms, not by referring to what an existing caller
does with it. Usually no use-case example is needed. Include one only when it
helps explain why the component is useful, not to supply missing meaning in the
contract. Use "i.e." to make a restatement or definition explicit and "e.g." to
identify an example.

Keep concise guidance that helps readers choose between related tools, such as
a compensated sum versus running statistics, or a one-shot solve versus a stored
factorization for successive solves. These are meaningful interface choices,
not stories about particular callers or claims of baseline quality.

Do not advertise baseline qualities expected of every dlib component. For example,
write "a pivoted factorization", not "a reusable pivoted factorization". Calling
one component "reusable", "correct", or "reliable" adds no contract and can imply
that components without that label lack the quality. Document nontrivial state
transitions and failure outcomes, such as a failed factorization invalidating a
previously prepared object, rather than generic promises of continued use.

Assume the reader knows C++. Do not repeat facts visible in the function
signature, such as passing an argument by value, or explain general language
mechanics such as `std::move()`. Similarly, omit routine reassurances of logical
constness, such as "B is unchanged" for a const input B or "preserves the factors"
for a const solve operation. Document meaningful observable effects, aliasing,
and failure-state guarantees separately.

Do not document which header to include in a contract. Omit routine
allocation-failure boilerplate such as "throws `std::bad_alloc` if allocation
fails". Document function-specific failure behavior, including error reporting
and the state left after failure, where relevant.

Explain what an argument or entry means, not just its C++ type. Define that
meaning positively and stop once it is clear. For example, "graph[i] lists the
indices of vertex i's neighbors" is sufficient; appending "not matrix
coefficients" adds nothing. Do not carry such contrasts from engineering notes
into the contract merely because the notes mention them.

State the guarantees rather than listing unspecified behavior. This applies to
generic disclaimers such as "nothing else is guaranteed", not just caveats about
future library versions. Anything not guaranteed by the contract is already
subject to change. Keep qualifications that define the actual operation, such
as which approximate quantity a numerical tolerance bounds; those are not generic
disclaimers. Avoid repeating a guarantee without adding meaning, e.g. "otherwise
returns false" after "returns true if and only if ...".

Do not enumerate special cases already covered by a general guarantee. For
example, when a matrix solve returns X with X.nc() == B.nc() and places no
restriction on B.nc(), "multiple or zero right-hand sides are supported" adds
nothing. Apply this test within clauses too: "the term is omitted, even if its
inputs contain NaNs" repeats a consequence of omitting the term. State the
omission rather than cataloguing values of an operand that is not evaluated.

Describe client-visible behavior, not the implementation's sequence of steps.
For example, whether a solve can return nonfinite values affects how callers use
its result; an internal rescaling and substitution step does not belong in the
public contract. Keep such explanations in implementation comments. Include
algorithmic details only when they define the promised result, caller obligations,
or another client-relevant guarantee. Defining the requested mathematical
operation is different from narrating how the implementation computes it.
Once a result guarantee is stated, omit an equivalent step-by-step restatement.

If a claim needs several caveats, first check whether it is too broad. Prefer
a precise, narrowly scoped guarantee over a claim followed by exceptions. For
example, an update may promise `#C == alpha*A*trans(A) + beta*make_symmetric(C)`
and permit A and C to alias. If beta is zero, say that the
`beta*make_symmetric(C)` term is omitted. Do not say "C is not read" and then
add warnings that it might still be read through A. "A and C may alias" already
permits overlap without changing the promised calculation; descriptions of
copies or individual reads and writes do not clarify that permission.

## Make every contract clear from the client's perspective

Every function and class contract must explain its operation or abstraction so
a client can understand what it provides, recognize when it is useful, and use
it productively. Exact predicates support that understanding; describing internal
calculations does not replace it.

Use existing public C++ operations to express objectives and results when they
are clearer and more concise than prose. For example, describe least squares as
minimizing `sum(squared(A*X-B))`, choosing the minimizer with the smallest
`sum(squared(X))`. Rely on the contracts of these public operations rather than
redefining them or translating each part of the expression into prose. Keep
additional clauses only for guarantees the expression does not convey, such as
aliasing support, skipping evaluation of a zero-weighted term, or failure effects.
These are meaningful guarantees, unlike examples that merely repeat them.
Retain any necessary numerical qualifications.

Open every nontrivial contract with a short, high-level description that gives
the reader an intuitive understanding of what the component represents or what
the operation does. For a function, put this overview at the start of `ensures`;
for a class, start its `WHAT THIS OBJECT REPRESENTS` description this way. Explain
the task and any objective or constraint that distinguishes it from related
operations. Then introduce the input/output meanings and precise guarantees.
Readers should not have to reconstruct the purpose from a list of conditions or
recognize an unexplained algorithm name. An equation or minimization statement
using familiar public operations can itself supply the overview and precise
meaning; do not add a prose introduction that only says the same thing again.

This does not require a separate statement of usefulness, a use-case example,
or phrases such as "this is useful because". Usually a clear definition of the
operation, its inputs, and its results makes its purpose apparent. Add further
explanation only when needed to understand the interface. The test is whether
the contract communicates a useful abstraction, not whether it includes
motivational prose.

Define one model that covers all supported modes, then explain how the inputs
select its parts. Do not open with a special case and later replace it with a
different definition. For example, describe a block matrix using `diag(z)`, then
define `z[i]` for each diagonal mode. Keep ownership rules, optional observer
protocols, and detailed error measures outside that opening explanation.

Diagnostics illustrate this general rule. Explain what was detected, its possible
effect on the result, and what the caller can do about it through the interface.
Distinguish a warning from failure, and a reason for caution from proof that a
result is wrong. State whether the observation concerns the latest operation or
accumulated history.
Keep qualifications that prevent callers from misinterpreting the diagnostic.

Precision belongs at the level of the promised abstraction. Preconditions and
promised numerical bounds need exact definitions. A heuristic warning, however,
should be described as a warning, not upgraded into a guarantee of failure or
successful recovery. An exact internal threshold or branch condition does not
explain why a caller should care. Keep detection mechanics in implementation
comments unless they are needed to understand a client-facing guarantee.

For example, suppose a matrix factorization offers compensated arithmetic to
reduce rounding error, and its solve operation reports whether a residual target
was reached. A cancellation diagnostic can be documented as follows:

```cpp
    bool had_cancellation() const;
    /*!
        ensures
            - Returns whether numerical cancellation has been detected during a
              factorization attempt that may have reduced the accuracy of the
              computed factors.
            - A true result warns that compensated arithmetic may be beneficial.
              It does not mean that the factorization failed. Use the solve's
              residual report to assess whether the requested accuracy was reached.
            - A false result does not establish that the factors are accurate.
            - Returns false initially. Once true, remains true across subsequent
              factorization attempts.
    !*/
```

Replacing this explanation with a pivot-ratio threshold and the dimensions at
which an internal test runs would describe the detector without explaining the
diagnostic. Conversely, saying only "numerical trouble occurred" would not tell
the caller whether the factors remain usable or what to check next.

## Expose choices the caller can make, not knobs for making the code work

For every public parameter, ask: what requirement of the caller's problem
determines its value? The contract must let the caller make that choice without
reading the implementation, knowing a particular backend, or experimenting until
the function happens to work. A name, default, legal range, and description of an
internal comparison do not answer that question.

Never expose a knob whose purpose is "choose the right setting or this function
will not work." That shifts an implementation defect onto the caller. The tool
must choose its own numerical safeguards and handle unsuccessful computations as
specified outcomes. Do not disguise such a knob by renaming it, giving it a
default, putting it in an options struct, or calling it advanced. If no meaningful
caller decision justifies it, remove it from the interface. When writing a clear
contract proves difficult for this reason, change the design, not just the prose.

Distinguish these implementation settings from genuine problem and resource
choices. A maximum geometric error in meters, a penalty explicitly included in
the objective, or a maximum iteration count can express what the caller wants
or can afford. State the quantity, units or scaling, effect of changing the
parameter, and outcome when the requested result cannot be obtained. Do not
delete such controls merely because their names include "tolerance" or "limit".
A resource limit must permit an explicit incomplete result; it must not mean
that correctness silently depends on guessing a sufficiently large value.

For example, a convenience least-squares solver should accept A and B and attempt
the minimum-norm least-squares solution. It should not require the caller to tune
a pivot cutoff to make that computation succeed. Numerical rank decisions belong
inside that solver. A deliberate change to the mathematical problem, such as
penalizing large solution coefficients, needs its own clearly defined operation;
it is not an unexplained accuracy knob. Removing a cutoff must not be accompanied
by a false promise of exact solutions for ill-conditioned inputs. Specify useful,
honest numerical outcomes, and test the automatic policy on difficult inputs.

Define each interface concept before listing its rules. Explain how each input
entry corresponds to the objects being processed and how its value affects the
result. When input preferences, resource accounting, and cost estimates have
separate roles, give those roles separate contracts instead of burying several
protocols in one parameter list. Distinguish the resources spent computing a
result from an estimate of the resources needed to use it later.

During a review, inspect the implementation and actual callers before deciding
whether a parameter is a problem choice or an internal setting. Tests that vary
a knob do not by themselves establish a reason to expose it. Check the resulting
interface and its failure behavior after simplifying it; do not merely move the
same undocumented tuning burden to another public tool.

## Explain the operation and the constraints it respects

A useful general way to explain an operation is "does X, subject to Y
constraints." Start with the task or objective, then explain the restrictions
the result must satisfy. Follow that intuitive overview with the definitions
and precise guarantees. This technique applies to any constrained operation,
whether or not it has overloads; it is not a required wording template.

For example, a path-finding contract could open with "Finds a minimum-total-weight
path between two graph vertices, using only edges marked as permitted." The rest
of the contract would define the inputs, the path's weight, and the outcome when
no permitted path exists. The opening gives the reader a task and its constraint
before presenting those details.

Describe constraints through their effect on the result. Saying how the
implementation compares two fields is not a substitute for explaining what
their values require or prefer. Distinguish mandatory conditions from heuristic
aims and tie-breaking preferences. A heuristic can aim to reduce a cost without
promising the least possible cost. Constraints on the result belong in `ensures`;
they are not, merely by being constraints, caller obligations for `requires`.

An overload that adds constraints is one application of this general technique.
Its additional arguments may instead express preferences or request a different
operation; explain that effect on the result too.

As a worked example, consider two overloads that choose an order for removing
vertices from an undirected graph. Vertices have indices in `[0,graph.size())`, and
`graph[i]` lists the neighbors of vertex i. Removing a vertex first adds missing
edges between its remaining neighbors. Different removal orders can therefore
create different numbers of edges. The basic overload uses a heuristic to seek
an order with few added edges. A second overload also accepts `priorities`, one
record per original vertex, with fields named `elimination_group` and
`tie_break_priority`.

An opening such as "returns vertices in elimination order according to their
groups" leaves both the operation and the effect of the extra argument unclear.
The second overload's postconditions can instead explain them in this order:

```cpp
        ensures
            - Computes an order for removing vertices from graph, aiming to limit
              the number of edges added while respecting the precedence constraints
              specified by priorities. Removing a vertex means first adding any
              missing edges between pairs of its remaining neighbors, then deleting
              the vertex and its incident edges.
            - Returns a permutation p of graph's vertex indices. p[k] is the
              original index of the vertex to remove at step k, starting with p[0].
              priorities[i] specifies the constraints and preferences for original
              vertex i.
            - For every pair of vertices i and j, if
              priorities[i].elimination_group < priorities[j].elimination_group,
              then i precedes j in p. Equal group values alone impose no relative
              order.
            - Subject to these constraints, uses a heuristic to limit the added
              edges. Among choices the heuristic considers equally preferable
              within a group, smaller tie_break_priority values are preferred.
              Smaller original vertex indices break remaining ties.
```

The opening provides the purpose and defines vertex removal without requiring
the reader to know the term "elimination." The next clause defines the output
and connects each priority record to its vertex. The inequality states an exact
constraint on that output; the final clause distinguishes the heuristic's aim
from the caller's tie-breaking preference. Neither field needs knowledge of a
particular implementation or of code that calls this function to have meaning.

An overload can refer to concepts defined by another overload, but include enough
context to make its own purpose and the effect of its extra arguments clear.
Do not say "the same result, except ..." unless the stated relationship between
the results is actually guaranteed. Sharing an algorithm does not imply that
an overload preserves the original result's relative ordering or changes it as
little as possible.

## Write and review the contract in order

1. Decide what the interface is meant to promise. For a new tool, design its
   contract before implementing it. For an existing tool, check its public
   specification against the implementation and tests. Investigate disagreements;
   do not silently weaken the contract to match a bug. Separate caller obligations
   from outcomes and incidental implementation facts. If the evidence leaves a
   necessary behavior unresolved, report the gap instead of inventing a guarantee
   or precondition.
2. Explain the operation in its own terms. Define what the inputs represent and
   what the result means. Introduce specialized terms before relying on them;
   naming a technique or saying what a caller uses it for is not a definition.
3. Make the requirements and results precise. For an index mapping, state its
   direction, range, size, and uniqueness where relevant. For a numerical test,
   define the quantity being tested and the tolerance's meaning. State distinct
   success, failure, and exceptional effects when the operation has them.
4. Read the draft with its declaration and referenced public specifications, but
   without the implementation or any calling code. Can a reader explain the
   operation, recognize when to use it, decide whether a call is permitted, and
   interpret each outcome well enough to act on it?
   Exact predicates alone do not make an unexplained abstraction understandable.
5. Make a separate deletion pass over the completed draft. For each clause, ask
   what a reader would lose if it were removed. A true statement is not enough:
   it must add information not already supplied by the signature or another
   clause. Check for signature repetition, general C++ explanations, repeated
   guarantees, irrelevant contrasts, private details, caller stories, include
   instructions, and generic non-guarantees.
   Keep the definitions and qualifications needed to understand the guarantees.

For example, suppose a matrix factorization class has public operations
`decompose(A)`, `nr()`, and `is_singular()`. They respectively factor a real
symmetric matrix A, report its order, and report unusable factors. Its solve
operation can be documented as follows:

```cpp
    matrix<double> solve (
        const matrix<double>& B
    ) const;
    /*!
        requires
            - !is_singular()
            - B.nr() == nr()
        ensures
            - Returns X satisfying A*X == B, up to numerical error, where A
              was passed to the most recent decompose().
            - X.nr() == nr() && X.nc() == B.nc()
            - A solution outside double's numerical range may contain
              nonfinite values.
        throws
            - dlib::error if numerical failure prevents computing X.
              The factorization remains usable after this exception.
    !*/
```

Do not append that B is unchanged, that the const solve preserves the factors,
that multiple or zero right-hand sides are supported, or that the implementation
tries scaled substitution on overflow. Those additions provide no further
client-facing guarantee here. The range warning and the state after an exception
do provide information the caller needs. For a different operation, supported
input/output aliasing, a no-allocation guarantee, or invalidation after failure
may likewise be important; concision does not mean removing those guarantees.

For example, this hypothetical function inverts a permutation. Its contract
defines the input and the direction of the result without referring to a caller:

```cpp
    std::vector<std::size_t> invert_permutation (
        const std::vector<std::size_t>& p
    );
    /*!
        requires
            - p contains each integer in [0,p.size()) exactly once.
        ensures
            - Returns q giving the position of each value in p, i.e.
              q.size() == p.size() and q[p[i]] == i for every i < p.size().
    !*/
```

## Design the interface and its contract together

Writing a clearer contract can reveal a better abstraction and justify changing
an existing component's behavior. Judge the proposed behavior by whether it still
solves the intended clients' problems, not merely by whether it matches the current
implementation. Make such changes deliberately: update the implementation,
callers, and tests together, and identify any compatibility consequences. This is
different from silently weakening a promise to excuse an implementation bug.

A component should represent one understandable concept. When code becomes
complicated, stop and ask: what tool, if I had it, would make this code simple
and clear? Design that tool and its contract, then ask
the same question about its implementation. Continue until each layer consists
of understandable operations built on other understandable operations. Good
software is layers of simple libraries built on one another; each layer's
contracts make the next layer more obviously correct.

Moving sections of a large implementation into separately named files is not
enough. Each boundary needs a clear meaning, caller obligations, and observable
result. For example, a standalone linear-algebra component used by a linear
programming (LP) solver must be describable and testable without the solver's
optimization model, objective, result status, or options object.

Not every helper needs to become public. A factorization can contain private
packing kernels and arithmetic helpers. The requirement is coherent layers with
contracts, not the largest possible collection of public APIs.

Keep caller policy outside the reusable tool. For example, a routine that reorders
matrix rows and columns can accept caller-supplied ordering priorities. An LP
solver can assign those priorities based on its constraint types; the matrix
routine should not need to interpret optimization constraints. Similarly, a tool
can report work and storage estimates to an observer without interpreting solver
options or producing solver statuses.
Specify the observer's operations, exception behavior, and permitted effects.
Provide a simple overload for callers who do not need that control.

Separate diagnostics from decisions. A matrix factorization can report numerical
difficulties without silently switching to higher-precision arithmetic on the next
call. Its caller can use those reports to explicitly request a different precision
mode, if the tool supports one. Test that the numerical tool actually honors those
explicit choices, including after a previous attempt reported trouble.
This does not require exposing internal safeguard settings. A tool offering an
explicit arithmetic choice must honor that choice; a solver whose contract owns
the numerical policy should select its safeguards internally. Neither design
should ask a caller to discover a setting that makes supported use correct.

## State exact caller obligations

Every function needs a contract in dlib's `requires` and `ensures` notation,
including internal tools. If there is any possibility that a call violates a
callee's precondition, the calling code has a bug. Fix the caller or redesign the
callee to remove that requirement. "It probably will not happen" is not a
justification. A caller must be able to guarantee every requirement before calling.

Every requirement must be an exact predicate stated in terms the caller can
observe. Use C++ expressions where they make the condition clearer. For example,
consider a linear-system solver with public observations `has_factor()` (whether
a usable factorization is available) and `dimension()` (the number of unknowns).
Its solve operation accepts a right-hand-side vector `rhs`, a preallocated result
vector `output`, and an accuracy target `accuracy`. Its preconditions could include:

```cpp
requires
    - has_factor() == true
    - rhs.size() == dimension()
    - output.size() == dimension()
    - accuracy >= 0 && std::isfinite(accuracy)
```

Do not use phrases such as "valid input", "compatible workspace", or "agrees
with the border" without an exact definition and a way for the caller to check
it. Specify dimensions, index ranges, scalar ranges, and aliasing rules directly.
Use a public predicate when a nontrivial structural check deserves a name.

For structural requirements, define the mathematical object rather than naming
an implementation convention. For example, a sparse matrix factorization needs
a pattern describing where its factors can have nonzero entries, including new
entries created during elimination. Calling this "a valid factor pattern" is not
enough: define the permitted elimination order and which entries the pattern must
contain. A symbolic-analysis operation, which computes the pattern before the
numerical factorization, can establish that invariant once. Its clients should
not have to construct or synchronize hidden factor metadata themselves.

Omit `requires` entirely when there are no requirements. Never write
`requires - none`.

Give each independent requirement its own bullet. Do not join unrelated
predicates with semicolons or pack several arguments' requirements onto one
line. Conditions describing a single argument, such as
`accuracy >= 0 && std::isfinite(accuracy)`, can stay together.

## Prefer requirements enforced by types

The best way to express a requirement is through the type system, when possible,
so that misuse is a build error rather than a runtime error. Prefer an interface
that makes invalid combinations unrepresentable over one that merely lists them
in a comment. For example, distinct result types should expose only the fields
appropriate to their outcome. A tool that needs an analyzed sparsity pattern can
accept an immutable analysis object instead of unrelated arrays that callers
must keep synchronized.

Types do not eliminate all runtime requirements: a vector's length or a numerical
factorization's success may only be known at runtime. Validate those conditions
at the appropriate boundary and make failure an explicit outcome. Do not add
wrapper types that merely move the same unchecked obligation elsewhere.

## Require only what callers can guarantee

Preconditions concern conditions under the caller's control. Never require that
file reads will succeed, memory will be available, or unchecked floating-point
arithmetic will remain finite. Checking a file's existence does not guarantee a
successful read; finite operands alone do not guarantee finite arithmetic results.
A finite-argument precondition is appropriate when the caller can check that
argument before the call and handle a nonfinite value without making the call.

If a condition cannot be a precondition, make it part of the specified outcome.
For example, a file-reading function should return an error when the read fails,
or specify the exception it throws and the resulting state. A numerical tool
should report arithmetic overflow or failure to achieve accuracy. Describe normal
outcomes in `ensures` and exceptional exits in `throws`; neither is a caller bug.
Do not move an impossible requirement into a callback contract or a lower layer.

## Use only caller visible concepts in contracts

An interface contract must refer to its arguments, public observations, and
defined mathematical concepts, not private representation. For example, suppose
a solver accepts a vector `rhs` and internally caches another vector in a private
member named `border`. The following contract is bad because the caller cannot
observe `border`, and because "agrees with the model" has no precise definition:

```cpp
requires
    - rhs.size() == border.size()
    - the cached border agrees with the model
```

If the solver has a public `dimension()` observation giving the required vector
length, `rhs.size() == dimension()` expresses the size requirement precisely.
But adding accessors for every private field is not the objective. If a hidden
cache must track changing inputs, consider computing it inside the operation or
binding it to immutable data. Remove synchronization duties that the caller
cannot reliably fulfill.

Prefer read-only value observations when callers do not need to mutate the
representation.

Representation invariants belong in an implementation `CONVENTION` block.
They explain how private data realizes the public abstraction; they are not
extra, undocumented requirements on callers. If a helper cannot be specified
without private details from another component, reconsider the helper's boundary.

## Audit callers as well as callees

For each call, identify what establishes every precondition: input validation,
construction, an earlier successful operation, a loop invariant, or an explicit
check. Follow exceptional and early-return paths too. Passing tests alone does
not justify an obligation that the caller never establishes.

Use `DLIB_ASSERT` or `DLIB_CASSERT` to check preconditions as described in the
contribution guidelines. Assertions should match the documented predicates,
not introduce hidden requirements. Prefer owning containers or explicit views
over unchecked pointer-and-length combinations. Make borrowed lifetimes and
invalidation rules explicit; redesign ownership when they are hard to guarantee.

Callback contracts need lifetime and reentrancy rules too. For example, a
workspace can own a temporary vector and lend it to a callback only for that
call. Specify whether the callback may retain references, resize the vector,
change other arguments, or call back into the workspace. This lets
the owner establish nonaliasing instead of asking callers to reason about hidden
scratch buffers.

When specifying requirements on a template parameter, explicitly identify what
the type must provide. For example, write "Observer must provide public member
functions charge() and storage()" before describing their arguments and behavior.
Do not leave the reader to infer whether a listed name denotes a member function,
a free function, a data member, or a nested type.
Describe callable arguments by their invocation, such as `callback(value)`,
without requiring a member function when an ordinary function or function pointer
is also accepted. Document a required nested type separately from required methods.

A reader must be able to implement that protocol from its contract. Name each
method's arguments and explain their meaning, units, whether a quantity is an
increment or a total, when the method is called, and what returning or throwing
does. For example, `charge(units)` can report the estimated work of the next batch,
whereas `storage(bytes)` can report a total workspace requirement. "Admit work"
does not define either protocol. If calculating those estimates requires knowing
private array layouts or algorithmic cost weights, the library should calculate
them rather than requiring the observer to reconstruct them. Keep test
instrumentation out of the required public interface when it can be supplied
independently.

Name the actual callable or observer instance in exception clauses, e.g.
"Exceptions from `evaluate(x)` propagate", rather than an undefined "Callback".

Include callback construction and copying in that analysis. Passing a callback
by value can allocate or throw even if invoking it cannot. When no ownership is
needed, a const-callable reference can avoid those copies and admit noncopyable
callbacks. State that the reference is not retained after the operation.

If a caller cannot guarantee a requirement, do not merely document it and carry
on. Change the API, check and handle the condition, bind the necessary state at
construction, or make the operation support a wider set of inputs.

## Describe observable results and failure states

An `ensures` clause describes what the operation establishes on normal return.
Use `#expression` for the value after the call and the unprefixed expression for
its value on entry, following dlib's notation. For example, an operation that
inserts one element into a container with a public `size()` observation could
include this postcondition:

```cpp
ensures
    - #size() == size() + 1
```

Specify which outputs change and what remains usable after a declined operation
or exception. A factorization that fails halfway through must not leave an old
success flag inviting callers to use partial factors. State whether an operation
is transactional, retains a partial result, or invalidates previously usable state.

Use typed outcomes when different results have different meaningful fields. In
particular, use `type_safe_union` instead of an all-purpose result structure whose
documentation says that some fields are meaningless for some statuses. Dlib's
`type_safe_union` holds at most one of its specified alternative types at a time.
For example, a parser can return either a parsed value or an error description,
without presenting an error as a result that also contains a meaningless value.

When an operation produces diagnostics that describe its result, prefer returning
them with that result rather than returning a boolean and requiring a separate
"last result" query. A boolean can hide important distinctions, such as a finite
approximation versus one that meets the requested accuracy. A returned report
also belongs to a particular call; later operations cannot silently overwrite it.
Use stored state when it is part of the abstraction, not merely to transport an
operation's outputs to its caller.

Do not expose an entire scratch buffer when only part of it is a meaningful
result. Return a value or a bounded view containing just the result, and specify
the view's lifetime and invalidation rules. Likewise, keep temporary quantities
local instead of publishing them as mutable members. The caller should not have
to know which fields are scratch, which have been initialized, or which must be
kept synchronized.

Do not pad simple accessor contracts with "does not allocate or modify state".
State what they return, plus any relevant lifetime or invalidation rule. Resource
guarantees are worth stating on operations where they matter, such as a
preallocated factorization or solve; they are not boilerplate for every function.

## Separate invalid use from numerical outcomes

Malformed external input needs validation; it is not automatically a programming
bug. Likewise, failure to converge, singularity, range loss during arithmetic,
or inability to meet an accuracy target must be handled as documented numerical
outcomes when they can arise from supported inputs. Do not hide them behind a
requirement that the algorithm will succeed.

A numerical contract must say what was computed and what its diagnostics mean.
A residual measures the error obtained by substituting an approximate solution
into the equations being solved. Returning finite numbers, meeting a specified
residual tolerance, satisfying an optimizer's acceptance criteria, and providing
a mathematically exact proof of a result are different guarantees. Do not call
an estimate of rounding error a guaranteed interval bound or claim exact
arithmetic for a compensated floating-point accumulator.

If regularization changes the system being factored, for example by adding terms
to its diagonal to improve numerical stability, specify the changed system and
whether the solve tests the residual of the original or changed equations. Do not
let a general matrix API silently inherit application-specific criteria such as
"accurate enough for the optimizer's next iteration".

## Organize classes and specifications in dlib style

Use C++17 nested namespace definitions, such as `namespace dlib::lp_detail`,
instead of opening a separate block for each namespace. Indent the contents by
one level within the combined namespace block.

Use `static_assert` for compile-time checks, not the legacy `COMPILE_TIME_ASSERT`
macro.

Put `/*!` on its own line, with the comment's contents starting on the next
line, indented one level. Never put `requires`, `ensures`, or
`WHAT THIS OBJECT REPRESENTS` on the opening line. This applies even to short
accessor contracts: do not compress them into `/*! ensures ... !*/`. Put the
closing `!*/` on its own line, aligned with the opening delimiter. For example,
consider an accessor for a container of elements of type `T`, with zero-based
indices and a public `size()` observation:

```cpp
    const T& operator[](std::size_t index) const;
    /*!
        requires
            - index < size()
        ensures
            - Returns the entry at index.
    !*/
```

Indent a bullet list one level farther than the heading or sentence introducing
it. Align continuation lines with the text after the bullet, not with the dash.

Keep each function declaration and its contract together, then leave a blank
line before the next function declaration. Likewise, leave a blank line between
function definitions. Short accessors are not an exception. For example, a
container's public interface could include:

```cpp
    std::size_t size() const;
    /*!
        ensures
            - Returns the number of elements in this container.
    !*/

    bool empty() const;
    /*!
        ensures
            - Returns true if and only if size() == 0.
    !*/
```

Open a class with its `WHAT THIS OBJECT REPRESENTS` description. Explain the
abstraction, not its private data structures. Use `INITIAL VALUE` when useful.
Put the public interface first and the private members last. For example:

```cpp
class example_tool
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            Describe the mathematical or operational concept here.
    !*/

public:
    // Public operations and their contracts.

private:
    // Representation convention, helpers, and storage.
};
```

Public library components have separate specification and implementation files,
following the neighboring component's organization. General matrix operations
and decomposition APIs belong with the matrix tools, with their public contracts
in `matrix/matrix_la_abstract.h` where appropriate. Clients should use those C++
APIs, not raw bindings to optional numerical libraries such as BLAS or LAPACK.

Use spaces, descriptive `lower_case_with_underscores` names, and the brace and
declaration style of nearby dlib code.

Put separate statements on separate lines. Readability matters in numerical
code, including tests and small helpers. Use resource-owning objects and
containers that release their resources automatically (RAII);
retain the appropriate copyright and Boost Software License notices.

## Contain backend and platform details

Avoid preprocessor switches in algorithms and high-level callers. Header guards
and dlib's assertion conventions still apply. Where backend or compiler selection
is necessary, contain it inside the focused tool that needs it.

The tool's C++ contract must make sense without knowing which backend is selected.
Provide a self-contained implementation when optional BLAS or LAPACK acceleration
is unavailable. The optimized implementation must satisfy the same contract;
different floating-point results or pivot choices are allowed only within that
contract. Do not imply that vendor-library allocations are controlled by dlib.

## Test abstractions independently and preserve callers

Give each extracted tool tests expressed in its own domain. A matrix solver test
should build matrices and right-hand sides, not an artificial LP. Test empty
inputs, boundaries, aliasing where supported, repeated use, failed-operation
recovery, and numerical cases relevant to the contract. Check residuals or known
answers independently rather than calling the implementation's own acceptance
test as the only oracle.

When extracting a tool from an existing algorithm, also compare the old and new
versions of that algorithm on retained regression inputs, including outcomes,
outputs, and resource counters where applicable. Use assertions and sanitizers,
test native and accelerated builds, and time both versions on the same inputs
under the same conditions. For solvers, count incomplete solves explicitly;
early failure is not a speed improvement. These checks support the refactor but
do not prove correctness on every possible input.
Inspect per-case timing changes as well as aggregates: an unchanged average can
hide a regression on a particular workload. Repeat suspicious measurements
before attributing them to either the code or measurement noise.

For a sequence of extractions, retain the original baseline and check the final
composition against it too. Small costs that look harmless at each step can
accumulate. Instrumented profiling helps locate those costs, but its overhead
makes it unsuitable as the final performance comparison; repeat that comparison
without instrumentation.

Preserve arithmetic order and existing numerical policy during extraction. If a
broader API requires new behavior, identify and test that change separately.
Prefer small reviewable steps over combining extraction, new algorithms, and
performance tuning in the same change.

## Review checklist and how this guide evolves

Before considering a component ready:

- Can its purpose be explained without another component's private details?
- Can a reader understand the documentation and examples without private
  discussions or development history? Are the necessary concepts introduced?
- Can the reader explain the operation as well as check its predicates? Are
  input representations, output mappings, and specialized terms defined?
- Does every function and class contract make its purpose and productive use
  clear, without relying on implementation details or adding motivational
  boilerplate?
- Does each nontrivial contract open with an intuitive overview before giving
  its detailed conditions, with the necessary terminology introduced?
- Is the task or objective clear, together with any constraints the result must
  satisfy? Are mandatory constraints distinguished from heuristic aims and
  preferences?
- For an overload, is the effect of each added argument on the result clear?
- Can callers choose every public parameter from their problem requirements or
  resource budget, rather than by guessing what makes the implementation work?
  Have unexplained tuning knobs been removed rather than renamed or repackaged?
- Are independent interface concepts defined separately before their rules are
  combined? Are internal safeguards the implementation's responsibility?
- Does each diagnostic explain its significance and what the caller can do
  with it, without confusing a warning with failure or an accuracy guarantee?
- Can its correctness be reasoned about using only its contract and its callees'
  contracts, independently of the rest of the system?
- Is every requirement exact, observable, and justified at every call site?
- Can the caller actually guarantee it, including on exceptional paths?
- Could types rule out invalid uses instead of requiring comment-only obligations?
- Could construction, ownership, or a wider supported input domain remove a
  difficult requirement?
- Do normal return, numerical failure, and exceptional exits have clear effects?
- Are result types appropriate for each outcome?
- Are simple contracts concise, without empty requirements or accessor boilerplate?
- Does the contract describe the component itself, without repeating its signature,
  general C++ rules, header-inclusion instructions, or future-version disclaimers?
- Do comment contents start on the line after `/*!`, with `!*/` on its own line?
- Is each function and its contract separated from the next function by a blank line?
- Do public interfaces precede private members, with an opening class description?
- Are platform switches hidden inside the relevant low-level tool?
- Can a reusable component compile and be tested without its original application?
- Do integration and performance checks support preservation of caller behavior?

When a review reveals a gap, add the smallest general rule and a concrete example
that would have prevented it. Then apply that rule to an actual component and
audit its callers. Record remaining work in the implementation notes, not as a
claim that an unfinished component already satisfies this guide.

Test substantive guide changes with independent reviewers who receive the guide
and a small API example, without the discussion that motivated the change. Ask
them to propose both an interface and its contract, and to explain how a caller
would choose each parameter and interpret failure. Include contrasting cases:
an unexplained numerical pivot cutoff should disappear, while a distance error
in meters or an explicit work allowance should remain meaningful. Check the
resulting design, not whether the reviewer can repeat the rule. If a reviewer
merely renames the knob or promises impossible numerical success, revise the
guidance and repeat the exercise.
