# Prototypes

Candidate Optora components that a reproduction needs and Optora does not
ship yet. The first one planned is the continuous-adversary Wasserstein
inner maximization of Sinha, Namkoong and Duchi (ICLR 2018), in
`continuous_wasserstein/`.

A prototype lives here until a reproduction has shown which API it really
needs. It is then proposed upstream as an Optora issue, implemented in
Optora with Optora's own test suites, and the prototype is archived once
the reproduction imports the released version.

Each prototype keeps its own tests in `tests/prototypes/` so the later
Optora implementation can be checked against it.
