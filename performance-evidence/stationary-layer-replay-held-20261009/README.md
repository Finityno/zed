# Held stationary layer replay tracking

One frozen comparison of 1,624 native children across 58 complete App/Window
workloads passes both allocation targets but fails 14 ordinary CPU controls.
Retained redraws with 200/512 rows use 7.13/6.63% fewer allocation calls and
0.100/0.081% fewer requested bytes. Held/peak requested heap is unchanged.
Every requested-memory, allocation-churn and native RSS control passes.
Do not resample this design unchanged or weaken its gates. No PR, merge or
consumer adoption is qualified.

Stationary replay needs no layer-omission stack; moved clipping is unchanged.
No App fields, timers, caches or renderer resources are added. Thirty-six
native contract invocations pass, including nested/partial/moved layers,
retained output, wheel/dependency changes, panels, layout/atomic claims,
read/nested/idle, existing Scene/App/Subscriber and independent resize checks.
The private fully warmed allocation contract fails before and passes after;
all 1,624 paired scene, hitbox and ownership outputs match.

All raw samples, frozen method, exact source/compiler/runtime/common 59 native
externs/binary identities and sequential process-group receipts are archived
and every member hash verified. Thirty-six complete-component attribution
children and both original setup failures remain preserved. The first test
helper used unsupported Debug; the second warmed only one of two alternating
bounds-tree buffers. All private corrections preceded any performance sample
and left production behavior and performance fixtures/gates unchanged.
Original failed binaries remain losslessly preserved and hash identified.

Linux fake-platform/System component measurements do not establish app,
physical GPU, macOS/Windows or shipped-allocator gains. Timing excludes setup
and four warmups; census includes complete owners and 128 stimuli, frozen
before the candidate. Continue with a distinct important ownership design.
