# PEtab timecourse
This is an example implementation of an extension of PEtab, for the specification of time-dependent conditions.

# Installation
Clone this repository then install it into your Python (virtual) environment.
```bash
git clone --recurse-submodules https://github.com/dilpath/petab_timecourse
cd petab_timecourse
pip install -e .[examples]
```

# Examples
The examples depend on [AMICI](https://github.com/AMICI-dev/AMICI) for simulation and [pyPESTO](https://github.com/ICB-DCM/pyPESTO) for optimization, but these are independent of the PEtab extension.


# File formats
## Timecourse table
TODO

## Timecourse parameters table
WIP

- all columns from normal PEtab parameters table
- `timecourseIds`
- `conditionIds`
- `type`
  - `time` to estimate the start time of the period
    - then ignore `parameterId` etc?!
  - `value` to estimate the value that the parameter takes during the period

### TODO
How to specify parameter estimation problem when estimating time?
- a lot of possible flexibility...
  - use `objectivePrior...` to apply constraints to the values that each estimated time point can take
  - for consecutively estimated time periods
    - the lower bound of the next period should match the upper bound of the previous period
  
